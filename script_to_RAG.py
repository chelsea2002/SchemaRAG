import json
import re
from typing import Dict, List, Tuple, Set
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where, Comparison
from sqlparse.tokens import Keyword, DML
from tqdm import tqdm
from llm import LLM_generation
from anthropic import Anthropic


class RAGDataValidator:
    """
    RAG Data Generator and Validator Implementation
    Uses LLM_generation for generation, Claude API for validation
    """

    def __init__(self, api_key: str = None):
        """
        Initialize the validator with Claude API
        
        Args:
            api_key: Anthropic API key for validation (if None, will try to read from environment)
        """
        # Claude API client for validation only
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-sonnet-4-20250514"  # Claude 4 Sonnet for validation

    def generate_rag_triple(self, question: str, sql: str, schema: str, database_name: str) -> Dict:
        """
        Generate structurally similar Question-SQL-Schema triples using LLM_generation
        
        Args:
            question: Original question
            sql: Original SQL query
            schema: Database schema
            database_name: Database name
            
        Returns:
            Generated triple dictionary
        """
        prompt = f"""Given the following database schema and a question-SQL pair, generate a NEW question-SQL pair that:

1. Uses the SAME database schema
2. Has SIMILAR SQL structure (same type of joins, aggregations, filtering patterns)
3. Asks a DIFFERENT semantic question
4. Produces valid, executable SQL

Database Schema:
{schema}

Original Question: {question}

Original SQL: {sql}

Generate a new question and corresponding SQL query that maintains structural similarity but asks a different question.

Respond in JSON format:
{{
    "new_question": "...",
    "new_sql": "..."
}}"""

        instruct = "You are an expert in database query generation. Generate a new question-SQL pair following the specified requirements and output format."
        
        try:
            # Use LLM_generation for generation task
            results = LLM_generation(instruct, prompt, n=1)
            response_text = results[0] if results else ""
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "question": result["new_question"],
                    "sql": result["new_sql"],
                    "schema": schema,
                    "database": database_name,
                    "original_question": question,
                    "original_sql": sql
                }
            else:
                raise ValueError("Failed to extract JSON from response")
        except Exception as e:
            raise ValueError(f"Failed to generate RAG triple: {e}")

    def extract_tables_columns_from_sql(self, sql: str) -> Tuple[Set[str], Set[str]]:
        """
        Extract table names and column names from SQL query
        
        Args:
            sql: SQL query statement
            
        Returns:
            (tables, columns) Set of table names and column names
        """
        tables = set()
        columns = set()

        try:
            parsed = sqlparse.parse(sql)[0]

            # Extract table names
            from_seen = False
            for token in parsed.tokens:
                if from_seen:
                    if isinstance(token, IdentifierList):
                        for identifier in token.get_identifiers():
                            table_name = identifier.get_real_name()
                            if table_name:
                                tables.add(table_name.lower())
                    elif isinstance(token, Identifier):
                        table_name = token.get_real_name()
                        if table_name:
                            tables.add(table_name.lower())
                    from_seen = False

                if token.ttype is Keyword and token.value.upper() in ('FROM', 'JOIN'):
                    from_seen = True

            # Extract column names (simplified version)
            # Use regex to extract table.column format
            column_pattern = r'(\w+)\.(\w+)'
            matches = re.findall(column_pattern, sql.lower())
            for table, column in matches:
                columns.add(f"{table}.{column}")

        except Exception as e:
            print(f"Error parsing SQL: {e}")

        return tables, columns

    def check_structural_consistency(self, original_sql: str, generated_sql: str) -> bool:
        """
        Check SQL structural consistency
        
        Args:
            original_sql: Original SQL
            generated_sql: Generated SQL
            
        Returns:
            Whether structurally consistent
        """
        try:
            orig_parsed = sqlparse.parse(original_sql)[0]
            gen_parsed = sqlparse.parse(generated_sql)[0]

            # Check number and type of keywords
            orig_keywords = [t.value.upper() for t in orig_parsed.tokens if t.ttype is Keyword]
            gen_keywords = [t.value.upper() for t in gen_parsed.tokens if t.ttype is Keyword]

            # Check if main SQL components are similar
            critical_keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'JOIN', 'HAVING']
            orig_components = set(k for k in orig_keywords if k in critical_keywords)
            gen_components = set(k for k in gen_keywords if k in critical_keywords)

            # Structural similarity: at least 80% of critical components match
            if len(orig_components) == 0:
                return False

            similarity = len(orig_components & gen_components) / len(orig_components)
            return similarity >= 0.8

        except Exception as e:
            print(f"Error checking structural consistency: {e}")
            return False

    def check_semantic_matching(self, question: str, sql: str, schema: str) -> Tuple[bool, str]:
        """
        Use Claude API to check semantic matching between question and SQL
        
        Args:
            question: Natural language question
            sql: SQL query
            schema: Database schema
            
        Returns:
            (is_matched, explanation) Whether semantically matched and explanation
        """
        prompt = f"""Given the following database schema, question, and SQL query, determine if the SQL query correctly answers the question.

Database Schema:
{schema}

Question: {question}

SQL Query: {sql}

Does this SQL query correctly and completely answer the question? Consider:
1. Are all necessary tables and columns included?
2. Are the JOIN conditions correct?
3. Are the WHERE conditions appropriate?
4. Is the aggregation/grouping correct?
5. Does the query return the information asked for in the question?

Respond with ONLY "YES" or "NO" followed by a brief explanation."""

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0,
                system="You are a database expert. Validate whether the SQL query correctly answers the given question.",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            response = message.content[0].text.strip()
            
            # Check if response starts with YES
            is_matched = response.upper().startswith("YES")
            return is_matched, response
            
        except Exception as e:
            print(f"Error in semantic matching check: {e}")
            return False, f"Error: {str(e)}"

    def validate_rag_triple(self, triple: Dict, original_sql: str) -> Tuple[bool, Dict]:
        """
        Validate RAG triple
        
        Args:
            triple: Generated triple
            original_sql: Original SQL (for structural comparison)
            
        Returns:
            (is_valid, validation_details) Whether valid and validation details
        """
        validation_details = {
            "structural_consistent": False,
            "semantic_matched": False,
            "semantic_explanation": "",
            "valid": False
        }

        # 1. Structural consistency check
        print("Checking structural consistency...")
        structural_check = self.check_structural_consistency(
            original_sql,
            triple["sql"]
        )
        validation_details["structural_consistent"] = structural_check

        if not structural_check:
            print("  ✗ Structural consistency check failed")
            return False, validation_details

        print("  ✓ Structural consistency check passed")

        # 2. Semantic matching check using Claude API
        print("Checking semantic matching with Claude API...")
        semantic_check, explanation = self.check_semantic_matching(
            triple["question"],
            triple["sql"],
            triple["schema"]
        )
        validation_details["semantic_matched"] = semantic_check
        validation_details["semantic_explanation"] = explanation

        if not semantic_check:
            print("  ✗ Semantic matching check failed")
            print(f"  Explanation: {explanation}")
            return False, validation_details

        print("  ✓ Semantic matching check passed")
        print(f"  Explanation: {explanation[:100]}...")

        # Both checks passed
        validation_details["valid"] = True
        print("  ✓ All validation checks passed")
        return True, validation_details

    def process_dataset(self, input_file: str, output_file: str, max_samples: int = None) -> Dict:
        """
        Process entire dataset, generate and validate RAG triples
        
        Args:
            input_file: Input JSON file path
            output_file: Output JSON file path
            max_samples: Maximum number of samples to process (None means process all)
            
        Returns:
            Statistics dictionary
        """
        # Load original data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if max_samples:
            data = data[:max_samples]

        validated_triples = []
        stats = {
            "total": len(data),
            "generated": 0,
            "structural_passed": 0,
            "semantic_passed": 0,
            "final_validated": 0,
            "failed": 0
        }

        for idx, item in enumerate(tqdm(data, desc="Processing dataset")):
            print(f"\n{'='*60}")
            print(f"Processing sample {idx+1}/{len(data)}")
            print(f"{'='*60}")

            try:
                # Generate RAG triple
                print("Generating RAG triple...")
                triple = self.generate_rag_triple(
                    question=item["question"],
                    sql=item["query"],
                    schema=item["schema"],
                    database_name=item.get("db_id", "unknown")
                )
                stats["generated"] += 1
                print(f"  Generated question: {triple['question'][:100]}...")

                # Validate triple
                is_valid, validation_details = self.validate_rag_triple(
                    triple,
                    item["query"]
                )

                # Update statistics
                if validation_details["structural_consistent"]:
                    stats["structural_passed"] += 1
                if validation_details["semantic_matched"]:
                    stats["semantic_passed"] += 1

                if is_valid:
                    triple["validation"] = validation_details
                    validated_triples.append(triple)
                    stats["final_validated"] += 1
                    print(f"  ✓ Triple validated and saved")
                else:
                    stats["failed"] += 1
                    print(f"  ✗ Triple validation failed")

            except Exception as e:
                print(f"  ✗ Error processing sample: {e}")
                stats["failed"] += 1
                continue

        # Save validated data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(validated_triples, f, indent=2, ensure_ascii=False)

        # Print statistics
        print(f"\n{'='*60}")
        print("Processing Complete - Statistics:")
        print(f"{'='*60}")
        print(f"Total samples: {stats['total']}")
        print(f"Successfully generated: {stats['generated']}")
        print(f"Structural consistency passed: {stats['structural_passed']} "
              f"({stats['structural_passed']/stats['total']*100:.1f}%)")
        print(f"Semantic matching passed: {stats['semantic_passed']} "
              f"({stats['semantic_passed']/stats['total']*100:.1f}%)")
        print(f"Final validated: {stats['final_validated']} "
              f"({stats['final_validated']/stats['total']*100:.1f}%)")
        print(f"Failed: {stats['failed']}")
        print(f"{'='*60}")

        return stats


def main():
    """
    Main function: Demonstrate RAG data generation and validation workflow
    """
    # Initialize validator
    # Generation uses LLM_generation (no key needed)
    # Validation uses Claude API (pass key or use env variable ANTHROPIC_API_KEY)
    validator = RAGDataValidator(api_key="your-api-key-here")  # Or leave None to use env variable

    datasets = {
        "spider": {
            "input": "spider_dev.json",
            "output": "RAG_spider.json"
        },
        "bird": {
            "input": "bird_dev.json",
            "output": "RAG_bird.json"
        }
    }

    # Process each dataset
    for dataset_name, paths in datasets.items():
        print(f"\n{'='*80}")
        print(f"Processing {dataset_name.upper()} dataset")
        print(f"{'='*80}\n")

        stats = validator.process_dataset(
            input_file=paths["input"],
            output_file=paths["output"],
            max_samples=10  # Use small sample for testing, set to None for production
        )

        # Save statistics
        stats_file = paths["output"].replace(".json", "_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
