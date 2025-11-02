import json
import re
from typing import Dict, List, Set, Tuple
from tqdm import tqdm
import sqlparse
from llm import LLM_generation  


class CoTGenerator:
    def __init__(self):
        """Initialize the CoT generator"""
        pass
        
    def create_cot_prompt(self, question: str, database_schema: str, 
                          ground_truth_sql: str) -> str:
        """
        Create prompt for LLM to generate CoT for schema linking.
        As per Equation 2 in the paper, we provide ground-truth SQL.
        """
        prompt = f"""You are an expert in database schema linking for Text-to-SQL tasks.

Given a natural language question and a database schema, identify which tables and columns are needed to answer the question.

**Database Schema:**
{database_schema}

**Question:** {question}

**Ground Truth SQL (for reference):**
{ground_truth_sql}

Please provide your reasoning in the following format:

****
1. Understand the key concepts in the question:
   • [Identify key phrases in the question]
   • [Map them to what they mean in database terms]
   • [Note what operations are required]

2. Analyze database table relationships:
   • [Identify which tables contain relevant information]
   • [Explain the relationships between tables using foreign keys]
   • [Note how tables need to be joined]

3. Key field for filtering: **[main_table.key_field]** ([explain why this field is critical])
   [Additional explanation of why this field is most relevant]
****

**[Provide a summary paragraph explaining the reasoning, emphasizing the most critical field(s) for answering the question. End with:]

The key field matching the question is: [table.column].**

Example format:
****
1. Understand the key concepts in the question:
   • `heads of the departments`: Find the head of each department (head_ID in head table)
   • `older than 56`: corresponds to the age column in the head table
   • `How many`: Count the number of department heads meeting the criteria

2. Analyze database table relationships:
   • Head information stored in the head table
   • Head-department relationship in management table (management.head_ID ↔ head.head_ID)
   • Department info in department table (management.department_ID ↔ department.Department_ID)

3. Key field for filtering: **head.age** (determines if > 56)
   The head table stores personal information directly related to age filtering.
****

**The key field for determining whether the person in charge is older than 56 is head.age, as the head table stores the relevant personal information. The management table links the person in charge with the department, but it does not directly provide the age information. Therefore, head.age is the most directly related field for answering this question. The key field matching the question is: [head.age].**
"""
        return prompt
    
    def generate_cot(self, question: str, database_schema: str, 
                     ground_truth_sql: str) -> Dict:
        """Generate CoT using LLM_generation function"""
        prompt = self.create_cot_prompt(question, database_schema, ground_truth_sql)
        instruct = "You are a database schema linking expert. Follow the output format exactly as specified."
        
        try:
            results = LLM_generation(instruct, prompt, n=1)
            cot_output = results[0] if results else ""
            
            return {
                "success": True,
                "cot": cot_output
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def parse_cot_output(self, cot_text: str) -> Tuple[Set[str], Set[str], List[str]]:
        """
        Parse CoT output to extract predicted tables, columns, and key fields.
        Returns: (set of tables, set of columns, list of key fields)
        """
        tables = set()
        columns = set()
        key_fields = []
        
        key_field_pattern = r'The key field matching the question is:\s*\[(.*?)\]'
        key_match = re.search(key_field_pattern, cot_text, re.IGNORECASE)
        if key_match:
            key_field_str = key_match.group(1)
            key_fields = [kf.strip() for kf in key_field_str.split(',')]
        
        filtering_pattern = r'Key field for filtering:\s*\*\*(.*?)\*\*'
        filter_matches = re.findall(filtering_pattern, cot_text, re.IGNORECASE)
        for match in filter_matches:
            field = match.strip()
            if field and field not in key_fields:
                key_fields.append(field)
        
        column_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)\b'
        column_matches = re.findall(column_pattern, cot_text)
        for col in column_matches:
            columns.add(col.lower())
            table = col.split('.')[0].lower()
            tables.add(table)
        
        table_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s+table'
        table_matches = re.findall(table_pattern, cot_text, re.IGNORECASE)
        for table in table_matches:
            tables.add(table.lower())
        
        return tables, columns, key_fields
    
    def extract_sql_entities(self, sql: str) -> Tuple[Set[str], Set[str]]:
        """
        Extract tables and columns used in the SQL query.
        Returns: (set of tables, set of columns in table.column format)
        """
        parsed = sqlparse.parse(sql)[0]
        tables = set()
        columns = set()
        
        def extract_from_token(token):
            if token.ttype is None:
                if hasattr(token, 'tokens'):
                    for t in token.tokens:
                        extract_from_token(t)
            else:
                token_str = str(token).strip()
                if '.' in token_str:
                    columns.add(token_str.lower())
        
        from_seen = False
        for token in parsed.tokens:
            if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'FROM':
                from_seen = True
            elif from_seen and token.ttype is None:
                table_name = str(token).strip().split()[0]
                if table_name.upper() not in ['WHERE', 'GROUP', 'ORDER', 'LIMIT', 'JOIN']:
                    tables.add(table_name.lower())
                from_seen = False
            
            extract_from_token(token)
        
        return tables, columns
    
    def validate_cot_format(self, cot_output: str) -> Tuple[bool, str]:
        """
        Validate CoT output format compliance.
        Returns: (is_valid, error_message)
        """
        if cot_output.count('****') < 2:
            return False, "Missing **** markers (need at least 2)"
        
        if not re.search(r'1\.\s*Understand the key concepts in the question', cot_output, re.IGNORECASE):
            return False, "Missing Step 1: 'Understand the key concepts in the question'"
        
        if not re.search(r'2\.\s*Analyze database table relationships', cot_output, re.IGNORECASE):
            return False, "Missing Step 2: 'Analyze database table relationships'"
        
        if not re.search(r'3\.\s*Key field for filtering', cot_output, re.IGNORECASE):
            return False, "Missing Step 3: 'Key field for filtering'"
        
        if not re.search(r'The key field matching the question is:', cot_output, re.IGNORECASE):
            return False, "Missing final declaration: 'The key field matching the question is:'"
        
        return True, "Format is valid"
    
    def validate_cot(self, cot_output: str, ground_truth_sql: str, 
                     predicted_label: str, ground_truth_label: str, 
                     stats: Dict = None) -> bool:
        """
        Validate CoT output:
        1. Final answer must match ground-truth label
        2. Extracted entities must be consistent with SQL query
        3. Output must follow the required format
        """
        format_valid, error_msg = self.validate_cot_format(cot_output)
        if not format_valid:
            if stats and 'format_errors' in stats:
                if "Step 1" in error_msg:
                    stats['format_errors']['missing_step1'] += 1
                elif "Step 2" in error_msg:
                    stats['format_errors']['missing_step2'] += 1
                elif "Step 3" in error_msg:
                    stats['format_errors']['missing_step3'] += 1
                elif "key field" in error_msg.lower():
                    stats['format_errors']['missing_key_field'] += 1
                elif "****" in error_msg:
                    stats['format_errors']['missing_markers'] += 1
            print(f"Format validation failed: {error_msg}")
            return False
        
        if predicted_label != ground_truth_label:
            if stats:
                stats['label_mismatch'] += 1
            print(f"Label mismatch")
            return False
        
        try:
            cot_tables, cot_columns, key_fields = self.parse_cot_output(cot_output)
            sql_tables, sql_columns = self.extract_sql_entities(ground_truth_sql)
            
            if not key_fields:
                if stats:
                    stats['entity_inconsistent'] += 1
                print(f"No key fields extracted")
                return False
            
            if cot_tables and not any(t in sql_tables for t in cot_tables):
                if stats:
                    stats['entity_inconsistent'] += 1
                print(f"Entity inconsistency")
                return False
            
            return True
        except Exception as e:
            if stats:
                stats['entity_inconsistent'] += 1
            print(f"Validation error: {e}")
            return False
    
    def process_dataset(self, dataset_path: str, output_path: str):
        """
        Process entire dataset and generate filtered CoT training data.
        Implements the filtering process described in Equations 2-4.
        """
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        filtered_data = []
        stats = {
            "total": len(dataset),
            "generated": 0,
            "validated": 0,
            "failed": 0,
            "format_errors": {
                "missing_step1": 0,
                "missing_step2": 0,
                "missing_step3": 0,
                "missing_key_field": 0,
                "missing_markers": 0
            },
            "label_mismatch": 0,
            "entity_inconsistent": 0
        }
        
        for item in tqdm(dataset, desc="Generating CoT"):
            question = item['question']
            schema = item['schema']
            ground_truth_sql = item['sql']
            
            # Auto-generate label from SQL
            sql_tables, sql_columns = self.extract_sql_entities(ground_truth_sql)
            ground_truth_label = json.dumps({
                "tables": sorted(list(sql_tables)),
                "columns": sorted(list(sql_columns))
            })
            
            result = self.generate_cot(question, schema, ground_truth_sql)
            
            if not result['success']:
                stats['failed'] += 1
                continue
            
            stats['generated'] += 1
            cot = result['cot']
            
            predicted_label = self.extract_label_from_cot(cot)
            
            if self.validate_cot(cot, ground_truth_sql, predicted_label, ground_truth_label, stats):
                filtered_data.append({
                    'question': question,
                    'schema': schema,
                    'cot': cot,
                    'label': ground_truth_label,
                    'sql': ground_truth_sql
                })
                stats['validated'] += 1
        
        with open(output_path, 'w') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"Processing Statistics:")
        print(f"{'='*60}")
        print(f"Total samples:           {stats['total']}")
        print(f"Successfully generated:  {stats['generated']}")
        print(f"Validated and saved:     {stats['validated']}")
        print(f"Failed to generate:      {stats['failed']}")
        print(f"\nValidation Breakdown:")
        print(f"  Label mismatch:        {stats['label_mismatch']}")
        print(f"  Entity inconsistent:   {stats['entity_inconsistent']}")
        print(f"\nFormat Error Breakdown:")
        print(f"  Missing **** markers:  {stats['format_errors']['missing_markers']}")
        print(f"  Missing Step 1:        {stats['format_errors']['missing_step1']}")
        print(f"  Missing Step 2:        {stats['format_errors']['missing_step2']}")
        print(f"  Missing Step 3:        {stats['format_errors']['missing_step3']}")
        print(f"  Missing key field:     {stats['format_errors']['missing_key_field']}")
        print(f"\nSuccess Rate:")
        if stats['generated'] > 0:
            print(f"  Validation rate:       {stats['validated']/stats['generated']*100:.2f}%")
        print(f"  Overall success:       {stats['validated']/stats['total']*100:.2f}%")
        print(f"{'='*60}")
    
    def extract_label_from_cot(self, cot: str) -> str:
        """Extract the predicted label (schema linking result) from CoT"""
        tables, columns, key_fields = self.parse_cot_output(cot)
        return json.dumps({
            "tables": sorted(list(tables)), 
            "columns": sorted(list(columns)),
            "key_fields": key_fields
        })


def main():
    """Main execution function"""
    INPUT_DATASET = "data.json"
    OUTPUT_DATASET = "data_cot.json"
    
    generator = CoTGenerator()
    
    generator.process_dataset(INPUT_DATASET, OUTPUT_DATASET)


if __name__ == "__main__":

    main()
