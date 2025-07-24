import sqlite3
import sqlparse
from sqlparse.sql import Statement, Token, TokenList
from sqlparse.tokens import Keyword, Name, Literal, Operator, Punctuation
import ast
from typing import List, Dict, Tuple, Set, Any, Optional
import numpy as np
from dataclasses import dataclass
import re


@dataclass
class SQLCandidate:
    """Data class for SQL candidate queries"""
    sql: str
    index: int
    
    
@dataclass
class EvaluationScore:
    """Data class for evaluation scores"""
    executability: float  # Executability score (0 or 1)
    schema_conformity: float  # Schema conformity score (0-1)
    example_consistency: float  # Example consistency score (0-1)
    

class ASTProcessor:
    """AST processor for calculating abstract syntax tree edit distance of SQL queries"""
    
    @staticmethod
    def parse_sql_to_ast(sql: str) -> Dict:
        """Parse SQL into simplified AST representation"""
        if not sql or not sql.strip():
            return {'type': 'Empty', 'value': '', 'tokens': []}
            
        try:
            parsed_statements = sqlparse.parse(sql)
            if not parsed_statements:
                return {'type': 'Empty', 'value': '', 'tokens': []}
                
            # Take the first statement
            statement = parsed_statements[0]
            return ASTProcessor._build_ast_dict(statement)
        except Exception as e:
            # Return empty AST when parsing fails
            return {'type': 'Error', 'value': str(e), 'tokens': []}
    
    @staticmethod
    def _build_ast_dict(token) -> Dict:
        """Recursively build AST dictionary"""
        if token is None:
            return {'type': 'None', 'value': '', 'tokens': []}
            
        # Get basic token information
        token_type = type(token).__name__
        token_value = str(token).strip()
        
        # If it's a TokenList type (contains sub-tokens), process recursively
        if hasattr(token, 'tokens') and token.tokens:
            child_tokens = []
            for sub_token in token.tokens:
                # Skip whitespace and meaningless tokens
                if ASTProcessor._is_meaningful_token(sub_token):
                    child_ast = ASTProcessor._build_ast_dict(sub_token)
                    if child_ast['type'] != 'None':  # Only add meaningful child nodes
                        child_tokens.append(child_ast)
            
            return {
                'type': token_type,
                'value': token_value,
                'ttype': str(token.ttype) if hasattr(token, 'ttype') and token.ttype else None,
                'tokens': child_tokens
            }
        else:
            # Leaf node
            return {
                'type': token_type,
                'value': token_value,
                'ttype': str(token.ttype) if hasattr(token, 'ttype') and token.ttype else None,
                'tokens': []
            }
    
    @staticmethod
    def _is_meaningful_token(token) -> bool:
        """Determine if token is meaningful (filter out whitespace and other useless tokens)"""
        if token is None:
            return False
            
        token_str = str(token).strip()
        if not token_str:
            return False
            
        # Filter out pure whitespace and punctuation (except some important ones)
        if hasattr(token, 'ttype'):
            if token.ttype in (sqlparse.tokens.Whitespace, 
                              sqlparse.tokens.Whitespace.Newline,
                              sqlparse.tokens.Comment.Single,
                              sqlparse.tokens.Comment.Multiline):
                return False
        
        return True
    
    @staticmethod
    def calculate_edit_distance(ast1: Dict, ast2: Dict) -> float:
        """Calculate normalized edit distance between two ASTs"""
        
        def node_weight(node: Dict) -> int:
            """Calculate node weight"""
            if not node:
                return 0
            # Base weight is 1, if there are child nodes, calculate recursively
            weight = 1
            for child in node.get('tokens', []):
                weight += node_weight(child)
            return weight
        
        def compute_edit_distance(node1: Dict, node2: Dict) -> int:
            """Compute edit distance between two AST nodes"""
            # Both nodes are empty
            if not node1 and not node2:
                return 0
            
            # One of the nodes is empty
            if not node1:
                return node_weight(node2)
            if not node2:
                return node_weight(node1)
            
            # Compare key attributes of nodes
            nodes_equal = ASTProcessor._nodes_equal(node1, node2)
            
            tokens1 = node1.get('tokens', [])
            tokens2 = node2.get('tokens', [])
            
            if nodes_equal and not tokens1 and not tokens2:
                # Leaf nodes and equal
                return 0
            
            if nodes_equal:
                # Nodes are equal, compare child nodes
                return ASTProcessor._compute_sequence_edit_distance(tokens1, tokens2)
            else:
                # Nodes are not equal, consider substitute, delete, insert operations
                substitute_cost = 1 + ASTProcessor._compute_sequence_edit_distance(tokens1, tokens2)
                delete_cost = node_weight(node1)
                insert_cost = node_weight(node2)
                
                return min(substitute_cost, delete_cost, insert_cost)
        
        # Calculate edit distance
        distance = compute_edit_distance(ast1, ast2)
        
        # Normalize: divide by the maximum weight of the two ASTs
        weight1 = node_weight(ast1)
        weight2 = node_weight(ast2)
        max_weight = max(weight1, weight2)
        
        if max_weight == 0:
            return 0.0
        
        return min(1.0, distance / max_weight)
    
    @staticmethod
    def _nodes_equal(node1: Dict, node2: Dict) -> bool:
        """Determine if two AST nodes are equal"""
        if not node1 or not node2:
            return False
        
        # Compare node types
        if node1.get('type') != node2.get('type'):
            return False
        
        # Compare token types
        if node1.get('ttype') != node2.get('ttype'):
            return False
        
        # For certain critical node types, compare values
        critical_types = ['Keyword', 'Name', 'Literal']
        if node1.get('type') in critical_types:
            # Normalize values for comparison (ignore case and whitespace)
            val1 = node1.get('value', '').strip().lower()
            val2 = node2.get('value', '').strip().lower()
            return val1 == val2
        
        return True
    
    @staticmethod
    def _compute_sequence_edit_distance(seq1: List[Dict], seq2: List[Dict]) -> int:
        """Compute edit distance between two AST node sequences"""
        m, n = len(seq1), len(seq2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize boundary conditions
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    # Insert j nodes
                    dp[i][j] = sum(ASTProcessor._node_weight_simple(seq2[k]) for k in range(j))
                elif j == 0:
                    # Delete i nodes
                    dp[i][j] = sum(ASTProcessor._node_weight_simple(seq1[k]) for k in range(i))
                else:
                    # Calculate costs of three operations
                    node1, node2 = seq1[i-1], seq2[j-1]
                    
                    # Substitution cost
                    if ASTProcessor._nodes_equal(node1, node2):
                        substitute_cost = ASTProcessor._compute_sequence_edit_distance(
                            node1.get('tokens', []), node2.get('tokens', [])
                        )
                    else:
                        substitute_cost = (ASTProcessor._node_weight_simple(node1) + 
                                         ASTProcessor._node_weight_simple(node2))
                    
                    # Delete and insert costs
                    delete_cost = ASTProcessor._node_weight_simple(node1)
                    insert_cost = ASTProcessor._node_weight_simple(node2)
                    
                    dp[i][j] = min(
                        dp[i-1][j-1] + substitute_cost,  # Substitute
                        dp[i-1][j] + delete_cost,        # Delete
                        dp[i][j-1] + insert_cost         # Insert
                    )
        
        return dp[m][n]
    
    @staticmethod
    def _node_weight_simple(node: Dict) -> int:
        """Simple node weight calculation (non-recursive)"""
        return 1 if node else 0


class ParetoOptimal:
    """Pareto Optimal SQL Generator"""
    
    def __init__(self, database_path: str = None):
        """
        Initialize po
        
        Args:
            database_path: SQLite database path for executability checking
        """
        self.database_path = database_path
        self.ast_processor = ASTProcessor()
        
        # Extended SQL keywords list
        self.sql_keywords = {
            'select', 'from', 'where', 'join', 'inner', 'left', 'right', 'full', 'outer',
            'on', 'and', 'or', 'not', 'in', 'exists', 'like', 'between', 'is', 'null',
            'group', 'by', 'order', 'having', 'limit', 'offset', 'distinct', 'all',
            'union', 'intersect', 'except', 'case', 'when', 'then', 'else', 'end',
            'insert', 'update', 'delete', 'create', 'drop', 'alter', 'table', 'view',
            'index', 'into', 'values', 'set', 'as', 'asc', 'desc', 'count', 'sum',
            'avg', 'min', 'max', 'with', 'recursive', 'over', 'partition', 'window',
            'cast', 'convert', 'substring', 'trim', 'upper', 'lower', 'length',
            'coalesce', 'nullif', 'round', 'floor', 'ceil', 'abs', 'mod', 'power',
            'sqrt', 'log', 'exp', 'sin', 'cos', 'tan', 'concat', 'replace'
        }
    
    def evaluate_executability(self, sql: str) -> float:
        """
        Evaluate SQL executability
        
        Args:
            sql: SQL query string
            
        Returns:
            Executability score (1.0 for executable, 0.0 for non-executable)
        """
        if not self.database_path:
            # If no database, perform simple syntax check
            try:
                parsed = sqlparse.parse(sql)
                return 1.0 if parsed else 0.0
            except:
                return 0.0
        
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Try to execute SQL (use LIMIT 1 to avoid large result sets)
            limited_sql = self._add_limit_to_sql(sql, 1)
            cursor.execute(limited_sql)
            cursor.fetchall()
            conn.close()
            return 1.0
        except Exception as e:
            if conn:
                conn.close()
            return 0.0
    
    def _add_limit_to_sql(self, sql: str, limit: int) -> str:
        """Add LIMIT clause to SQL"""
        sql = sql.strip().rstrip(';')
        if 'LIMIT' not in sql.upper():
            sql += f' LIMIT {limit}'
        return sql
    
    def evaluate_schema_conformity(self, sql: str, schema_links: Set[str]) -> float:
        """
        Evaluate schema conformity
        
        Args:
            sql: SQL query string
            schema_links: Set of schema links (table and column names)
            
        Returns:
            Schema conformity score (between 0-1)
        """
        schema_used = self._extract_schema_from_sql(sql)
        
        if not schema_used and not schema_links:
            return 1.0
        
        if not schema_used:
            return 0.0
            
        if not schema_links:
            return 0.0
        
        # Calculate intersection and union
        intersection = schema_used.intersection(schema_links)
        union = schema_used.union(schema_links)
        
        # Use Jaccard similarity
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        
        # Also use coverage: how much of schema_used is in schema_links
        coverage = len(intersection) / len(schema_used) if schema_used else 0.0
        
        # Combine both metrics
        return (jaccard_similarity + coverage) / 2.0
    
    def _extract_schema_from_sql(self, sql: str) -> Set[str]:
        """Extract schema elements (table and column names) used in SQL"""
        schema_elements = set()
        
        # Preprocess SQL: remove string constants to avoid misidentification
        sql_cleaned = self._remove_string_literals(sql)
        
        # Use regex to extract all possible identifiers
        # Match words starting with letters, containing letters, numbers, underscores
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]*\b', sql_cleaned)
        
        # Convert to lowercase and filter keywords
        for word in words:
            word_lower = word.lower()
            if word_lower not in self.sql_keywords:
                schema_elements.add(word_lower)
        
        # Additional processing: extract table names and column names from table.column format
        dot_patterns = re.findall(r'\b([a-zA-Z][a-zA-Z0-9_]*\.[a-zA-Z][a-zA-Z0-9_]*)\b', sql_cleaned)
        for pattern in dot_patterns:
            parts = pattern.split('.')
            for part in parts:
                part_lower = part.lower()
                if part_lower not in self.sql_keywords:
                    schema_elements.add(part_lower)
        
        return schema_elements
    
    def _remove_string_literals(self, sql: str) -> str:
        """Remove string literals from SQL to avoid misidentification"""
        # Remove single-quoted strings
        sql = re.sub(r"'[^']*'", "''", sql)
        # Remove double-quoted strings
        sql = re.sub(r'"[^"]*"', '""', sql)
        # Remove backtick-quoted strings (MySQL style)
        sql = re.sub(r'`[^`]*`', '``', sql)
        return sql
    
    def evaluate_example_consistency(self, sql: str, examples: List[str]) -> float:
        """
        Evaluate example consistency
        
        Args:
            sql: Candidate SQL query
            examples: List of example SQL queries
            
        Returns:
            Example consistency score (between 0-1)
        """
        if not examples:
            return 0.0
        
        sql_ast = self.ast_processor.parse_sql_to_ast(sql)
        similarities = []
        
        for example in examples:
            example_ast = self.ast_processor.parse_sql_to_ast(example)
            distance = self.ast_processor.calculate_edit_distance(sql_ast, example_ast)
            similarity = 1.0 - distance  # Convert distance to similarity
            similarities.append(max(0.0, similarity))  # Ensure similarity is non-negative
        
        # Return average similarity
        return sum(similarities) / len(similarities)
    
    def evaluate_sql_candidates(
        self, 
        candidates: List[str], 
        schema_links: Set[str], 
        examples: List[str]
    ) -> List[Tuple[SQLCandidate, EvaluationScore]]:
        """
        Evaluate all SQL candidate queries
        
        Args:
            candidates: List of SQL candidate queries
            schema_links: Schema link information
            examples: List of example SQL queries
            
        Returns:
            List of candidate queries and their evaluation scores
        """
        evaluated_candidates = []
        
        for i, sql in enumerate(candidates):
            candidate = SQLCandidate(sql=sql, index=i)
            
            # Evaluate three dimensions
            executability = self.evaluate_executability(sql)
            schema_conformity = self.evaluate_schema_conformity(sql, schema_links)
            example_consistency = self.evaluate_example_consistency(sql, examples)
            
            score = EvaluationScore(
                executability=executability,
                schema_conformity=schema_conformity,
                example_consistency=example_consistency
            )
            
            evaluated_candidates.append((candidate, score))
        
        return evaluated_candidates
    
    def find_pareto_optimal(
        self, 
        evaluated_candidates: List[Tuple[SQLCandidate, EvaluationScore]]
    ) -> List[SQLCandidate]:
        """
        Find Pareto optimal solution set
        
        Args:
            evaluated_candidates: List of evaluated candidate queries
            
        Returns:
            List of Pareto optimal SQL candidate queries
        """
        # First filter out non-executable queries
        executable_candidates = [
            (candidate, score) for candidate, score in evaluated_candidates
            if score.executability > 0.0
        ]
        
        if not executable_candidates:
            return []
        
        pareto_optimal = []
        
        for i, (candidate_i, score_i) in enumerate(executable_candidates):
            is_dominated = False
            
            for j, (candidate_j, score_j) in enumerate(executable_candidates):
                if i == j:
                    continue
                
                # Check if candidate_i is dominated by candidate_j
                if (score_j.schema_conformity >= score_i.schema_conformity and
                    score_j.example_consistency >= score_i.example_consistency and
                    (score_j.schema_conformity > score_i.schema_conformity or
                     score_j.example_consistency > score_i.example_consistency)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(candidate_i)
        
        return pareto_optimal
    
    def select_final_sql(
        self, 
        candidates: List[str], 
        schema_links: Set[str], 
        examples: List[str],
        selection_strategy: str = "balanced"
    ) -> str:
        """
        Select the final SQL query
        
        Args:
            candidates: List of SQL candidate queries
            schema_links: Schema link information
            examples: List of example SQL queries
            selection_strategy: Selection strategy ("balanced", "schema_priority", "example_priority")
            
        Returns:
            The selected final SQL query
        """
        if not candidates:
            return ""
        
        # Evaluate all candidate queries
        evaluated_candidates = self.evaluate_sql_candidates(candidates, schema_links, examples)
        
        # Find Pareto optimal solutions
        pareto_optimal = self.find_pareto_optimal(evaluated_candidates)
        
        if not pareto_optimal:
            # If no Pareto optimal solutions, return the first executable query
            for candidate, score in evaluated_candidates:
                if score.executability > 0.0:
                    return candidate.sql
            return candidates[0]  # Final fallback option
        
        if len(pareto_optimal) == 1:
            return pareto_optimal[0].sql
        
        # If there are multiple Pareto optimal solutions, select based on strategy
        best_candidate = None
        best_score = -1.0
        
        for candidate in pareto_optimal:
            # Get corresponding evaluation score
            score = None
            for c, s in evaluated_candidates:
                if c.index == candidate.index:
                    score = s
                    break
            
            if score is None:
                continue
            
            # Calculate combined score based on selection strategy
            if selection_strategy == "balanced":
                combined_score = (score.schema_conformity + score.example_consistency) / 2.0
            elif selection_strategy == "schema_priority":
                combined_score = 0.7 * score.schema_conformity + 0.3 * score.example_consistency
            elif selection_strategy == "example_priority":
                combined_score = 0.3 * score.schema_conformity + 0.7 * score.example_consistency
            else:
                combined_score = (score.schema_conformity + score.example_consistency) / 2.0
            
            if combined_score > best_score:
                best_score = combined_score
                best_candidate = candidate
        
        return best_candidate.sql if best_candidate else pareto_optimal[0].sql


# Usage example and testing
def demo_pareto_optimal_selection():
    """Demonstrate the usage of Pareto optimal selection"""
    
    # Create po instance
    po = ParetoOptimal()
    # po = ParetoOptimal(database_path=db_path)

    
    # Example data
    candidate_sqls = [
        "SELECT name FROM customers WHERE age > 25",
        "SELECT customer_name FROM customer WHERE customer_age > 25",
        "SELECT c.name FROM customers c WHERE c.age > 25 ORDER BY c.name",
        "SELECT * FROM customers WHERE age > 25",
        "SELECT name, age FROM customers WHERE age > 25 AND city = 'New York'"
    ]
    
    schema_links = {"customers", "name", "age", "city", "customer_id"}
    
    example_sqls = [
        "SELECT name FROM employees WHERE salary > 50000",
        "SELECT product_name FROM products WHERE price > 100"
    ]
    
    print("=== AST Parsing Test ===")
    for i, sql in enumerate(candidate_sqls):
        ast_result = po.ast_processor.parse_sql_to_ast(sql)
        print(f"SQL {i+1}: {sql}")
        print(f"AST Type: {ast_result.get('type')}")
        print(f"Number of child nodes: {len(ast_result.get('tokens', []))}")
        print()
    
    print("=== Schema Extraction Test ===")
    for i, sql in enumerate(candidate_sqls):
        extracted = po._extract_schema_from_sql(sql)
        print(f"SQL {i+1}: {sql}")
        print(f"Extracted Schema: {extracted}")
        conformity = po.evaluate_schema_conformity(sql, schema_links)
        print(f"Schema conformity score: {conformity:.3f}")
        print()
    
    # Select final SQL
    final_sql = po.select_final_sql(
        candidates=candidate_sqls,
        schema_links=schema_links,
        examples=example_sqls,
        selection_strategy="balanced"
    )
    
    print("=== Complete Evaluation Results ===")
    print("Candidate SQL queries:")
    for i, sql in enumerate(candidate_sqls):
        print(f"{i+1}. {sql}")
    
    print(f"\nSelected final SQL: {final_sql}")
    
    # Show detailed evaluation information
    evaluated = po.evaluate_sql_candidates(candidate_sqls, schema_links, example_sqls)
    pareto_optimal = po.find_pareto_optimal(evaluated)
    
    print("\nDetailed evaluation results:")
    for candidate, score in evaluated:
        print(f"SQL {candidate.index + 1}:")
        print(f"  Executability: {score.executability:.3f}")
        print(f"  Schema conformity: {score.schema_conformity:.3f}")
        print(f"  Example consistency: {score.example_consistency:.3f}")
        print()
    
    print("Pareto optimal solutions:")
    for candidate in pareto_optimal:
        print(f"  SQL {candidate.index + 1}: {candidate.sql}")


if __name__ == "__main__":
    demo_pareto_optimal_selection()