import json
import re
from typing import Dict, List, Any

def parse_database_schema(database_text: str) -> Dict[str, Any]:
    """
    Parse database schema text to extract table names and column names
    
    Args:
        database_text: Text containing database schema information
    
    Returns:
        Dictionary containing tables and columns information
    """
    schema = {
        "tables": [],
        "columns": {}
    }
    
    # Use regex to find all table definitions
    table_pattern = r'# Table: (\w+)\s*\[(.*?)\]'
    tables = re.findall(table_pattern, database_text, re.DOTALL)
    
    for table_name, table_content in tables:
        schema["tables"].append(table_name)
        
        # Extract column names
        columns = []
        # Match format like: (cdscode:TEXT, Primary Key, Examples: [01100170109835, 01100170112607])
        column_pattern = r'\(([^:]+):'
        column_matches = re.findall(column_pattern, table_content)
        
        for column_name in column_matches:
            # Clean column name, remove possible spaces and special characters
            clean_column = column_name.strip()
            if clean_column:
                columns.append(clean_column)
        
        schema["columns"][table_name] = columns
    
    return schema

def process_json_file(input_file: str, output_file: str = None):
    """
    Process JSON file, add schema key to each dictionary
    
    Args:
        input_file: Input JSON file path
        output_file: Output JSON file path (if None, overwrite original file)
    """
    try:
        # Read JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure data is a list
        if not isinstance(data, list):
            raise ValueError("JSON file should contain a list")
        
        # Process each dictionary
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                print(f"Warning: Element {i+1} is not a dictionary, skipping")
                continue
            
            if "database" not in item:
                print(f"Warning: Dictionary {i+1} does not have 'database' key, skipping")
                continue
            
            try:
                # Parse database field
                database_text = item["database"]
                schema = parse_database_schema(database_text)
                
                # Add schema key
                item["schema"] = schema
                
                print(f"Successfully processed dictionary {i+1}, found {len(schema['tables'])} tables")
                
            except Exception as e:
                print(f"Error processing dictionary {i+1}: {e}")
                continue
        
        # Save results
        output_path = output_file if output_file else input_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Processing completed, results saved to: {output_path}")
        
    except FileNotFoundError:
        print(f"Error: File not found {input_file}")
    except json.JSONDecodeError:
        print(f"Error: {input_file} is not a valid JSON file")
    except Exception as e:
        print(f"Error occurred while processing file: {e}")

if __name__ == "__main__":
    # Usage example: specify file paths directly
    # process_json_file('train_input.json', 'train_output.json')
