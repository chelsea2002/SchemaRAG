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
    
    # First find all table start positions
    table_start_pattern = r'# Table: (\w+)\s*\['
    table_starts = []
    
    for match in re.finditer(table_start_pattern, database_text):
        table_starts.append({
            'name': match.group(1),
            'start': match.start(),
            'bracket_start': match.end() - 1  # Position of '['
        })
    
    # Find corresponding end position for each table
    for i, current_table in enumerate(table_starts):
        # Determine search range
        search_start = current_table['bracket_start'] + 1
        if i + 1 < len(table_starts):
            search_end = table_starts[i + 1]['start']
        else:
            search_end = len(database_text)
        
        search_text = database_text[search_start:search_end]
        
        # Find matching ], this ] should be at line start (or preceded only by whitespace)
        end_bracket_pattern = r'^\s*\]'
        end_match = re.search(end_bracket_pattern, search_text, re.MULTILINE)
        
        if end_match:
            table_content_end = search_start + end_match.start()
            table_content = database_text[current_table['bracket_start'] + 1:table_content_end]
            
            # Extract column names
            column_pattern = r'\((\w+):'
            column_matches = re.findall(column_pattern, table_content)
            
            schema["tables"].append(current_table['name'])
            schema["columns"][current_table['name']] = column_matches
    
    return schema

def process_database_field(item: Dict[str, Any], field_name: str = "database") -> bool:
    """
    Process a single database field and add corresponding schema
    
    Args:
        item: Dictionary containing the database field
        field_name: Name of the database field to process
        
    Returns:
        True if processed successfully, False otherwise
    """
    if field_name not in item:
        return False
    
    try:
        database_text = item[field_name]
        schema = parse_database_schema(database_text)
        item["schema"] = schema
        return True
    except Exception as e:
        print(f"Error processing {field_name} field: {e}")
        return False

def process_json_file(input_file: str, output_file: str = None):
    """
    Process JSON file, add schema key to each dictionary and similar items
    
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
            
            print(f"\n--- Processing dictionary {i+1} ---")
            
            # Process main database field
            if "database" in item:
                success = process_database_field(item, "database")
                if success:
                    schema = item["schema"]
                    total_columns = sum(len(cols) for cols in schema['columns'].values())
                    print(f"Main database: found {len(schema['tables'])} tables, {total_columns} columns total")
                    
                    # Print column count for each table
                    for table_name, columns in schema['columns'].items():
                        print(f"  - {table_name}: {len(columns)} columns")
                else:
                    print("Failed to process main database field")
            else:
                print("Warning: No 'database' key found in main item")
            
            # Process similar items if they exist
            if "similar" in item and isinstance(item["similar"], list):
                print(f"Processing {len(item['similar'])} similar items...")
                
                for j, similar_item in enumerate(item["similar"]):
                    if not isinstance(similar_item, dict):
                        print(f"  Warning: Similar item {j+1} is not a dictionary, skipping")
                        continue
                    
                    if "database" in similar_item:
                        success = process_database_field(similar_item, "database")
                        if success:
                            schema = similar_item["schema"]
                            total_columns = sum(len(cols) for cols in schema['columns'].values())
                            print(f"  Similar item {j+1}: found {len(schema['tables'])} tables, {total_columns} columns total")
                        else:
                            print(f"  Failed to process similar item {j+1} database field")
                    else:
                        print(f"  Warning: Similar item {j+1} has no 'database' key")
            else:
                print("No 'similar' field found or it's not a list")
        
        # Save results
        output_path = output_file if output_file else input_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\nProcessing completed, results saved to: {output_path}")
        
    except FileNotFoundError:
        print(f"Error: File not found {input_file}")
    except json.JSONDecodeError:
        print(f"Error: {input_file} is not a valid JSON file")
    except Exception as e:
        print(f"Error occurred while processing file: {e}")


if __name__ == "__main__":
    # Process actual file
    process_json_file('train_input.json', 'train_output.json')
