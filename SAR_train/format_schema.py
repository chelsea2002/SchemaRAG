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
    
    # 先找到所有表的开始位置
    table_start_pattern = r'# Table: (\w+)\s*\['
    table_starts = []
    
    for match in re.finditer(table_start_pattern, database_text):
        table_starts.append({
            'name': match.group(1),
            'start': match.start(),
            'bracket_start': match.end() - 1  # '[' 的位置
        })
    
    # 为每个表找到对应的结束位置
    for i, current_table in enumerate(table_starts):
        # 确定搜索范围
        search_start = current_table['bracket_start'] + 1
        if i + 1 < len(table_starts):
            search_end = table_starts[i + 1]['start']
        else:
            search_end = len(database_text)
        
        search_text = database_text[search_start:search_end]
        
        # 找到匹配的 ]，这个 ] 应该在行首（或者前面只有空白字符）
        end_bracket_pattern = r'^\s*\]'
        end_match = re.search(end_bracket_pattern, search_text, re.MULTILINE)
        
        if end_match:
            table_content_end = search_start + end_match.start()
            table_content = database_text[current_table['bracket_start'] + 1:table_content_end]
            
            # 提取列名
            column_pattern = r'\((\w+):'
            column_matches = re.findall(column_pattern, table_content)
            
            schema["tables"].append(current_table['name'])
            schema["columns"][current_table['name']] = column_matches
    
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
                
                # 打印更详细的信息
                total_columns = sum(len(cols) for cols in schema['columns'].values())
                print(f"Successfully processed dictionary {i+1}, found {len(schema['tables'])} tables, {total_columns} columns total")
                
                # 打印每个表的列数
                for table_name, columns in schema['columns'].items():
                    print(f"  - {table_name}: {len(columns)} columns")
                
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
    
    # 然后处理实际文件
    process_json_file('train_input.json', 'train_output.json')
