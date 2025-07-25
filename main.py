import pandas as pd
import torch
from arg import main_args
import os
from llm import LLM_generation
import json
import time
import logging
import re
from po import ParetoOptimal

generation_prompt = '''
You are an NL2SQL expert
I will give you the database structure, the most likely table columns. You can use this information to perform the NL2SQL task.
Please read and understand the database schema carefully, and generate an executable SQL. The generated SQL is protected by ```sql and ```.
'''

args = main_args()
DATASET = f"./data_with_sk.json"
OUTPUT_FILE = "log/predicted.sql"
SIMILAR_FILE = "top_k_similarities.json"

with open(SIMILAR_FILE, 'r', encoding='utf-8') as f:
    examples = json.load(f)

def load_data(DATASET):
    return pd.read_json(DATASET)

# Spider
def prompt_maker(question, database, schema_links_pred):   
    prompt = f"\n### Question:{question}" + f"\n### Database: {database}" + f"\n### Possible schemas:{schema_links_pred}"
    print(prompt)
    return prompt

def extract_schema_links(json_dict, database_name, schema_links):
    """
    Extract key-value pairs from a parsed JSON dictionary for specified database and schema links
    
    Parameters:
    json_dict (dict): Already parsed JSON dictionary
    database_name (str): Database name
    schema_links (list): List in table_name.column_name format
    
    Returns:
    dict: Dictionary of matching key-value pairs
    """
    # Check input type
    if isinstance(json_dict, str):
        # If input is string, try to parse as dictionary (for backward compatibility)
        try:
            import json
            import re
            
            # Since the provided content is not in standard JSON format, preprocessing is needed
            content_lines = json_dict.strip().split('\n')
            parsed_dict = {}
            
            current_key = None
            current_value = ""
            
            for line in content_lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if it's the start of a new key-value pair
                match = re.match(r'"([^"]+)": "(.*)', line)
                if match:
                    if current_key:  # Save the previous key-value pair
                        parsed_dict[current_key] = current_value.strip()
                        
                    current_key = match.group(1)
                    current_value = match.group(2)
                    
                    # If current line is a complete key-value pair
                    if line.endswith('",'):
                        current_value = current_value[:-1]  # Remove the trailing comma
                        parsed_dict[current_key] = current_value
                        current_key = None
                        current_value = ""
                else:
                    # Continue the value of the previous key-value pair
                    if current_key:
                        current_value += " " + line
                        if line.endswith('",'):
                            current_value = current_value[:-1]  # Remove the trailing comma
                            parsed_dict[current_key] = current_value
                            current_key = None
                            current_value = ""
            
            # Handle the last key-value pair
            if current_key:
                parsed_dict[current_key] = current_value
                
            # Assign the parsed dictionary to json_dict
            json_dict = parsed_dict
        except Exception as e:
            print(f"String parsing failed: {e}")
            return {}
    
    # Filter results based on schema_links
    result = {}
    for link in schema_links:
        try:
            table_name, column_name = link.split('.')
            
            # Find matching keys
            for key, value in json_dict.items():
                parts = key.split('|')
                if len(parts) == 3 and parts[0] == database_name and parts[1] == table_name:
                    result[key] = value
        except Exception as e:
            print(f"Processing link {link} failed: {e}")
            continue
    
    return result

def extract_sql_content(text):
    """
    Extract SQL content between ```sql and ``` in text, and remove line breaks.
    
    Parameters:
    text (str): Input text containing SQL code blocks
    
    Returns:
    str: Extracted SQL content with line breaks and leading/trailing whitespace removed,
         returns empty string if not found
    """
    # Find the position of the first ```sql
    start_index = text.find('```sql')
    
    # If ```sql is not found, return empty string
    if start_index == -1:
        return ''
    
    # From after ```sql, find the ending ``` marker
    end_index = text.find('```', start_index + 6)
    
    # If ending marker is not found, return empty string
    if end_index == -1:
        return ''
    
    # Extract content between ```sql and ```, remove line breaks and trim whitespace
    sql_content = text[start_index + 6:end_index].replace('\n', ' ').strip()
    
    return sql_content

def convert_schema_links_to_set(schema_links):
    """
    Convert schema_links to set format for ParetoOptimal usage
    
    Parameters:
    schema_links: Schema links in list, string, or other formats
    
    Returns:
    set: Set containing all schema elements
    """
    schema_set = set()
    
    if isinstance(schema_links, list):
        for link in schema_links:
            if isinstance(link, str):
                # Handle table.column format
                if '.' in link:
                    parts = link.split('.')
                    schema_set.update(parts)
                else:
                    schema_set.add(link)
    elif isinstance(schema_links, str):
        # If it's a string, try to parse
        if '.' in schema_links:
            parts = schema_links.split('.')
            schema_set.update(parts)
        else:
            schema_set.add(schema_links)
    
    return schema_set

def extract_example_sqls(top_k_matches):
    """
    Extract example SQLs from top_k_matches
    
    Parameters:
    top_k_matches: Data structure containing similar examples
    
    Returns:
    list: List of example SQLs
    """
    example_sqls = []
    
    if isinstance(top_k_matches, list):
        for match in top_k_matches:
            if isinstance(match, dict):
                # Try different possible key names
                sql_keys = ['sql', 'query', 'SQL', 'Query']
                for key in sql_keys:
                    if key in match:
                        example_sqls.append(match[key])
                        break
            elif isinstance(match, str):
                example_sqls.append(match)
    
    return example_sqls

def generate_and_validate_sql(
    question, 
    database,
    schema_links, 
    top_k_matches, 
    db_id, 
    max_retries=3, 
    retry_delay=3
):
    """
    Generate SQL with retry mechanism for LLM generation and SQL validation using ParetoOptimal.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize ParetoOptimal instance
    po = ParetoOptimal(db_id=db_id)
    
    # Convert schema_links to set format
    schema_set = convert_schema_links_to_set(schema_links)
    
    # Extract example SQLs
    example_sqls = extract_example_sqls(top_k_matches)
    
    prompt = prompt_maker(question, database, schema_links) 

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempt {attempt}: Generating SQL...")
            
            # Generate SQL candidates
            responses = LLM_generation(generation_prompt, prompt)
            candidate_sqls = []
            
            # Extract SQL from all responses
            for i, res in enumerate(responses):
                sql = extract_sql_content(res)
                if sql and sql.strip():  # Only add non-empty SQL
                    candidate_sqls.append(sql.strip())
            
            if not candidate_sqls:
                logger.warning(f"Attempt {attempt}: No valid SQL extracted from responses")
                if attempt < max_retries:
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error("All attempts failed. No valid SQL generated.")
                    return ""
            
            logger.info(f"Attempt {attempt}: Generated {len(candidate_sqls)} SQL candidates")
            
            # Use ParetoOptimal to select the best SQL
            try:
                final_sql = po.select_final_sql(
                    candidates=candidate_sqls,
                    schema_links=schema_set,
                    examples=example_sqls,
                    selection_strategy="balanced"
                )
                
                if final_sql and final_sql.strip():
                    logger.info(f"Successfully selected final SQL: {final_sql[:100]}...")
                    return final_sql
                else:
                    logger.warning(f"Attempt {attempt}: ParetoOptimal returned empty SQL")
                    
            except Exception as e:
                logger.error(f"Attempt {attempt}: ParetoOptimal selection failed: {str(e)}")
                # If ParetoOptimal fails, return the first candidate SQL
                if candidate_sqls:
                    logger.info("Falling back to first candidate SQL")
                    return candidate_sqls[0]
            
            if attempt < max_retries:
                logger.info(f"Attempt {attempt} failed, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                
        except Exception as e:
            logger.error(f"Attempt {attempt}: Unexpected error: {str(e)}")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                logger.error("All attempts failed due to unexpected errors.")
                return ""
    
    logger.error("All attempts exhausted. Returning empty string.")
    return ""

def file_exists(file_path):
    return os.path.isfile(file_path)

if __name__ == '__main__':
    val_df = load_data(DATASET)
    print(f"Number of data samples {val_df.shape[0]}")
    RESULT = []
    index = 0
    
    if file_exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            line_count = sum(1 for line in f)
    else:
        line_count = 0
    
    for index, row in val_df.iterrows():
        if index < line_count:
            continue
            
        print(f"-------------------------------------------index is {index}-------------------------------------------")
        question = row['question']
        database = row['database']
        db_id = row['db_id']
        schema_links = row['schema_links']
        top_k_matches = examples[index]['top_k_matches']
        
        
        sql = generate_and_validate_sql(question, database, schema_links, top_k_matches, db_id)
        
        RESULT.append([question, sql])
        
        # Batch save results
        if (index + 1) % 5 == 0 or index == len(val_df) - 1:
            df = pd.DataFrame(RESULT, columns=['NLQ', 'PREDICTED SQL'])
            results = df['PREDICTED SQL'].tolist()
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                for line in results:
                    f.write(f"{line}\n")
            RESULT = []
            
        print(f"Generated SQL: {sql}")
        print(f"Completed {index + 1}/{len(val_df)} samples")