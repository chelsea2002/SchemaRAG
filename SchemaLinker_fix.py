from FlagEmbedding import FlagModel
import torch.nn.functional as F
import torch
import json
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flag_model = FlagModel('/path/to/model/bge-large-en-v1.5', use_fp16=True)

INPUT_SCHEMA = "data_pre.json"
OUTPUT_SCHEMA = "data_pre_fix.json"

def extract_table_columns(schema_text):
    """
    Extract all table column information from database schema text and return in 'table.column' format
    
    Args:
        schema_text (str): Text containing table structure information
        
    Returns:
        list: List containing all entries in 'table.column' format
    """
    result = []
    current_table = None
    
    # Process text line by line
    for line in schema_text.split('\n'):
        line = line.strip()
        
        # Check if it's a table definition line
        if line.startswith('# Table:'):
            current_table = line[8:].strip()
            continue
            
        # If within table definition and line contains column definition
        if current_table and line.startswith('(') and ':' in line:
            # Extract column name
            column_name = line[1:line.find(':')].strip()
            # Add table.column format entry to result list
            result.append(f"{current_table}.{column_name}")
    
    return result

def extract_db_schema(text):
    """
    Extract database schema part from provided text and parse table columns
    
    Args:
        text (str): Complete text containing DB_ID and Schema
        
    Returns:
        list: List containing all entries in 'table.column' format
    """
    # Find schema section
    schema_start = text.find('【Schema】')
    if schema_start == -1:
        return []
    
    schema_text = text[schema_start + len('【Schema】'):]
    
    # May need to find schema end position if there are subsequent sections
    foreign_keys_start = schema_text.find('【Foreign keys】')
    if foreign_keys_start != -1:
        schema_text = schema_text[:foreign_keys_start]
    
    return extract_table_columns(schema_text)

with open(INPUT_SCHEMA, 'r',encoding='utf-8') as f:
    schemas = json.load(f)

for i, schema in enumerate(tqdm(schemas, desc="Processing schemas")):
    databases = extract_db_schema(schema['database'])
    schema_pres = schema['schema_links_pred']
    
    if not databases or not schema_pres:
        schema['schema_pre_fixs'] = []
        continue
    
    # Batch encode and convert to tensor
    db_embeds = torch.tensor(flag_model.encode(databases), dtype=torch.float32).to(device)  # shape: (db_len, dim)
    schema_pre_embeds = torch.tensor(flag_model.encode(schema_pres), dtype=torch.float32).to(device)  # shape: (pre_len, dim)
    
    # Normalize
    db_embeds = F.normalize(db_embeds, p=2, dim=1)
    schema_pre_embeds = F.normalize(schema_pre_embeds, p=2, dim=1)
    
    # Calculate cosine similarity matrix: (pre_len, db_len)
    sim_matrix = torch.matmul(schema_pre_embeds, db_embeds.T)
    
    # Index of maximum value in each row (corresponding to which database field each schema_pred matches)
    max_indices = torch.argmax(sim_matrix, dim=1).tolist()
    
    # Corresponding database column names
    schema_fixs = [databases[idx] for idx in max_indices]
    schema['schema_pre_fixs'] = schema_fixs

with open(OUTPUT_SCHEMA, "w", encoding="utf-8") as f:
    json.dump(schemas, f, ensure_ascii=False, indent=2)