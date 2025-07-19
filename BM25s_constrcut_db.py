import json
import os
from function import get_schema_dict, extract_db_samples_enriched_bm25
from tqdm import tqdm

DATA = "./data.json"
OUTPUT_FILE = './data_with_sample.json'
CHECKPOINT_DIR = './checkpoints'
CHECKPOINT_INTERVAL = 100

# Create checkpoints directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Check if any checkpoints exist to resume from
def get_latest_checkpoint():
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('checkpoint_') and f.endswith('.json')]
    if not checkpoint_files:
        return None, 0
    
    # Extract checkpoint numbers and find the latest
    checkpoint_nums = [int(f.split('_')[1].split('.')[0]) for f in checkpoint_files]
    latest_num = max(checkpoint_nums)
    latest_file = f"checkpoint_{latest_num}.json"
    
    return os.path.join(CHECKPOINT_DIR, latest_file), latest_num * CHECKPOINT_INTERVAL

def replace_schema_examples(schema_text, new_examples_text):
    """
    Replace example values in a database schema with new examples.
    
    Args:
        schema_text (str): The original schema text with examples to replace
        new_examples_text (str): Text containing new example values
        
    Returns:
        str: Schema text with updated example values
    """
    # Parse the new examples
    new_examples = {}
    current_table = None
    
    # Process the new examples text to extract values
    for line in new_examples_text.strip().split('\n'):
        if line.startswith('## ') and 'table samples:' in line:
            current_table = line.split()[1]
            new_examples[current_table] = {}
        elif line.startswith('# Example values for '):
            if current_table:
                # Extract column and values from the line
                parts = line.split("'.")
                if len(parts) >= 2:
                    table_name = parts[0].split("'")[1]
                    column_name = parts[1].split("'")[1]
                    
                    # Extract values between the brackets
                    values_str = line.split('column: [')[1].split(']')[0]
                    # Split by comma, but handle NULL properly
                    values = []
                    for val in values_str.split(', '):
                        # Remove extra quotes for string values
                        cleaned_val = val.strip("'")
                        values.append(cleaned_val)
                    
                    if table_name not in new_examples:
                        new_examples[table_name] = {}
                    
                    new_examples[table_name][column_name] = values
    
    # Process and update the original schema
    lines = schema_text.split('\n')
    updated_lines = []
    
    current_table = None
    
    for line in lines:
        # Identify when we're in a new table definition
        if line.startswith('# Table: '):
            current_table = line.split('# Table: ')[1]
            updated_lines.append(line)
        # Check if this line contains examples to replace
        elif 'Examples:' in line and current_table in new_examples:
            # Extract column name from the line
            parts = line.split('(')[-1].split(',')
            if len(parts) >= 1:
                column_name = parts[0].split(':')[0]
                
                # Check if we have new examples for this column
                if column_name in new_examples[current_table]:
                    # Create the new examples string
                    new_examples_str = str(new_examples[current_table][column_name]).replace("'NULL'", "NULL")
                    # Replace the old examples with the new ones
                    before_examples = line.split('Examples: ')[0]
                    updated_line = f"{before_examples}Examples: {new_examples_str})"
                    updated_lines.append(updated_line)
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)
        else:
            updated_lines.append(line)
    
    return '\n'.join(updated_lines)

# Load data
with open(DATA, 'r', encoding='utf-8') as file:
    datasets = json.load(file)

# Check for existing checkpoint
latest_checkpoint, start_idx = get_latest_checkpoint()
if latest_checkpoint:
    print(f"Resuming from checkpoint: {latest_checkpoint} (starting at index {start_idx})")
    with open(latest_checkpoint, 'r', encoding='utf-8') as f:
        datasets = json.load(f)

# Process data with checkpoints
for index, dt in enumerate(tqdm(datasets[start_idx:], desc="Processing", initial=start_idx, total=len(datasets))):
    actual_idx = index + start_idx
    
    question = dt['question']
    evidence = dt['evidence']
    database = dt['database']
    db_id = dt['db_id']
    db_path = "/path/to/data/database/{db_id}/{db_id}.sqlite"

    schema_dict = get_schema_dict(db_path)
    sample = extract_db_samples_enriched_bm25(question, db_path, schema_dict,evidence, 2)# 
    new_database = replace_schema_examples(database, sample)
    dt['database'] = new_database
    
    # Save checkpoint every CHECKPOINT_INTERVAL items
    if (actual_idx + 1) % CHECKPOINT_INTERVAL == 0:
        checkpoint_num = (actual_idx + 1) // CHECKPOINT_INTERVAL
        checkpoint_file = os.path.join(CHECKPOINT_DIR, f"checkpoint_{checkpoint_num}.json")
        print(f"\nSaving checkpoint at index {actual_idx} to {checkpoint_file}")
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(datasets, f, ensure_ascii=False, indent=2)

# Save final results to the output file
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(datasets, f, ensure_ascii=False, indent=2)

print(f"Processing complete. Final output saved to {OUTPUT_FILE}")
