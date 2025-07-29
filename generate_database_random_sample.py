import pandas as pd
import os
import json
import sqlite3

DATASET = f"dev.json"

def load_data(dataset_path):
    return pd.read_json(dataset_path)

def extract_sqlite_schema(db_path, db_id="", max_samples=3):
    """
    Extract schema from SQLite database and convert to specified format
    
    Args:
        db_path: Path to SQLite database file
        db_id: Database identifier, can be empty
        max_samples: Maximum number of example values to extract per field
    """
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file does not exist: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        schema_sections = []
        foreign_keys = []
        
        # Process each table
        for table in tables:
            # Get table structure information
            cursor.execute(f"PRAGMA table_info({table})")
            columns_info = cursor.fetchall()
            
            # Get foreign key information
            cursor.execute(f"PRAGMA foreign_key_list({table})")
            fk_info = cursor.fetchall()
            
            # Process foreign key relationships
            for fk in fk_info:
                # fk format: (id, seq, table, from, to, on_update, on_delete, match)
                foreign_keys.append(f"{table}.{fk[3]}={fk[2]}.{fk[4]}")
            
            # Build table schema
            table_schema = f"# Table: {table}\n["
            
            for col_info in columns_info:
                # col_info format: (cid, name, type, notnull, dflt_value, pk)
                col_name = col_info[1]
                col_type = col_info[2].upper() if col_info[2] else "TEXT"
                is_pk = col_info[5]
                
                # Get sample data
                try:
                    cursor.execute(f"SELECT DISTINCT {col_name} FROM {table} WHERE {col_name} IS NOT NULL LIMIT {max_samples}")
                    examples = [str(row[0]) for row in cursor.fetchall()]
                except:
                    examples = []
                
                # Format column information
                pk_text = ", Primary Key" if is_pk else ""
                examples_text = str(examples) if examples else "[]"
                
                table_schema += f"\n({col_name}:{col_type}{pk_text}, Examples: {examples_text})"
            
            table_schema += "\n]"
            schema_sections.append(table_schema)
        
        # Build final schema string
        schema_text = f'【DB_ID】 {db_id}\n【Schema】\n'
        schema_text += '\n'.join(schema_sections)
        schema_text += '\n【Foreign keys】\n'
        schema_text += '\n'.join(foreign_keys)
        
        return schema_text
        
    finally:
        conn.close()

def file_exists(file_path):
    return os.path.isfile(file_path)

if __name__ == '__main__':
    dataset_df = load_data(DATASET)
    print(f"Number of data samples: {dataset_df.shape[0]}")
    processed_data = []
    
    # Start processing data
    for index, row in dataset_df.iterrows():
        print(f"-------------------------------------------Processing index {index}-------------------------------------------")
        db_id = row['db_id']
        question = row['question']
        query = row["query"]
        database_schema = extract_sqlite_schema(f"./database/{db_id}/{db_id}.sqlite", db_id, 2)
        
        data_entry = {
            "db_id": db_id,
            "question": question,
            "query": query,
            "database": database_schema
        }
        processed_data.append(data_entry)
    
    # Write processed data back to file
    with open('processed_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    
    print("Data has been written to processed_dataset.json file.")
