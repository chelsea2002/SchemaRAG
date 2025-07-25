import spacy
from arg import main_args


args = main_args()
def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


def generate_mappings(schema_linkings):
    result = []
    for key, value in schema_linkings.items():
        if key == 'value':
            result.append(f'Values [{value}] will be used')
        elif args.dataset == 'bird_dev':
            if key == 'column':
                result.append(f'"column":[{value} will be used')
        else:
            for field_type, field_values in value.items():
                if field_values:
                    for field in field_values:
                        result.append(f'"{field_type}": [{field}] will be used')
    return result

def mask_entities(text):
    # Load model
    nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)

    # Define POS tags and entity types to preserve
    preserved_pos = {'NOUN', 'PROPN'}  # Preserve nouns and proper nouns

    # Store results list
    result_tokens = []

    for token in doc:
        # If it's a noun or proper noun, use mask
        if token.pos_ in preserved_pos or token.ent_type_:
            # Prioritize entity type, if no entity type, use POS tag
            mask_type = token.ent_type_ if token.ent_type_ else token.pos_
            result_tokens.append(f'[{mask_type.upper()}]')
        else:
            # Keep non-target words unchanged
            result_tokens.append(token.text)

    return ' '.join(result_tokens)


import sqlite3


def validate_sql_statement(sql_query: str,db_id,data):
    """
    Validate SQL statement validity

    Parameters:
        sql_query: SQL query to validate

    Returns:
        bool(validity status, error message)
    """

    # Preprocessing: remove extra whitespace
    sql_query = sql_query.strip()

    # Basic check: whether SQL statement is empty
    if not sql_query:
        return False, "SQL statement cannot be empty"

    # Basic syntax check
    basic_checks = [
        # Check if parentheses match
        (sql_query.count('(') != sql_query.count(')'), "Parentheses don't match"),
        # Check if quotes appear in pairs
        (sql_query.count("'") % 2 != 0, "Single quotes don't match"),
        (sql_query.count('"') % 2 != 0, "Double quotes don't match")
    ]

    for condition, error_message in basic_checks:
        if condition:
            return False, error_message


    # Use SQLite for actual syntax validation
    try:
        conn = sqlite3.connect(f'data/{data}/database/{db_id}/{db_id}.sqlite')
        cursor = conn.cursor()
        cursor.execute(sql_query)
        conn.close()
        return 1.0

    except:
        return 0.0
    finally:
        # Ensure connection is closed
        if conn:
            conn.close()


import bm25s
import nltk
from typing import Any, Union, List, Dict
from nltk.tokenize import word_tokenize
import random
import logging


def get_schema_dict(db_path: str) -> Dict:
    """
    The function construct database schema from the database sqlite file in the form of dict.

    Arguments:
        db_path (str): The database sqlite file path.
    Returns:
        db_schema_dict (Dict[str, Dict[str, str]]): database schema dictionary whose keys are table names and values are dict with column names keys and data type with as values.
    """
    # Connecting to the sqlite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query to extract the names of all tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    table_names = [table[0] for table in tables]

    # Dictionary to hold table names and their CREATE TABLE statements
    db_schema_dict = {}

    for table_name in table_names:
        cursor.execute(f"PRAGMA table_info(`{table_name}`);")
        table_info = cursor.fetchall()  # in table_info, each row indicate (cid, column name, type, notnull, default value, is_PK)
        # print(f"Table {table_name} info: \n", table_info)
        db_schema_dict[table_name] = {col_item[1]: col_item[2] for col_item in table_info}

    # Close the connection
    cursor.close()
    conn.close()

    return db_schema_dict

def execute_sql(db_path: str, sql: str, fetch: Union[str, int] = "all") -> Any:
    """
    Executes an SQL query on a database and fetches results.

    Arguments:
        db_path (str): The database sqlite file path.
        sql (str): The SQL query to execute.
        fetch (Union[str, int]): How to fetch the results. Options are "all", "one", "random", or an integer.

    Returns:
        resutls: SQL execution results .
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            if fetch == "all":
                return cursor.fetchall()
            elif fetch == "one":
                return cursor.fetchone()
            elif fetch == "random":
                samples = cursor.fetchmany(10)
                return random.choice(samples) if samples else []
            elif isinstance(fetch, int):
                return cursor.fetchmany(fetch)
            else:
                raise ValueError("Invalid fetch argument. Must be 'all', 'one', 'random', or an integer.")
    except Exception as e:
        logging.error(f"Error in execute_sql: {e}\n db_path: {db_path}\n SQL: {sql}")
        raise e

def extract_db_samples_enriched_bm25(question: str,  db_path: str, schema_dict: Dict,
                                     evidence: str,sample_limit: int) -> str:
    # ,table_match: list
    """
    The function extract distict samples for given schema items from the database by ranking values using BM25.
    Ranking is not done seperately for all values of each table.column

    Arguments:
        question (str): considered natural language question
        evidence (str): given evidence about the question
        db_path (str): The database sqlite file path.
        schema_dict (Dict[str, List[str]]): Database schema dictionary where keys are table names and values are lists of column names

    Returns:
        db_samples (str): concatenated strings gives samples from each column
    """
    db_samples = "\n"

    question = question.replace('\"', '').replace("\'", "").replace("`", "")+ " " + evidence
    question_and_evidence = question 
    tokenized_question_evidence = word_tokenize(question_and_evidence)


    for table, col_list in schema_dict.items():
        # if table not in table_match:
        #     continue
        db_samples = db_samples + f"## {table} table samples:\n"
        for col in col_list:
            # if f"{table}.{col}" not in columns:
            #     continue
            try:
                col_distinct_values = execute_sql(db_path,
                                                  f"SELECT DISTINCT `{col}` FROM `{table}`")  # extract all distinct values
                col_distinct_values = [str(value_tuple[0]) if value_tuple and value_tuple[0] else 'NULL' for value_tuple
                                       in col_distinct_values]
                if 'NULL' in col_distinct_values:
                    isNullExist = True
                else:
                    isNullExist = False

                if len(col_distinct_values) > 0:
                    average_length = sum(len(value) for value in col_distinct_values) / len(col_distinct_values)
                else:
                    average_length = 0
                if average_length > 600:
                    col_distinct_values = [col_distinct_values[0]]

                if len(col_distinct_values) > sample_limit:
                    corpus = col_distinct_values.copy()
                    corpus = [f'{table} {col} {val}' for val in corpus]
                    corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
                    # Create BM25 retriever and build index
                    retriever = bm25s.BM25()
                    retriever.index(corpus_tokens)
                    if isinstance(tokenized_question_evidence, list):
                        # If it's a tokenized list, rejoin as string then tokenize
                        query_str = " ".join(tokenized_question_evidence)
                        query_tokens = bm25s.tokenize(query_str, stopwords="en")
                    else:
                        # If it's a string, tokenize directly
                        query_tokens = bm25s.tokenize(tokenized_question_evidence, stopwords="en")
    
                    # Retrieve top-n results
                    results, scores = retriever.retrieve(query_tokens, k=sample_limit)
                    col_distinct_values = results[0].tolist()
                    if isNullExist:
                        col_distinct_values.append("NULL")

                db_samples = db_samples + f"# Example values for '{table}'.'{col}' column: " + str(
                    col_distinct_values) + "\n"
            except Exception as e:
                sql = f"SELECT DISTINCT `{col}` FROM `{table}`"
                logging.error(f"Error in extract_db_samples_enriched_bm25: {e}\n SQL: {sql}")
                error = str(e)


    return db_samples
