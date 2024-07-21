import json
import re
from pymongo import MongoClient
from pymongo.errors import OperationFailure

import query_set

def analyze_sql_queries(sql_queries, mongo_client):
    db = mongo_client['benchmark_db']
    result = {}

    for idx,query in enumerate(sql_queries):
        # Basic regex to extract collection name (FROM clause)
        print(idx," -> ",extract_delete_fields(query))

def extract_where_fields(sql):
    where_clause = re.search(r'WHERE\s+([^;]+)', sql, re.IGNORECASE)
    if where_clause:
        conditions = where_clause.group(1)
        fields = re.findall(r'\b(\w+)\b\s*(?:=|>|<|>=|<=|<>|!=|BETWEEN|IN|IS)', conditions, re.IGNORECASE)
        return fields
    return []

def extract_aggregate_fields(sql):
    # Patterns to capture GROUP BY, ORDER BY, PARTITION BY, and aggregate functions
    group_by_pattern = r'GROUP BY\s+([^;]+?)(?:\s+ORDER BY|\s+LIMIT|\s*$)'
    order_by_pattern = r'ORDER BY\s+([^;]+?)(?:\s+LIMIT|\s*$)'
    partition_by_pattern = r'PARTITION BY\s+([^;]+?)(?:\s+ORDER BY|\s+LIMIT|\s*$)'
    aggregate_functions_pattern = r'\b(?:COUNT|SUM|AVG|MIN|MAX|STDDEV|VARIANCE|FIRST|LAST)\s*\(\s*(\w+)\s*\)'

    group_by_fields = []
    order_by_fields = []
    partition_by_fields = []
    aggregate_fields = []

    # Extract GROUP BY fields
    group_by_match = re.search(group_by_pattern, sql, re.IGNORECASE)
    if group_by_match:
        group_by_fields = re.findall(r'\b(\w+)\b', group_by_match.group(1))

    # Extract ORDER BY fields
    order_by_match = re.search(order_by_pattern, sql, re.IGNORECASE)
    if order_by_match:
        order_by_fields = re.findall(r'\b(\w+)\b', order_by_match.group(1))

    # Extract PARTITION BY fields
    partition_by_match = re.search(partition_by_pattern, sql, re.IGNORECASE)
    if partition_by_match:
        partition_by_fields = re.findall(r'\b(\w+)\b', partition_by_match.group(1))

    # Extract fields used in aggregate functions
    aggregate_fields = re.findall(aggregate_functions_pattern, sql, re.IGNORECASE)

    return remove_duplicates_preserve_order(group_by_fields+order_by_fields+partition_by_fields+aggregate_fields)

    # return {
    #     'group_by': list(set(group_by_fields)),
    #     'order_by': list(set(order_by_fields)),
    #     'partition_by':list(set(partition_by_fields)),
    #     'aggregate_functions': list(set(aggregate_fields))
    # }

def extract_join_fields(sql):
    join_pattern = r'\bJOIN\s+[^\s]+\s+ON\s+([^;]+?)(?:\s+WHERE|\s+GROUP BY|\s+ORDER BY|\s+LIMIT|\s*$)'

    join_fields = []

    # Extract JOIN conditions
    join_matches = re.findall(join_pattern, sql, re.IGNORECASE)
    for join_match in join_matches:
        # Find all fields in the JOIN condition
        fields = re.findall(r'\b(\w+\.\w+|\w+)\b', join_match)
        # Strip table name if present
        stripped_fields = [field.split('.')[-1] for field in fields]
        join_fields.extend(stripped_fields)

    return join_fields

def extract_insert_fields(sql):
    # Pattern to check if the query uses INSERT INTO without specifying fields
    insert_values_pattern = r'INSERT\s+INTO\s+[^\s]+\s+VALUES\s*\(.*\)'

    # Pattern to check for INSERT INTO with specified fields
    insert_fields_pattern = r'INSERT\s+INTO\s+[^\s]+\s*\(([^)]+)\)'

    # Check if the query inserts values without specifying fields
    if re.match(insert_values_pattern, sql, re.IGNORECASE):
        return ['*']

    insert_fields = []

    # Extract INSERT fields if specified
    insert_match = re.search(insert_fields_pattern, sql, re.IGNORECASE)
    if insert_match:
        fields = re.findall(r'\b(\w+)\b', insert_match.group(1))
        insert_fields.extend(fields)

    return insert_fields

def extract_delete_fields(sql):
    # Pattern to check if the query deletes all rows without a WHERE clause
    delete_all_pattern = r'DELETE\s+FROM\s+[^\s]+(?:\s*$|;)'

    # Pattern to check for DELETE operations with a WHERE clause
    delete_pattern = r'DELETE\s+FROM\s+[^\s]+\s+WHERE\s+([^;]+?)(?:\s+GROUP BY|\s+ORDER BY|\s+LIMIT|\s*$)'

    # Check if the query deletes all rows
    if re.match(delete_all_pattern, sql, re.IGNORECASE):
        return ['*']

    delete_fields = []

    # Extract DELETE conditions
    delete_match = re.search(delete_pattern, sql, re.IGNORECASE)
    if delete_match:
        # Find all fields in the DELETE condition, ensuring we exclude values
        fields = re.findall(r'\b(\w+)\b\s*(?:=|>|<|>=|<=|<>|!=|BETWEEN|IN|IS|LIKE)', delete_match.group(1))
        # Strip table name if present
        stripped_fields = [field.split('.')[-1] for field in fields]
        delete_fields.extend(stripped_fields)

    return delete_fields

def remove_duplicates_preserve_order(lst):
    seen = set()
    unique_list = []
    for item in lst:
        if item not in seen:
            unique_list.append(item)
            seen.add(item)
    return unique_list

if __name__ == "__main__":
    client = MongoClient('mongodb://localhost:27017/')

    # sql_queries = [query['sql'] for query in query_set.queries]
    analysis = analyze_sql_queries(query_set.query_list, client)

def saveFieldAnalysis(field_analysis, filename='field_analysis.json'):
    with open(filename, 'w') as file:
        json.dump(field_analysis, file, indent=4)
    # print(analysis)