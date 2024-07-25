from collections import defaultdict
from datetime import datetime
import json
import re
import time
from pymongo import MongoClient
from pymongo.errors import OperationFailure

import query_set

def add_index_info(db_conn, state):
    for collection_name, fields in state.items():
        collection = db_conn[collection_name]
        indexes = collection.index_information()
        indexed_fields = {key[0] for index in indexes.values() for key in index['key']}

        for field_name in fields:
            if field_name in indexed_fields:
                if field_name != '_id':
                    state[collection_name][field_name]['indexed'] = 1
            else:
                state[collection_name][field_name]['indexed'] = 0
    return state

def update_field_count(state, collection_name, fields, operation):
    for field in fields:
        if field in state[collection_name]:
            if operation not in state[collection_name][field]:
                state[collection_name][field][operation] = 0
            state[collection_name][field][operation] += 1


def add_operation_count_info(state, query_db):
    for collection_name, queries in query_db.items():
        for query in queries:
            sql = query['sql']
            where_fields = extract_where_fields(sql)
            insert_fields = extract_insert_fields(sql)
            delete_fields = extract_delete_fields(sql)
            join_fields = extract_join_fields(sql)
            aggregate_fields = extract_aggregate_fields(sql)

            update_field_count(state, collection_name, where_fields, 'where')
            update_field_count(state, collection_name, insert_fields, 'insert')
            update_field_count(state, collection_name, delete_fields, 'delete')
            update_field_count(state, collection_name, join_fields, 'join')
            update_field_count(state, collection_name, aggregate_fields, 'aggregation')

        operation_list = ['where','insert','delete','join','aggregation']
        for field_name in state[collection_name]:
            for operation in operation_list:
                if operation not in state[collection_name][field_name].keys():
                    state[collection_name][field_name][operation] = 0
    return state

def extract_where_fields(sql):
    where_clause = re.search(r'WHERE\s+([^;]+)', sql, re.IGNORECASE)
    if where_clause:
        conditions = where_clause.group(1)
        fields = re.findall(r'\b(\w+)\b\s*(?:=|>|<|>=|<=|<>|!=|BETWEEN|IN|IS)', conditions, re.IGNORECASE)
        return fields
    return []

def extract_aggregate_fields(sql):
    group_by_pattern = r'GROUP BY\s+([^;]+?)(?:\s+ORDER BY|\s+LIMIT|\s*$)'
    order_by_pattern = r'ORDER BY\s+([^;]+?)(?:\s+LIMIT|\s*$)'
    partition_by_pattern = r'PARTITION BY\s+([^;]+?)(?:\s+ORDER BY|\s+LIMIT|\s*$)'
    aggregate_functions_pattern = r'\b(?:COUNT|SUM|AVG|MIN|MAX|STDDEV|VARIANCE|FIRST|LAST)\s*\(\s*(\w+)\s*\)'

    group_by_fields = []
    order_by_fields = []
    partition_by_fields = []
    aggregate_fields = []

    group_by_match = re.search(group_by_pattern, sql, re.IGNORECASE)
    if group_by_match:
        group_by_fields = re.findall(r'\b(\w+)\b', group_by_match.group(1))

    order_by_match = re.search(order_by_pattern, sql, re.IGNORECASE)
    if order_by_match:
        order_by_fields = re.findall(r'\b(\w+)\b', order_by_match.group(1))

    partition_by_match = re.search(partition_by_pattern, sql, re.IGNORECASE)
    if partition_by_match:
        partition_by_fields = re.findall(r'\b(\w+)\b', partition_by_match.group(1))

    aggregate_fields = re.findall(aggregate_functions_pattern, sql, re.IGNORECASE)

    return remove_duplicates_preserve_order(group_by_fields+order_by_fields+partition_by_fields+aggregate_fields)

def extract_join_fields(sql):
    join_pattern = r'\bJOIN\s+[^\s]+\s+ON\s+([^;]+?)(?:\s+WHERE|\s+GROUP BY|\s+ORDER BY|\s+LIMIT|\s*$)'

    join_fields = []
    join_matches = re.findall(join_pattern, sql, re.IGNORECASE)
    for join_match in join_matches:
        fields = re.findall(r'\b(\w+\.\w+|\w+)\b', join_match)
        stripped_fields = [field.split('.')[-1] for field in fields]
        join_fields.extend(stripped_fields)

    return join_fields

def extract_insert_fields(sql):
    insert_values_pattern = r'INSERT\s+INTO\s+[^\s]+\s+VALUES\s*\(.*\)'
    insert_fields_pattern = r'INSERT\s+INTO\s+[^\s]+\s*\(([^)]+)\)'

    if re.match(insert_values_pattern, sql, re.IGNORECASE):
        return ['*']

    insert_fields = []
    insert_match = re.search(insert_fields_pattern, sql, re.IGNORECASE)
    if insert_match:
        fields = re.findall(r'\b(\w+)\b', insert_match.group(1))
        insert_fields.extend(fields)

    return insert_fields

def extract_delete_fields(sql):
    delete_all_pattern = r'DELETE\s+FROM\s+[^\s]+(?:\s*$|;)'
    delete_pattern = r'DELETE\s+FROM\s+[^\s]+\s+WHERE\s+([^;]+?)(?:\s+GROUP BY|\s+ORDER BY|\s+LIMIT|\s*$)'

    if re.match(delete_all_pattern, sql, re.IGNORECASE):
        return ['*']

    delete_fields = []
    delete_match = re.search(delete_pattern, sql, re.IGNORECASE)
    if delete_match:
        fields = re.findall(r'\b(\w+)\b\s*(?:=|>|<|>=|<=|<>|!=|BETWEEN|IN|IS|LIKE)', delete_match.group(1))
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

def extract_collection_fields(db_conn, query_db):
    collection_fields = {}
    for collection_name in query_db.keys():
        print(collection_name)
        document = db_conn[collection_name].find_one()
        collection_fields[collection_name] = list(document.keys())
        collection_fields[collection_name].remove('_id')
        print(collection_fields[collection_name])
    return collection_fields

def calculate_distinct_count(collection, field):
    try:
        pipeline = [
            {'$group': {'_id': f"${field}"}},
            {'$count': 'distinct_count'}
        ]
        result = list(collection.aggregate(pipeline))
        return result[0]['distinct_count'] if result else 0
    except OperationFailure as e:
        print(f"Error calculating distinct count for {field}: {e}")
        return 0

def infer_field_type(value):
    if isinstance(value, int):
        return 'int'
    elif isinstance(value, float):
        return 'float'
    elif isinstance(value, str):
        return 'str'
    elif isinstance(value, bool):
        return 'bool'
    elif isinstance(value, list):
        return 'list'
    elif isinstance(value, dict):
        return 'dict'
    elif isinstance(value, datetime):
        return 'datetime'
    else:
        return 'unknown'

def add_cardinality_and_type_info(db_conn, state):
    for collection_name, fields in state.items():
        collection = db_conn[collection_name]
        total_count = collection.count_documents({})

        field_cardinality = {}
        sample_document = collection.find_one()
        for field in fields:
            distinct_count = calculate_distinct_count(collection, field)
            print(collection_name, "->", field, ":", distinct_count, "/", total_count)
            cardinality_ratio = distinct_count / total_count if total_count > 0 else 0
            field_type = infer_field_type(sample_document.get(field)) if sample_document else 'unknown'
            field_cardinality[field] = {
                'cardinality': round(cardinality_ratio, 4),
                'type': field_type
            }
        state[collection_name] = field_cardinality
    return state

def execute_queries_og(db, query_db):
    metrics = {
        'executionTimeMillis': 0,
        'nReturned': 0,
        'totalKeysExamined': 0,
        'totalDocsExamined': 0
    }

    for collection, queries in query_db.items():
        ctr = 0;
        for query in queries:
            ctr+=1
            print(collection," -> ", ctr)
            if 'UPDATE' in query['sql'] or 'DELETE' in query['sql'] or 'INSERT INTO' in query['sql']:
                print('insert/update/delete')
                # start_time = time.time()
                # original_document = db[collection].find_one(query_info['filter'])
                # query_result = query_info['query'](db)
                # end_time = time.time()
            elif 'COUNT(*)' in query['sql']:
                print('count')
                filter_criteria = {'price': {'$gt': 500}}
                explain_plan = db.command(
                    'explain',
                    {
                        'aggregate': collection,
                        'pipeline': [
                            {'$match': filter_criteria},
                            {'$count': 'total_docs'}
                        ],
                        'cursor': {}
                    },
                    verbosity='executionStats'
                )
                metrics['executionTimeMillis'] += explain_plan['stages'][0]['$cursor']['executionStats']['executionTimeMillis']
                metrics['nReturned'] += explain_plan['stages'][0]['$cursor']['executionStats']['nReturned']
                metrics['totalKeysExamined'] += explain_plan['stages'][0]['$cursor']['executionStats']['totalKeysExamined']
                metrics['totalDocsExamined'] += explain_plan['stages'][0]['$cursor']['executionStats']['totalDocsExamined']
            elif 'pipeline' in query.keys():
                print('aggregate')
                explain_plan = db.command(
                    'explain',
                    {
                        'aggregate': collection,
                        'pipeline': query['pipeline'],
                        'cursor': {}
                    },
                    verbosity='executionStats'
                )
                metrics['executionTimeMillis'] += explain_plan['executionStats']['executionTimeMillis']
                metrics['nReturned'] += explain_plan['executionStats']['nReturned']
                metrics['totalKeysExamined'] += explain_plan['executionStats']['totalKeysExamined']
                metrics['totalDocsExamined'] += explain_plan['executionStats']['totalDocsExamined']
            else:
                print('read')
                explain_plan = query['query'](db).explain()
                metrics['executionTimeMillis'] += explain_plan['executionStats']['executionTimeMillis']
                metrics['nReturned'] += explain_plan['executionStats']['nReturned']
                metrics['totalKeysExamined'] += explain_plan['executionStats']['totalKeysExamined']
                metrics['totalDocsExamined'] += explain_plan['executionStats']['totalDocsExamined']

    return metrics

def saveAsJSON(field_analysis, filename='state.json'):
    with open(filename, 'w') as file:
        json.dump(field_analysis, file, indent=4)

def getStaticInfo(db_conn):
    state = extract_collection_fields(db_conn, query_set.query_db)
    state = add_cardinality_and_type_info(db_conn, state)
    state = add_operation_count_info(state, query_set.query_db)
    return state

def addIndexInfo(db_conn, state):
    state = add_index_info(db_conn, state)
    return state

def getQueryMetrics(db_conn):
    metrics = execute_queries_og(db_conn, query_set.query_db)
    return metrics

if __name__ == "__main__":
    client = MongoClient('mongodb://localhost:27017/')
    db_conn = client['benchmark_db1']

    state = getStaticInfo(db_conn)
    state = addIndexInfo(db_conn, state)
    saveAsJSON(state)

    metrics = getQueryMetrics(db_conn)
    print(metrics)
    client.close()

    # # #constants
    # state = extract_collection_fields(db_conn, query_set.query_db)
    # state = add_cardinality_and_type_info(db_conn, state)
    # state = add_operation_count_info(state, query_set.query_db)
    # # #variable
    # state = add_index_info(db_conn, state)
    # saveAsJSON(state)
    # metrics = execute_queries(db_conn, query_set.query_db)
    # print(metrics)
    # client.close()

