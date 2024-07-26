from pymongo import MongoClient

import query_set

def create_single_field_index(db_conn, collection_name, field_name):
    collection = db_conn[collection_name]

    indexes = collection.index_information()
    if any(field_name in index['key'][0] for index in indexes.values()):
        # print(f"Index on '{field_name}' already exists.")
        return 0
    else:
        collection.create_index([(field_name, 1)])
        # print(f"Index on '{field_name}' created successfully.")
        return 1

def delete_single_field_index(db_conn, collection_name, field_name):
    collection = db_conn[collection_name]

    indexes = collection.index_information()
    for index_name, index_info in indexes.items():
        if field_name in index_info['key'][0]:
            collection.drop_index(index_name)
            # print(f"Index on '{field_name}' deleted successfully.")
            return 1

    # print(f"No index found on '{field_name}' to delete.")
    return 0

def reset_index_config(db_conn, state):
    print("resetting indices")
    for collection in state.keys():
        for field in state[collection]:
            indexed = state[collection][field]['indexed']
            if(indexed):
                create_single_field_index(db_conn, collection, field)
            else:
                delete_single_field_index(db_conn, collection, field)

def clear_all_indices(db_conn, collections):
    for collection_name in collections:
        collection = db_conn[collection_name]
        indexes = collection.index_information()
        for index_name, index_info in indexes.items():
            if(index_name != '_id_'):
                collection.drop_index(index_name)
                print(f"Index on '{index_name}' deleted successfully.")

if __name__ == "__main__":
    client = MongoClient('mongodb://localhost:27017/')
    db_conn = client['benchmark_db1']
    # create_single_field_index(db_conn, 'collection_5', 'category')
    # create_single_field_index(db_conn, 'collection_1', 'address')
    clear_all_indices(db_conn, list(query_set.query_db.keys()))