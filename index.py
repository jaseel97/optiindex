def create_single_field_index(mongo_client, db_name, collection_name, field_name):
    db = mongo_client[db_name]
    collection = db[collection_name]

    indexes = collection.index_information()
    if any(field_name in index['key'][0] for index in indexes.values()):
        print(f"Index on '{field_name}' already exists.")
    else:
        collection.create_index([(field_name, 1)])
        print(f"Index on '{field_name}' created successfully.")

def delete_single_field_index(mongo_client, db_name, collection_name, field_name):
    db = mongo_client[db_name]
    collection = db[collection_name]

    indexes = collection.index_information()
    for index_name, index_info in indexes.items():
        if field_name in index_info['key'][0]:
            collection.drop_index(index_name)
            print(f"Index on '{field_name}' deleted successfully.")
            return

    print(f"No index found on '{field_name}' to delete.")