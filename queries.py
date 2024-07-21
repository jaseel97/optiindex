import pymongo
from pymongo import MongoClient
from datetime import datetime
import json

# Connection to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['benchmark_db']

# Example queries with SQL-like representations
queries = [
    {'description': 'Find one document from collection_1',
     'query': lambda db: db['collection_1'].find_one(),
     'sql': 'SELECT * FROM collection_1 LIMIT 1'},
    
    {'description': 'Count documents in collection_2 where price > 500',
     'query': lambda db: db['collection_2'].count_documents({'price': {'$gt': 500}}),
     'sql': 'SELECT COUNT(*) FROM collection_2 WHERE price > 500'},
    
    {'description': 'Find documents from collection_3 with amount between 1000 and 5000',
     'query': lambda db: list(db['collection_3'].find({'amount': {'$gte': 1000, '$lte': 5000}})),
     'sql': 'SELECT * FROM collection_3 WHERE amount BETWEEN 1000 AND 5000'},
    
    {'description': 'Aggregate on collection_4 to count users by role',
     'query': lambda db: list(db['collection_4'].aggregate([
         {'$group': {'_id': '$role', 'count': {'$sum': 1}}}
     ])),
     'sql': 'SELECT role, COUNT(*) FROM collection_4 GROUP BY role'},
    
    {'description': 'Find products in collection_5 with rating > 4 and in stock',
     'query': lambda db: list(db['collection_5'].find({'rating': {'$gt': 4}, 'discontinued': False})),
     'sql': 'SELECT * FROM collection_5 WHERE rating > 4 AND discontinued = FALSE'},

    # Write Queries
    {'description': 'Update email for a user in collection_1',
     'query': lambda db: db['collection_1'].update_one({'name': 'John Doe'}, {'$set': {'email': 'newemail@example.com'}}),
     'sql': 'UPDATE collection_1 SET email = "newemail@example.com" WHERE name = "John Doe"'},
    
    {'description': 'Increase price by 10% for products in collection_2 with price < 100',
     'query': lambda db: db['collection_2'].update_many({'price': {'$lt': 100}}, {'$mul': {'price': 1.1}}),
     'sql': 'UPDATE collection_2 SET price = price * 1.1 WHERE price < 100'},
    
    {'description': 'Delete documents from collection_3 with amount < 500',
     'query': lambda db: db['collection_3'].delete_many({'amount': {'$lt': 500}}),
     'sql': 'DELETE FROM collection_3 WHERE amount < 500'},
    
    {'description': 'Insert a new user into collection_4',
     'query': lambda db: db['collection_4'].insert_one({
         'username': 'new_user',
         'password': 'securepass',
         'last_login': datetime.now(),
         'is_active': True,
         'role': 'user'
     }),
     'sql': 'INSERT INTO collection_4 (username, password, last_login, is_active, role) VALUES ("new_user", "securepass", NOW(), TRUE, "user")'},
    
    {'description': 'Add a review count field to documents in collection_5',
     'query': lambda db: db['collection_5'].update_many({}, {'$set': {'review_count': 0}}),
     'sql': 'UPDATE collection_5 SET review_count = 0'},

    # Additional Queries
    {'description': 'Find documents in collection_1 where age > 30',
     'query': lambda db: list(db['collection_1'].find({'age': {'$gt': 30}})),
     'sql': 'SELECT * FROM collection_1 WHERE age > 30'},
    
    {'description': 'Count documents in collection_2 where quantity < 10',
     'query': lambda db: db['collection_2'].count_documents({'quantity': {'$lt': 10}}),
     'sql': 'SELECT COUNT(*) FROM collection_2 WHERE quantity < 10'},
    
    {'description': 'Find documents from collection_3 with currency USD',
     'query': lambda db: list(db['collection_3'].find({'currency': 'USD'})),
     'sql': 'SELECT * FROM collection_3 WHERE currency = "USD"'},
    
    {'description': 'Aggregate on collection_4 to count active users',
     'query': lambda db: list(db['collection_4'].aggregate([
         {'$match': {'is_active': True}},
         {'$group': {'_id': None, 'count': {'$sum': 1}}}
     ])),
     'sql': 'SELECT COUNT(*) FROM collection_4 WHERE is_active = TRUE'},
    
    {'description': 'Find products in collection_5 with review count > 100',
     'query': lambda db: list(db['collection_5'].find({'review_count': {'$gt': 100}})),
     'sql': 'SELECT * FROM collection_5 WHERE review_count > 100'},
    
    {'description': 'Update address for a user in collection_1',
     'query': lambda db: db['collection_1'].update_one({'name': 'Jane Doe'}, {'$set': {'address': '123 New Street'}}),
     'sql': 'UPDATE collection_1 SET address = "123 New Street" WHERE name = "Jane Doe"'},
    
    {'description': 'Decrease quantity by 5 for products in collection_2 with quantity > 20',
     'query': lambda db: db['collection_2'].update_many({'quantity': {'$gt': 20}}, {'$inc': {'quantity': -5}}),
     'sql': 'UPDATE collection_2 SET quantity = quantity - 5 WHERE quantity > 20'},
    
    {'description': 'Delete documents from collection_3 with date before 2020',
     'query': lambda db: db['collection_3'].delete_many({'date': {'$lt': datetime(2020, 1, 1)}}),
     'sql': 'DELETE FROM collection_3 WHERE date < "2020-01-01"'},
    
    {'description': 'Insert a new product into collection_5',
     'query': lambda db: db['collection_5'].insert_one({
         'product_name': 'new_product',
         'category': 'electronics',
         'rating': 4.5,
         'review_count': 10,
         'release_date': datetime.now(),
         'discontinued': False
     }),
     'sql': 'INSERT INTO collection_5 (product_name, category, rating, review_count, release_date, discontinued) VALUES ("new_product", "electronics", 4.5, 10, NOW(), FALSE)'},
    
    {'description': 'Add a discount field to documents in collection_2',
     'query': lambda db: db['collection_2'].update_many({}, {'$set': {'discount': 0}}),
     'sql': 'UPDATE collection_2 SET discount = 0'},

    # Join Queries
    {'description': 'Join users and orders collections to find users and their orders',
     'query': lambda db: list(db['users'].aggregate([
         {
             '$lookup': {
                 'from': 'orders',
                 'localField': '_id',
                 'foreignField': 'user_id',
                 'as': 'user_orders'
             }
         }
     ])),
     'sql': 'SELECT users.*, orders.* FROM users LEFT JOIN orders ON users._id = orders.user_id'}
]

# Function to analyze fields in queries
def fieldAnalyzer(queries):
    field_analysis = {}

    for query in queries:
        coll_name = None
        query_op = None
        query_fields = []

        # Check the SQL representation to determine the operation and fields involved
        sql_query = query['sql']
        if sql_query.startswith('SELECT'):
            query_op = 'read'
            coll_name = sql_query.split('FROM')[1].split()[0]
            fields = sql_query.split('SELECT')[1].split('FROM')[0].strip()
            if fields != '*':
                query_fields = [field.strip() for field in fields.split(',')]
        elif sql_query.startswith('UPDATE'):
            query_op = 'update'
            coll_name = sql_query.split('UPDATE')[1].split()[0]
            query_fields = [field.split('=')[0].strip() for field in sql_query.split('SET')[1].split('WHERE')[0].split(',')]
        elif sql_query.startswith('DELETE'):
            query_op = 'delete'
            coll_name = sql_query.split('FROM')[1].split()[0]
        elif sql_query.startswith('INSERT'):
            query_op = 'insert'
            coll_name = sql_query.split('INTO')[1].split('(')[0].strip()
            query_fields = [field.strip() for field in sql_query.split('(')[1].split(')')[0].split(',')]
        elif 'JOIN' in sql_query:
            query_op = 'join'
            coll_name = sql_query.split('FROM')[1].split()[0]
            query_fields = [field.strip() for field in sql_query.split('ON')[1].split('=')]

        if coll_name:
            if coll_name not in field_analysis:
                field_analysis[coll_name] = {}

            for field in query_fields:
                if field not in field_analysis[coll_name]:
                    field_analysis[coll_name][field] = {
                        'read': 0,
                        'update': 0,
                        'insert': 0,
                        'delete': 0,
                        'join': 0,
                        'aggregation': 0,
                        'cardinality': 0
                    }

                if query_op:
                    field_analysis[coll_name][field][query_op] += 1

    return field_analysis

# Function to save the dictionary to a file
def saveFieldAnalysis(field_analysis, filename='field_analysis.json'):
    with open(filename, 'w') as file:
        json.dump(field_analysis, file, indent=4)

# Execute example queries and analyze fields
for query in queries:
    try:
        result = query['query'](db)
        print(f"{query['description']}: {result}")
    except Exception as e:
        print(f"Error executing query '{query['description']}': {e}")

field_analysis = fieldAnalyzer(queries)
saveFieldAnalysis(field_analysis)

print("Field analysis saved to 'field_analysis.json'")
