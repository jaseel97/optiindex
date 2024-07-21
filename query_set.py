from datetime import datetime

query_list = [
    'SELECT * FROM collection_1 LIMIT 1',
    'SELECT COUNT(*) FROM collection_2 WHERE price > 500',
    'SELECT * FROM collection_3 WHERE amount BETWEEN 1000 AND 5000',
    'SELECT role, COUNT(*) FROM collection_4 GROUP BY role',
    'SELECT * FROM collection_5 WHERE rating > 4 AND discontinued = FALSE',
    'UPDATE collection_1 SET email = "newemail@example.com" WHERE name = "John Doe"',
    'UPDATE collection_2 SET price = price * 1.1 WHERE price < 100',
    'DELETE FROM collection_3 WHERE amount < 500',
    'INSERT INTO collection_4 (username, password, last_login, is_active, role) VALUES ("new_user", "securepass", NOW(), TRUE, "user")',
    'UPDATE collection_5 SET review_count = 0',
    'SELECT * FROM collection_1 WHERE age > 30',
    'SELECT COUNT(*) FROM collection_2 WHERE quantity < 10',
    'SELECT * FROM collection_3 WHERE currency = "USD"',
    'SELECT COUNT(*) FROM collection_4 WHERE is_active = TRUE',
    'SELECT * FROM collection_5 WHERE review_count > 100',
    'UPDATE collection_1 SET address = "123 New Street" WHERE name = "Jane Doe"',
    'UPDATE collection_2 SET quantity = quantity - 5 WHERE quantity > 20',
    'DELETE FROM collection_3 WHERE date < "2020-01-01"',
    'INSERT INTO collection_5 (product_name, category, rating, review_count, release_date, discontinued) VALUES ("new_product", "electronics", 4.5, 10, NOW(), FALSE)',
    'UPDATE collection_2 SET discount = 0',
    'SELECT users.*, orders.* FROM users LEFT JOIN orders ON users._id = orders.user_id',
    'SELECT age, COUNT(*) FROM collection_1 GROUP BY age',
    'SELECT company, SUM(price) FROM collection_2 GROUP BY company',
    'SELECT currency, AVG(amount) FROM collection_3 GROUP BY currency',
    'SELECT category, MAX(rating) FROM collection_5 GROUP BY category',
    'SELECT role, COUNT(*) FROM collection_4 WHERE is_active = TRUE GROUP BY role',
    'SELECT AVG(age) FROM collection_1',
    'SELECT SUM(quantity) FROM collection_2 WHERE in_stock = TRUE',
    'SELECT currency, AVG(amount), SUM(amount) FROM collection_3 GROUP BY currency',
    'SELECT category, COUNT(*) FROM collection_5 WHERE discontinued = TRUE GROUP BY category',
    'SELECT role, MAX(last_login), MIN(last_login) FROM collection_4 GROUP BY role'
]
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
     'sql': 'SELECT users.*, orders.* FROM users LEFT JOIN orders ON users._id = orders.user_id'},
    {
        'description': 'Count number of documents in collection_1 by age group',
        'query': lambda db: db['collection_1'].aggregate([
            {'$group': {'_id': '$age', 'count': {'$sum': 1}}}
        ]),
        'sql': 'SELECT age, COUNT(*) FROM collection_1 GROUP BY age'
    },
    {
        'description': 'Sum total price in collection_2 grouped by company',
        'query': lambda db: db['collection_2'].aggregate([
            {'$group': {'_id': '$company', 'total_price': {'$sum': '$price'}}}
        ]),
        'sql': 'SELECT company, SUM(price) FROM collection_2 GROUP BY company'
    },
    {
        'description': 'Average transaction amount in collection_3 grouped by currency',
        'query': lambda db: db['collection_3'].aggregate([
            {'$group': {'_id': '$currency', 'avg_amount': {'$avg': '$amount'}}}
        ]),
        'sql': 'SELECT currency, AVG(amount) FROM collection_3 GROUP BY currency'
    },
    {
        'description': 'Find the maximum rating for each category in collection_5',
        'query': lambda db: db['collection_5'].aggregate([
            {'$group': {'_id': '$category', 'max_rating': {'$max': '$rating'}}}
        ]),
        'sql': 'SELECT category, MAX(rating) FROM collection_5 GROUP BY category'
    },
    {
        'description': 'Count active users by role in collection_4',
        'query': lambda db: db['collection_4'].aggregate([
            {'$match': {'is_active': True}},
            {'$group': {'_id': '$role', 'count': {'$sum': 1}}}
        ]),
        'sql': 'SELECT role, COUNT(*) FROM collection_4 WHERE is_active = TRUE GROUP BY role'
    },
    {
        'description': 'Average age in collection_1',
        'query': lambda db: db['collection_1'].aggregate([
            {'$group': {'_id': None, 'avg_age': {'$avg': '$age'}}}
        ]),
        'sql': 'SELECT AVG(age) FROM collection_1'
    },
    {
        'description': 'Total quantity of items in stock in collection_2',
        'query': lambda db: db['collection_2'].aggregate([
            {'$match': {'in_stock': True}},
            {'$group': {'_id': None, 'total_quantity': {'$sum': '$quantity'}}}
        ]),
        'sql': 'SELECT SUM(quantity) FROM collection_2 WHERE in_stock = TRUE'
    },
    {
        'description': 'Average and total amount of transactions by currency in collection_3',
        'query': lambda db: db['collection_3'].aggregate([
            {'$group': {'_id': '$currency', 'avg_amount': {'$avg': '$amount'}, 'total_amount': {'$sum': '$amount'}}}
        ]),
        'sql': 'SELECT currency, AVG(amount), SUM(amount) FROM collection_3 GROUP BY currency'
    },
    {
        'description': 'Count of discontinued products by category in collection_5',
        'query': lambda db: db['collection_5'].aggregate([
            {'$match': {'discontinued': True}},
            {'$group': {'_id': '$category', 'count': {'$sum': 1}}}
        ]),
        'sql': 'SELECT category, COUNT(*) FROM collection_5 WHERE discontinued = TRUE GROUP BY category'
    },
    {
        'description': 'Maximum and minimum login time by user role in collection_4',
        'query': lambda db: db['collection_4'].aggregate([
            {'$group': {'_id': '$role', 'max_login': {'$max': '$last_login'}, 'min_login': {'$min': '$last_login'}}}
        ]),
        'sql': 'SELECT role, MAX(last_login), MIN(last_login) FROM collection_4 GROUP BY role'
    }
]