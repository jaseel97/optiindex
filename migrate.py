import pymongo
from pymongo import MongoClient
from faker import Faker
import random
from datetime import datetime

# Connection to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['benchmark_db1']

# Initialize Faker
faker = Faker()

# Collection definitions
collections = {
    'collection_1': [
        ('name', faker.name),
        ('address', faker.address),
        ('email', faker.email),
        ('age', lambda: random.randint(18, 80))
    ],
    'collection_2': [
        ('company', faker.company),
        ('price', lambda: round(random.uniform(10.0, 1000.0), 2)),
        ('quantity', lambda: random.randint(1, 100)),
        ('in_stock', lambda: random.choice([True, False]))
    ],
    'collection_3': [
        ('date', lambda: datetime.combine(faker.date_this_decade(), datetime.min.time())),
        ('transaction_id', faker.uuid4),
        ('amount', lambda: round(random.uniform(100.0, 10000.0), 2)),
        ('currency', lambda: random.choice(['USD', 'EUR', 'GBP', 'JPY']))
    ],
    'collection_4': [
        ('username', faker.user_name),
        ('password', faker.password),
        ('last_login', faker.date_time_this_year),
        ('is_active', lambda: random.choice([True, False])),
        ('role', lambda: random.choice(['admin', 'user', 'guest']))
    ],
    'collection_5': [
        ('product_name', faker.word),
        ('category', lambda: random.choice(['electronics', 'furniture', 'clothing', 'food'])),
        ('rating', lambda: round(random.uniform(1.0, 5.0), 1)),
        ('review_count', lambda: random.randint(0, 500)),
        ('release_date', lambda: datetime.combine(faker.date_this_century(), datetime.min.time())),
        ('discontinued', lambda: random.choice([True, False]))
    ]
}

# Function to create and populate collections
def create_and_populate_collections(db, collections, num_records=200000):
    for collection_name, fields in collections.items():
        collection = db[collection_name]
        records = []
        for _ in range(num_records):
            record = {field_name: field_generator() for field_name, field_generator in fields}
            records.append(record)
        collection.insert_many(records)
        print(f'Inserted {num_records} records into {collection_name}')

# Create and populate collections
create_and_populate_collections(db, collections)

print("Database setup and data population complete.")
