# Optiindex

<!-- ![Project Logo](url/to/logo.png) -->

## Table of Contents
- [About](#about)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)


## About
ML Model to find optimal index allocation

## Getting Started

### Prerequisites
- MongoDB 7+
- Python 3+
    - pymongo
    - faker
    - torch

### Setup

1. Clone the repository
   ```sh
   git clone https://github.com/jaseel97/optiindex.git
   ```
2. Install MongoDB and Atlas
3. Install the following Python libraries
    ```sh
   pip install faker pymongo tensorflow keras
   ```
4. In MongoDB, create a new database called benchmark_db1 using mongosh or atlas
5. Run migrate.py to populate the db with sample values
6. Run state_creator.py to create a sample state to check everything works