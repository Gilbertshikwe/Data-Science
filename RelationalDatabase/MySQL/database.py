import mysql.connector
from mysql.connector import Error
import pandas as pd
import matplotlib.pyplot as plt #type:ignore

def create_connection(host_name, user_name, password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=password,
            database=db_name
        )
        if connection.is_connected():
            print("Connection to MySQL DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection

def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query executed successfully")
    except Error as e:
        print(f"The error '{e}' occurred")

def insert_data(connection, query, data):
    cursor = connection.cursor()
    try:
        cursor.execute(query, data)
        connection.commit()
        print("Data inserted successfully")
    except Error as e:
        print(f"The error '{e}' occurred")

def read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Error as e:
        print(f"The error '{e}' occurred")

def update_data(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Data updated successfully")
    except Error as e:
        print(f"The error '{e}' occurred")

def delete_data(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Data deleted successfully")
    except Error as e:
        print(f"The error '{e}' occurred")

# Establish connection to MySQL
host = "localhost"
user = "root"
password = "Joykamau1"
database = "college"
connection = create_connection(host, user, password, database)

# Create users table
create_table_query = """
CREATE TABLE IF NOT EXISTS users (
  id INT AUTO_INCREMENT, 
  name TEXT NOT NULL, 
  age INT, 
  gender TEXT, 
  nationality TEXT, 
  PRIMARY KEY (id)
) ENGINE = InnoDB
"""
execute_query(connection, create_table_query)


# Insert sample data
insert_user_query = """
INSERT INTO users (name, age, gender, nationality)
VALUES (%s, %s, %s, %s)
"""

user_data = [
    ("Jane Smith", 35, "Female", "Canadian"),
    ("Michael Johnson", 42, "Male", "British"),
    ("Emily Davis", 27, "Female", "Australian"),
    ("David Wilson", 31, "Male", "American"),
    ("Sophia Thompson", 24, "Female", "New Zealander"),
    ("William Brown", 38, "Male", "Irish"),
    ("Olivia Taylor", 29, "Female", "South African"),
    ("James Anderson", 45, "Male", "Scottish"),
    ("Emma Miller", 33, "Female", "German")
]

for data in user_data:
    insert_data(connection, insert_user_query, data)

# Query data and load into pandas DataFrame
select_users_query = "SELECT * from users"
users = read_query(connection, select_users_query)
columns = ["id", "name", "age", "gender", "nationality"]
df = pd.DataFrame(users, columns=columns)
print("Data from 'users' table:")
print(df)


#Data Science workflows
# Assuming we have a table with user data

# Load the data
query = "SELECT * FROM users"
users_data = read_query(connection, query)
df = pd.DataFrame(users_data, columns=columns)

# Basic data analysis
print(df.describe())

# Filtering data
filtered_df = df[df['age'] > 35]
print(filtered_df)

# Plotting (requires matplotlib)

df['age'].hist()
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Update data
update_user_query = """
UPDATE users
SET age = 29
WHERE name = 'John Doe'
"""
update_data(connection, update_user_query)

# Delete data
delete_user_query = "DELETE FROM users WHERE name = 'John Doe'"
delete_data(connection, delete_user_query)

# Close connection
connection.close()


