# MySQL Basics

## 1. SQL Syntax

SQL (Structured Query Language) is a language used to manage and manipulate relational databases. Here are some fundamental SQL syntax concepts:

- **Creating Databases**: Use the `CREATE DATABASE` statement to create a new database.

  ```sql
  CREATE DATABASE my_database;
  ```

- **Creating Tables**: Use the `CREATE TABLE` statement to create a new table within a database.

  ```sql
  CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  ```

- **Inserting Data**: Use the `INSERT INTO` statement to insert new records into a table.

  ```sql
  INSERT INTO users (username, email) VALUES ('john_doe', 'john@example.com');
  ```

- **Querying Data**: Use the `SELECT` statement to retrieve data from one or more tables.

  ```sql
  SELECT * FROM users;
  ```

- **Updating Data**: Use the `UPDATE` statement to modify existing records in a table.

  ```sql
  UPDATE users SET email = 'john.doe@example.com' WHERE id = 1;
  ```

- **Deleting Data**: Use the `DELETE FROM` statement to remove records from a table.

  ```sql
  DELETE FROM users WHERE id = 1;
  ```

## 2. Data Types

MySQL supports various data types for storing different kinds of data. Some common data types include:

- **Numeric Types**: INT, FLOAT, DOUBLE, DECIMAL.
- **String Types**: VARCHAR, CHAR, TEXT.
- **Date and Time Types**: DATE, TIME, DATETIME, TIMESTAMP.
- **Boolean Type**: BOOL, BOOLEAN.

## 3. CRUD Operations

CRUD stands for Create, Read, Update, and Delete, which are the four basic functions of persistent storage. In MySQL:

- **Create**: Use `INSERT INTO` to add new records to a table.
- **Read**: Use `SELECT` to retrieve data from one or more tables.
- **Update**: Use `UPDATE` to modify existing records in a table.
- **Delete**: Use `DELETE FROM` to remove records from a table.

## 4. Constraints

Constraints are rules enforced on data columns to maintain the integrity and accuracy of the data. Some common constraints include:

- **Primary Key**: Uniquely identifies each record in a table.
- **Foreign Key**: Establishes a link between two tables.
- **Unique Key**: Ensures that all values in a column are distinct.
- **NOT NULL**: Ensures that a column cannot contain NULL values.

## 5. Functions and Operators

MySQL provides various built-in functions and operators for data manipulation and analysis. Some common functions include:

- **Aggregate Functions**: SUM, AVG, MIN, MAX, COUNT.
- **String Functions**: CONCAT, SUBSTRING, UPPER, LOWER.
- **Date Functions**: NOW, DATE_FORMAT, DATE_ADD, DATE_SUB.
- **Mathematical Functions**: ABS, CEIL, FLOOR, ROUND.

Learning these SQL basics will provide you with a strong foundation for working with MySQL databases and performing data manipulation tasks effectively. Practice writing SQL queries and experimenting with different data types and constraints to gain hands-on experience.


# Starting MySQL Server on Linux

To start the MySQL Server on Linux, follow these steps:

1. **Open Terminal**: Open the Terminal application on your Linux system. You can typically find it in the Applications menu or by searching for "Terminal".

2. **Run Command**: Use the following command to start the MySQL service:

```
sudo systemctl start mysql
```

This command will start the MySQL service on your Linux system.

3. **Verify Status**: You can verify that MySQL is running by executing the following command:

```
sudo systemctl status mysql
```

This command will display the status of the MySQL service, indicating whether it is running or not.

# Accessing MySQL Server on Linux

Once the MySQL Server is running, you can access it using the MySQL command-line client:

1. **Open Terminal**: If you haven't already, open the Terminal application.

2. **Connect to MySQL**: Use the following command to connect to the MySQL Server:

```
mysql -u root -p
```

You'll be prompted to enter the root password you set during the installation of MySQL Server. After entering the correct password, you'll be logged into the MySQL command-line client.

3. **MySQL Command-line Interface**: After successfully connecting, you'll be presented with the MySQL command-line interface, where you can execute SQL queries, create databases, tables, and perform various operations.

Note: If you encounter any issues or errors during this process, refer to the MySQL documentation or consult with your system administrator for further assistance.

# MySQL Connector Python Setup Guide

This guide will help you set up your environment to use the MySQL Connector for Python. It covers installation steps for both MySQL server and the required Python packages, as well as basic usage of the MySQL Connector for Python.

## Prerequisites

Before using the MySQL Connector for Python, ensure you have the following software installed on your system:

- Python 3.x
- pip (Python package installer)
- MySQL Server

## Step 1: Install MySQL Server

### On Ubuntu/Debian-based Systems

```bash
sudo apt-get update
sudo apt-get install mysql-server
```

### On Fedora/Red Hat-based Systems

```bash
sudo dnf install mysql-server
```

### On CentOS

```bash
sudo yum install mysql-server
```

### Start MySQL Service

```bash
sudo systemctl start mysql  # Use 'mysqld' if 'mysql' does not work
```

### Enable MySQL to Start on Boot

```bash
sudo systemctl enable mysql  # Use 'mysqld' if 'mysql' does not work
```

### Check MySQL Status

```bash
sudo systemctl status mysql  # Use 'mysqld' if 'mysql' does not work
```

The error `ERROR 1045 (28000): Access denied for user 'root'@'localhost' (using password: NO)` indicates that the root user requires a password, and you haven't provided one. If you don't know the current root password, you can reset it by following these steps:

### Step-by-Step Guide to Reset MySQL Root Password

1. **Stop the MySQL Server**:
   ```bash
   sudo systemctl stop mysql
   ```

2. **Start MySQL in Safe Mode**:
   Start MySQL without loading the grant tables, which store user privileges. This allows you to access the database without a password.
   ```bash
   sudo mysqld_safe --skip-grant-tables &
   ```

3. **Access MySQL as Root**:
   In a new terminal window, run:
   ```bash
   sudo mysql -u root
   ```

4. **Change the Root Password**:
   In the MySQL shell, change the root user's authentication method and set a new password.
   ```sql
   FLUSH PRIVILEGES;
   ALTER USER 'root'@'localhost' IDENTIFIED BY 'your_new_password';
   FLUSH PRIVILEGES;
   ```

   Replace `'your_new_password'` with your desired password.

5. **Exit the MySQL Shell**:
   ```sql
   EXIT;
   ```

6. **Stop MySQL Safe Mode**:
   Kill the MySQL safe mode process. You can find the process ID using `ps` and then kill it.
   ```bash
   sudo pkill mysqld
   ```

7. **Start the MySQL Server Normally**:
   ```bash
   sudo systemctl start mysql
   ```

8. **Test the New Password**:
   Ensure you can log in with the new password.
   ```bash
   mysql -u root -p
   ```

### Detailed Commands

Here are the commands in detail for your convenience:

```bash
# Step 1: Stop the MySQL Server
sudo systemctl stop mysql

# Step 2: Start MySQL in Safe Mode
sudo mysqld_safe --skip-grant-tables &

# Step 3: Access MySQL as Root
sudo mysql -u root

# Step 4: Change the Root Password
# In the MySQL shell
FLUSH PRIVILEGES;
ALTER USER 'root'@'localhost' IDENTIFIED BY 'your_new_password';
FLUSH PRIVILEGES;

# Step 5: Exit the MySQL Shell
EXIT;

# Step 6: Stop MySQL Safe Mode
sudo pkill mysqld

# Step 7: Start the MySQL Server Normally
sudo systemctl start mysql

# Step 8: Test the New Password
mysql -u root -p
```

Replace `'your_new_password'` with the password you want to set for the root user.


By following these steps, you should be able to reset the MySQL root password and resolve the authentication issue.

---

## Step 2: Install Python Packages

Open your terminal and run the following command to install `mysql-connector-python` and `pandas`:

```bash
pip install mysql-connector-python pandas
```

## Step 3: Using MySQL Connector Python

Below is a basic example of how to use the MySQL Connector for Python to connect to a MySQL database.

### Python Script: `connect_to_mysql.py`

```python
import mysql.connector
from mysql.connector import Error

def create_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        if connection.is_connected():
            print("Connection to MySQL DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")
    
    return connection

# Example usage
connection = create_connection("localhost", "root", "your_password", "your_database")
```

### Explanation

- **create_connection**: This function attempts to create a connection to the MySQL database using the provided host, username, password, and database name.
- **Example usage**: Replace `"localhost"`, `"root"`, `"your_password"`, and `"your_database"` with your actual MySQL credentials and database name.

## Step 4: Testing the Connection

To test the connection, run the `connect_to_mysql.py` script:

```bash
python connect_to_mysql.py
```

If the connection is successful, you will see the message "Connection to MySQL DB successful". If there is an error, the error message will be printed.

## Troubleshooting

- Ensure the MySQL server is running.
- Double-check your MySQL connection parameters (host, user, password, database).
- Check firewall settings to ensure the MySQL port (3306 by default) is not blocked.
- Verify MySQL server logs for any issues.

## Additional Resources

- [MySQL Connector/Python Documentation](https://dev.mysql.com/doc/connector-python/en/)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [MySQL Installation Guide](https://dev.mysql.com/doc/mysql-installation-excerpt/5.7/en/)

---

This README provides a step-by-step guide for setting up and using the MySQL Connector for Python, ensuring a smooth setup process.


# MySQL-Python Data Science Workflow

This README provides a step-by-step guide on how to use MySQL with Python for data science purposes. It covers connecting to a MySQL database, performing CRUD (Create, Read, Update, Delete) operations, and integrating SQL queries with Python for data manipulation and analysis. Additionally, it demonstrates how to use Python libraries like `pandas` for data manipulation and `matplotlib` for visualization.

## 1. Setting Up the Environment

Ensure you have the necessary packages installed:

```sh
pip install mysql-connector-python pandas matplotlib
```

## 2. Connecting to MySQL Database

Use the `mysql.connector` library to establish a connection to your MySQL database.

```python
import mysql.connector
from mysql.connector import Error

# Function to create a connection
def create_connection(host_name, user_name, user_password, db_name):
    ...
```

## 3. Creating Tables

Define a function to create a new table in the MySQL database.

```python
# Function to execute queries
def execute_query(connection, query):
    ...
```

## 4. Inserting Data

Define a function to insert data into the table.

```python
# Function to insert data
def insert_data(connection, query, data):
    ...
```

## 5. Querying Data

Retrieve data from the database using `SELECT` statements and load it into a `pandas` DataFrame for analysis.

```python
import pandas as pd

# Function to read query results
def read_query(connection, query):
    ...
```

## 6. Updating and Deleting Data

Define functions to update and delete data from the table.

```python
# Functions to update and delete data
def update_data(connection, query):
    ...

def delete_data(connection, query):
    ...
```

## 7. Integrating with Data Science Workflows

For data science workflows, perform data analysis and visualization using `pandas` and `matplotlib`.

This README provides a basic overview of using MySQL with Python for data science tasks. For more advanced usage and integration with machine learning workflows, additional steps and libraries may be required.

---

Feel free to expand upon this README with additional information or customization based on your specific project requirements.