# MongoDB Server Setup

This guide will walk you through the process of installing and starting the MongoDB server on your local machine.

## Prerequisites

Before proceeding, ensure that you have the following prerequisites installed:

- For Windows: Make sure you have a compatible version of Windows (7, 8, or 10) installed.
- For macOS: Make sure you have a compatible version of macOS (10.10 or later) installed.
- For Linux: Make sure you have a compatible version of your Linux distribution installed.

## Installation

### Windows

1. Download the MongoDB Community Server from the official website: https://www.mongodb.com/download-center/community
2. Choose the appropriate version for your system (e.g., Windows 64-bit x64) and download the `.msi` installer.
3. Run the downloaded `.msi` installer and follow the prompts to complete the installation.
4. Once the installation is complete, you should have MongoDB installed on your system.

### macOS

1. Download the MongoDB Community Server from the official website: https://www.mongodb.com/download-center/community
2. Choose the appropriate version for your system (e.g., macOS 64-bit x64) and download the `.tgz` package.
3. Extract the downloaded `.tgz` package to a desired location (e.g., `/usr/local/mongodb`).
4. Add the MongoDB binaries to your system's `PATH` environment variable by adding the following line to your shell configuration file (e.g., `~/.bash_profile`):
   ```
   export PATH=/usr/local/mongodb/bin:$PATH
   ```
5. Reload the shell configuration file: `source ~/.bash_profile`

### Linux

The installation process for Linux varies depending on your distribution. Refer to the official MongoDB documentation for detailed instructions: https://docs.mongodb.com/manual/administration/install-on-linux/

Here is a summarized process written as a README for installing and setting up MongoDB on a Linux system:

---

# MongoDB Installation and Setup Guide

This guide provides step-by-step instructions to install and set up MongoDB on a Linux system.

## Step 1: Install MongoDB Server

1. **Download MongoDB**: Visit the [MongoDB Community Download Page](https://www.mongodb.com/try/download/community).
2. **Select Version and Platform**: Choose the appropriate version for your operating system and download it.
3. **Follow Installation Instructions**: Follow the provided instructions to complete the installation for your specific OS.

## Step 2: Create the Data Directory

MongoDB requires a data directory to store its data.

1. **Create the Data Directory**:

   ```bash
   sudo mkdir -p /data/db
   ```

2. **Set Permissions for the Data Directory**:

   MongoDB needs appropriate permissions to access the data directory. Set the permissions by running:

   ```bash
   sudo chown -R $(whoami) /data/db
   ```

   This command changes the ownership of the `/data/db` directory and its contents to your current user account.

## Step 3: Start MongoDB Server

1. **Start MongoDB**:

   Start the MongoDB server by running the `mongod` command:

   ```bash
   mongod
   ```

Sure, here's a README section that explains how to stop the MongoDB server running on port 27017:

### Stopping the MongoDB Server

If you need to stop the MongoDB server that is running on port 27017, follow these steps:

1. **Find the Process ID (PID) of the Process Using Port 27017:**

   Open a terminal and run the following command to list the process ID (PID) of the process using port 27017:
   ```sh
   sudo lsof -i :27017
   ```

   This command will display the details of the process using port 27017. Look for the PID in the output. It will look something like this:
   ```
   COMMAND  PID  USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
   mongod   1234  mongodb  11u  IPv4  12345  0t0  TCP *:27017 (LISTEN)
   ```

   In this example, the PID is `1234`.

2. **Stop the Process:**

   Use the PID found in the previous step to stop the MongoDB server process. Replace `<PID>` with the actual process ID in the following command:
   ```sh
   sudo kill -9 <PID>
   ```

   For example, if the PID is `1234`, the command will be:
   ```sh
   sudo kill -9 1234
   ```

   This command forcibly stops the process running on port 27017.

### Note

Using `kill -9` is a forceful way to stop a process. It is generally recommended to use the proper shutdown command for MongoDB when possible to ensure data integrity:
1. Open another terminal window.
2. Connect to the MongoDB server using the MongoDB shell (`mongosh` or `mongo`).
   ```sh
   mongosh
   ```
   or for older versions:
   ```sh
   mongo
   ```
3. Use the `admin` database and issue the shutdown command:
   ```javascript
   use admin
   db.shutdownServer()
   ```

By following the proper shutdown procedure, you can ensure that MongoDB stops gracefully, maintaining data integrity.

## Troubleshooting

If you encounter issues starting MongoDB, follow these additional steps:

1. **Check for Running MongoDB Instances**:

   Use the following command to check if another MongoDB instance is already running and occupying the default port (27017):

   ```bash
   pgrep mongod
   ```

2. **Stop Running MongoDB Instances**:

   If you see any process IDs (PIDs), stop the running MongoDB instance using the `kill` command:

   ```bash
   sudo kill <PID>
   ```

   Replace `<PID>` with the actual process ID.

## Additional Resources

- [MongoDB Documentation](https://docs.mongodb.com/): For detailed information and advanced configurations.
- [MongoDB Community](https://community.mongodb.com/): For support and discussions.

---

By following these steps, you should be able to successfully install and set up MongoDB on your Linux system. If you encounter further issues, refer to the MongoDB documentation or seek assistance from the MongoDB community.

# Working with Documents and Collections in MongoDB

Working with documents and collections using the MongoDB shell (often referred to as the `mongo` shell) is fundamental for interacting with MongoDB databases. This README provides a step-by-step guide to help you get started with basic operations like inserting, querying, updating, and deleting documents within collections.

## Connecting to MongoDB

To start the `mongo` shell and connect to your MongoDB instance, open your terminal and run:

```sh
mongosh
```

## Basic Operations in MongoDB Shell

### 1. Selecting a Database

Switch to the desired database (or create it if it doesn't exist):

```sh
use myDatabase
```

### 2. Inserting Documents

Insert a single document into a collection:

```sh
db.myCollection.insertOne({ name: "John", age: 30, city: "New York" })
```

Insert multiple documents:

```sh
db.myCollection.insertMany([
  { name: "Jane", age: 25, city: "Chicago" },
  { name: "Mike", age: 35, city: "San Francisco" }
])
```

### 3. Querying Documents

Find a single document:

```sh
db.myCollection.findOne({ name: "John" })
```

Find multiple documents (all documents in the collection):

```sh
db.myCollection.find()
```

Find documents with a specific condition:

```sh
db.myCollection.find({ age: { $gt: 30 } })
```

Pretty print the results for readability:

```sh
db.myCollection.find().pretty()
```

### 4. Updating Documents

Update a single document:

```sh
db.myCollection.updateOne(
  { name: "John" },
  { $set: { age: 31 } }
)
```

Update multiple documents:

```sh
db.myCollection.updateMany(
  { city: "New York" },
  { $set: { city: "Brooklyn" } }
)
```

### 5. Deleting Documents

Delete a single document:

```sh
db.myCollection.deleteOne({ name: "John" })
```

Delete multiple documents:

```sh
db.myCollection.deleteMany({ age: { $lt: 30 } })
```

### 6. Creating an Index

Create an index on a field to improve query performance:

```sh
db.myCollection.createIndex({ name: 1 })
```

## Examples of Common Operations

- **Find all documents and sort them by age in descending order:**

  ```sh
  db.myCollection.find().sort({ age: -1 })
  ```

- **Find the first 5 documents in the collection:**

  ```sh
  db.myCollection.find().limit(5)
  ```

- **Count the number of documents in a collection:**

  ```sh
  db.myCollection.countDocuments()
  ```

- **Find documents where the name starts with 'J':**

  ```sh
  db.myCollection.find({ name: { $regex: "^J" } })
  ```

## Aggregation Framework

For more complex queries and data transformations, you can use the aggregation framework:

- **Group documents by city and count the number of users in each city:**

  ```sh
  db.myCollection.aggregate([
    { $group: { _id: "$city", count: { $sum: 1 } } }
  ])
  ```

## Schema Validation

MongoDB allows you to enforce a schema on your collections using JSON Schema:

- **Create a collection with schema validation:**

  ```sh
  db.createCollection("validatedCollection", {
    validator: {
      $jsonSchema: {
        bsonType: "object",
        required: ["name", "age"],
        properties: {
          name: {
            bsonType: "string",
            description: "must be a string and is required"
          },
          age: {
            bsonType: "int",
            minimum: 0,
            description: "must be an integer greater than or equal to 0 and is required"
          }
        }
      }
    }
  })
  ```

## Exiting the Mongo Shell

To exit the `mongo` shell, simply type:

```sh
exit
```

This README covers the basics of working with documents and collections in MongoDB using the `mongo` shell. For more advanced usage and features, refer to the [MongoDB documentation](https://docs.mongodb.com/).
---

# MongoDB and Python Tutorial

## Introduction

MongoDB is a NoSQL, document-oriented database designed for scalability, flexibility, and performance. It stores data in flexible, JSON-like documents. Python, with its simplicity and readability, is a powerful language to interact with MongoDB, making use of the `pymongo` library.

## Prerequisites

- Python 3.x
- MongoDB server installed and running
- `pymongo` library

## Installation

### Install MongoDB

Follow the official MongoDB installation guide for your operating system: [MongoDB Installation Guide](https://docs.mongodb.com/manual/installation/)

### Install `pymongo`

To interact with MongoDB in Python, you need to install the `pymongo` library. You can install it using `pip`:
```sh
pip install pymongo
```

## Connecting to MongoDB

Here's a simple example of how to connect to a MongoDB instance using Python:

```python
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')

# Select a database
db = client.myDatabase

# Select a collection
collection = db.myCollection
```

## Basic CRUD Operations

### Inserting Documents

Insert a single document:
```python
document = { "name": "John", "age": 30, "city": "New York" }
collection.insert_one(document)
```

Insert multiple documents:
```python
documents = [
    { "name": "Jane", "age": 25, "city": "Chicago" },
    { "name": "Mike", "age": 35, "city": "San Francisco" }
]
collection.insert_many(documents)
```

### Querying Documents

Find a single document:
```python
document = collection.find_one({ "name": "John" })
print(document)
```

Find multiple documents:
```python
for doc in collection.find():
    print(doc)
```

Find documents with a specific condition:
```python
for doc in collection.find({ "age": { "$gt": 30 } }):
    print(doc)
```

### Updating Documents

Update a single document:
```python
collection.update_one(
    { "name": "John" },
    { "$set": { "age": 31 } }
)
```

Update multiple documents:
```python
collection.update_many(
    { "city": "New York" },
    { "$set": { "city": "Brooklyn" } }
)
```

### Deleting Documents

Delete a single document:
```python
collection.delete_one({ "name": "John" })
```

Delete multiple documents:
```python
collection.delete_many({ "age": { "$lt": 30 } })
```

## Advanced Operations

### Indexing

Create an index on a field to improve query performance:
```python
collection.create_index([("name", pymongo.ASCENDING)])
```

### Aggregation

Using the aggregation framework to perform complex data transformations:
```python
pipeline = [
    { "$group": { "_id": "$city", "count": { "$sum": 1 } } }
]

result = collection.aggregate(pipeline)
for doc in result:
    print(doc)
```

### Schema Validation

Enforce schema validation using `pymongo`:
```python
db.create_collection("validatedCollection", validator={
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["name", "age"],
        "properties": {
            "name": {
                "bsonType": "string",
                "description": "must be a string and is required"
            },
            "age": {
                "bsonType": "int",
                "minimum": 0,
                "description": "must be an integer greater than or equal to 0 and is required"
            }
        }
    }
})
```

## Example Application

Here's an example of a simple application that uses MongoDB and Python:

```python
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')

# Select a database and collection
db = client.myDatabase
collection = db.myCollection

# Insert a document
collection.insert_one({ "name": "Alice", "age": 28, "city": "Los Angeles" })

# Query the collection
for doc in collection.find():
    print(doc)

# Update a document
collection.update_one({ "name": "Alice" }, { "$set": { "age": 29 } })

# Delete a document
collection.delete_one({ "name": "Alice" })
```

## Conclusion

This guide provides an introduction to using MongoDB with Python. It covers the basics of connecting to MongoDB, performing CRUD operations, and using advanced features like indexing and aggregation. For more detailed information, refer to the [MongoDB documentation](https://docs.mongodb.com/) and the [pymongo documentation](https://pymongo.readthedocs.io/).

---

This README should give you a solid foundation for working with MongoDB using Python. It includes installation instructions, basic CRUD operations, and examples of advanced usage.
