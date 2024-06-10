from pymongo import MongoClient

# Connect to the MongoDB server
client = MongoClient("mongodb://localhost:27017/")

# Access the "exampledb" database
db = client["exampledb"]

# Access the "users" collection
users = db["users"]

# Insert documents
users.insert_one({"name": "Alice", "age": 25, "city": "New York"})
users.insert_many([
    {"name": "Bob", "age": 30, "city": "Chicago"},
    {"name": "Charlie", "age": 35, "city": "San Francisco"}
])

# Read documents
print("Users in New York:")
for user in users.find({"city": "New York"}):
    print(user)

# Update documents
users.update_one({"name": "Alice"}, {"$set": {"age": 26}})
print("Updated Alice's age:")
print(users.find_one({"name": "Alice"}))

# Delete documents
users.delete_one({"name": "Charlie"})
print("Users after deleting Charlie:")
for user in users.find():
    print(user)
