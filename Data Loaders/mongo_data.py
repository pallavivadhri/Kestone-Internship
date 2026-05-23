from pymongo import MongoClient
import json

#connection_string
mongo_cs ="mongodb+srv://vadhri:scBf3RaqaDSsTwxi@cluster0.dotrgan.mongodb.net/"
# Connect to MongoDB
client = MongoClient(mongo_cs)

# Select database and collection
db = client["Books"]
collection = db["Ratings"]

# Load data from example2.json
with open(r"C:\Users\palla\OneDrive\Desktop\Intern\Data Loaders\example2.json", "r") as file:
    books_data = json.load(file)

# Insert data into MongoDB
collection.insert_many(books_data)

print("Data inserted successfully!")
