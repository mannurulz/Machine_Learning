# Imports
import pymongo
import pandas as pd
import numpy as np
from sklearn import datasets
import pickle
import time
import pymongo

# Sample data iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

## from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Sample Model for testing
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
#print(gnb.predict(X_test))

# Define function to save model in MongoDB
def save_model_to_db(model, client, db, dbconnection, model_name):
    import pickle
    import time
    import pymongo
    #pickling the model
    pickled_model = pickle.dumps(model)
    
    #saving model to mongoDB
    # creating connection
    myclient = pymongo.MongoClient(client)
    
    #creating database in mongodb
    mydb = myclient[db]
    
    #creating collection
    mycon = mydb[dbconnection]
    info = mycon.insert_one({model_name: pickled_model, 'name': model_name, 'created_time':time.time()})
    print(info.inserted_id, ' saved with this id successfully!')
    
    details = {
        'inserted_id':info.inserted_id,
        'model_name':model_name,
        'created_time':time.time()
    }
    
    return details

def load_saved_model_from_db(model_name, client, db, dbconnection):
    json_data = {}
    
    #saving model to mongoDB
    # creating connection
    myclient = pymongo.MongoClient(client)
    
    #creating database in mongodb
    mydb = myclient[db]
    
    #creating collection
    mycon = mydb[dbconnection]
    data = mycon.find({'name': model_name})
    
    
    for i in data:
        json_data = i
    #fetching model from db
    pickled_model = json_data[model_name]
    
    return pickle.loads(pickled_model)


#saving model to mongo
details = save_model_to_db(model = gnb, client ='mongodb+srv://myUser:*******@cluster0.nnca0.mongodb.net/<dbname>?retryWrites=true&w=majority', db = 'myDB', dbconnection = 'myCol', model_name = 'mygnb')

#fetching model from mongo
gnb = load_saved_model_from_db(model_name = details['model_name'], client = 'mongodb+srv://myUser:myPass123@cluster0.nnca0.mongodb.net/<dbname>?retryWrites=true&w=majority', db = 'myDB', dbconnection = 'myCol')

print(gnb.predict(X_test))