from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
import time


def setup_mongodb(mongodb_uri, mongodb_db, mongodb_collection):
    # Set up MongoDB connection
    client = MongoClient(mongodb_uri, appname="chat_pdf")
    db = client[mongodb_db]
    collection = db[mongodb_collection]
    return client, collection

def check_index(collection, index_name="vector_index", creation=True):
    # waiting for the index to be created or deleted
    while True:
        existed = False
        for index in collection.list_search_indexes():
            if index["name"] == index_name:
                if creation and index["status"] == "READY":
                    existed = True
                elif not creation:
                    existed = True

        if existed and creation:
            return True
        if not existed and not creation:
            return True
        time.sleep(5)

def create_search_vector_index(mongodb_uri, mongodb_db, mongodb_collection):
    client, collection = setup_mongodb(mongodb_uri, mongodb_db, mongodb_collection)
            
    search_index_model = SearchIndexModel(definition={
                                    "fields": [
                                    {
                                        "type": "vector",
                                        "numDimensions": 768,
                                        "path": "embedding",
                                        "similarity":  "cosine"
                                    },
                                    {
                                        "type": "filter",
                                        "path": "file_name"
                                    }
                                    ]
                                },
                                name="vector_index",
                                type="vectorSearch",
                                )
    try:
        collection.drop_search_index("vector_index")
        check_index(collection, index_name="vector_index", creation=False)
    except:
        pass
    collection.create_search_index(model=search_index_model)            
    check_index(collection, index_name="vector_index", creation=True)
        

    client.close()