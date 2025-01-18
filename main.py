# from fastapi import FastAPI, Request
# from sentence_transformers import SentenceTransformer
# from langchain.embeddings import SentenceTransformerEmbeddings
# # # from langchain.vectorstores import Chroma
# import requests
# import chromadb
#
# app = FastAPI()
#
# model = SentenceTransformer('all-MiniLM-L6-v2')
# embedding_model = SentenceTransformerEmbeddings(model=model)
# client = chromadb.Client()
# collection = client.create_collection("my-collection")
#
#
# @app.post("/query_feeder")
# async def query_feeder(request: Request):
#     data = await request.json()
#     user_id = data['user_id']
#     chat_id = data['chat_id']
#     group_id = data['group_id']
#     query = data['query']
#
#     # Embed the query
#     embedding = get_embedding(query)
#
#     # Search in the vector database
#     response = search_vector_db(embedding)
#
#     # Send the response to another service
#     results = forward_response(response)
#
#     payload = {
#         "status": "success",
#         "message": "Query Processed Successfully",
#         "results": results
#     }
#     return results
#
#
# def get_embedding(query):
#     return embedding_model.encode(query).tolist()
#
#
# def search_vector_db(embedding):
#     result = collection.query(query_embeddings=[embedding], n_results=3)
#     return result
#
#
# def forward_response(response):
#     url = "https://example.com/receive_response"
#     result = requests.post(url, json=response)
#     return result
#


from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
# import chromadb

app = Flask(__name__)

# Load the embedding model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# client = chromadb.Client()
# collection = client.create_collection("my-collection")


@app.route('/query_feeder', methods=['POST'])
def query_feeder():
    data = request.json
    user_id = data['user_id']
    chat_id = data['chat_id']
    group_id = data['group_id']
    query = data['query']

    embedding = get_embedding(query)

    print(embedding)


    response_data = search_vector_db(embedding)

    print(response_data)

    result = forward_response(response_data)

    final_response = {
                "status": "success",
                "message": "Query Processed Successfully",
                "results": result
    }
    return jsonify(final_response)


def get_embedding(query):
    return model.encode(query).tolist()


def search_vector_db(embedding):
    # results = collection.query(query_embeddings=[embedding], n_results=1)
    results = [{"pdf_id": "123", "group_id": "456", "page_number": 20},{"pdf_id": "3", "group_id": "456", "page_number": 30}]
    response_data = {
        "status": "success",
        "message": "Query Processed Successfully",
        "results": results
    }
    return response_data


def forward_response(response):
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        # Nouman's format
    }
    #Naouamn's URL
    url = "xyz"
    # response = requests.post(url, json=payload, headers=headers)

    # if response.status_code == 200:
    #     print("Success:", response.json())
    # else:
    #     print("Error:", response.status_code, response.text)
    return [{"pdf_id": "123", "group_id": "456", "page_number": 20, "response": "processed text", "error_code": " "}]


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
