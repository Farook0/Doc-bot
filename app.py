from flask import render_template, jsonify, request, Flask
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from src.prompt import *
from dotenv import load_dotenv
import os
import flask

app = Flask(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Download embeddings and connect to Pinecone
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone connection
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "mbot"

# Check if the index exists; if not, create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Adjust based on the embedding model used
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Load the index
docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)

# Define the prompt template
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Load the Llama model
llm = CTransformers(
    model="Model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={'max_new_tokens': 512, 'temperature': 0.8}
)

# Create the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Home route rendering chat interface
@app.route("/")
def index():
    return render_template('chat.html')

# Chat route to process user input
@app.route("/get", methods=["POST"])
def chat():
    try:
        # Get user input from the request
        msg = request.form.get("msg")
        
        # Process the query using the RetrievalQA chain
        result = qa({"query": msg})
        
        # Extract the response and return it
        response = result["result"]
        print("Response: ", response)
        return jsonify(response=response)
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify(response="Sorry, I encountered an error processing your request.")

if __name__ == "__main__":
    app.run(debug=True)
