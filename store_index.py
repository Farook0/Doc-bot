from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as LangchainPinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()  
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

extracted_data = load_pdf("Data/")
text_chunks = text_split(extracted_data)
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
# Create a Pinecone vector store from the text embeddings
docsearch = LangchainPinecone.from_texts(
    [t.page_content for t in text_chunks],
    embeddings,
    index_name=index_name
)