from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
# from langchain.vectorstores import Pinecone
# import pinecone
from pinecone import Pinecone
from dotenv import load_dotenv
import os
os.chdir('c:/Users/gt260/End-to-end-Medical-Chatbot-using-Llama2')

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
index_name = "medical-bot"

#Creating Embeddings for Each of The Text Chunks & storing
texts = []
# docsearch = PineconeVectorStore.from_texts(texts, embedding = embeddings, index_name=index_name)
docsearch = Pinecone.from_texts(texts, embedding = embeddings, index_name=index_name)
docsearch.add_texts([t.page_content for t in text_chunks])

