import getpass
import os
import configparser
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# Get open AI key
config_path = r"C:\Users\felixstuyck\OneDrive - Finvision\Documenten\Python\AI assistant config.ini"
config = configparser.ConfigParser()
config.read(config_path)
OPEN_AI_KEY = config['API_KEY_Assistent']['API_KEY']
os.environ["OPENAI_API_KEY"] = getpass.getpass(OPEN_AI_KEY)

# Load and process the document
loader = TextLoader(R"C:\Users\felixstuyck\OneDrive - Finvision\Documenten\Stow stored procedure.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Generate embeddings and create FAISS index
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)
print(f"Number of documents in the index: {db.index.ntotal}")

# Query the vectorstore
query = "What did the president say about Ketanji Brown Jackson"
docs = db.similarity_search(query)
print(f"Top result content: {docs[0].page_content}")

# Use as retriever
retriever = db.as_retriever()
docs = retriever.invoke(query)
print(f"Top result content (retriever): {docs[0].page_content}")

# Similarity search with score
docs_and_scores = db.similarity_search_with_score(query)
for doc, score in docs_and_scores:
    print(f"Content: {doc.page_content}, Score: {score}")

# Save and load FAISS index
db.save_local("faiss_index")
new_db = FAISS.load_local("faiss_index", embeddings)
docs = new_db.similarity_search(query)
print(f"Top result content (loaded index): {docs[0].page_content}")
