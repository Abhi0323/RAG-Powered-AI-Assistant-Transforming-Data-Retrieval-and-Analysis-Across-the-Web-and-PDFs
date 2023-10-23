import os
import streamlit as st
import pickle
import PyPDF2
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
import tempfile
import time
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain


OPENAI_API_KEY ='User Your Open AI API KEY'
url_file_path = "url_faiss_store_openai.pkl"

# Streamlit setup
st.set_page_config(
    page_title="Personal AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("<h1 style='text-align: center; color: black;'>🤖 Personal AI Assistant</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='text-align: center; color: black;'>Assistant Console</h3>", unsafe_allow_html=True)

# ---- URL Loading & Embedding ----
num_links = st.sidebar.slider("How many links do you want to input?", min_value=1, max_value=5, value=1)
urls = [st.sidebar.text_input(f"URL {i+1}", key=f"url{i}") for i in range(num_links)]
if urls:
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "."], chunk_size=1000)
    url_docs = text_splitter.split_documents(data)
    if url_docs:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        url_vectorindex_openai = FAISS.from_documents(url_docs, embeddings)
        with open(url_file_path, "wb") as f:
            pickle.dump(url_vectorindex_openai, f)

# ---- PDF Loading & Embedding ----
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=['pdf'])
if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "."], chunk_size= 500)
    pdf_docs = text_splitter.split_text(pdf_text)
    if pdf_docs:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        pdf_vectors = FAISS.from_texts(pdf_docs, embeddings)
        

# ---- Query Interface ----
llm = OpenAI(temperature=0.9, max_tokens=500, openai_api_key=OPENAI_API_KEY)
data_source = st.selectbox("What do you want to inquire about?", ["URL", "PDF"])

if data_source == "URL":
    query_url = st.text_input('Ask your question about URLs:')
    if query_url:
        if os.path.exists(url_file_path):  # Ensure URL database exists
            with open(url_file_path, "rb") as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query_url}, return_only_outputs=True)
                st.header("Answer based on URLs:")
                st.subheader(result['answer'])

elif data_source == "PDF":
    query_pdf = st.text_input('Ask your question about PDFs:')
    if query_pdf:
        docs = pdf_vectors.similarity_search(query_pdf)

        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query_pdf)
           
        st.write(response)

    if st.button("Summarize PDF"):
        def summarize_pdfs_from_folder(pdfs_folder):
            summaries = []
            for pdf_file in pdfs_folder:
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_path = temp_file.name
                    temp_file.write(pdf_file.getvalue())
                loader = PyPDFLoader(temp_path)
                docs = loader.load_and_split()
                chain = load_summarize_chain(llm, chain_type="map_reduce")
                summary = chain.run(docs)
                summaries.append(summary)
                os.remove(temp_path)
            return summaries

        summaries = summarize_pdfs_from_folder([uploaded_file])
        for summary in summaries:
            st.write(summary)





