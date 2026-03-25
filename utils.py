from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import os

def create_qa_chain():

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_api_key_here"

    # Load Data
    with open("data.txt", "r") as file:
        data = file.read()

    # Split text
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    texts = splitter.split_text(data)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vector DB
    db = FAISS.from_texts(texts, embeddings)

    # LLM
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature":0.5, "max_length":512}
    )

    # RAG Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever()
    )

    return qa
