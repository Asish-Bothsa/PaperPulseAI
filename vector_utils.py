from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
import shutil
import streamlit as st
@st.cache_resource
def get_embedding_model():
    try:
        model_name = "BAAI/bge-base-en-v1.5"
        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception as e:
        print(f"⚠️ Failed to load {model_name}. Error: {e}")
        print("➡️ Falling back to all-mpnet-base-v2.")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


def create_vector_store(documents, embedding_model, persist_directory="chroma_db"):
    if not documents:
        print("No documents provided — cannot create vectorstore.")
        return None

    vectorstore = Chroma.from_documents(
        documents,
        embedding_model,
        persist_directory=persist_directory
    )
    vectorstore.persist()  # save immediately
    print(f"Vector store created with {len(documents)} documents at {persist_directory}.")
    return vectorstore

def save_vector_store(vectorstore):
    vectorstore.persist()
    print("Vector store changes persisted to disk.")

def add_documents_to_vector_store(vectorstore, documents):
    if not documents:
        print("No documents to add.")
        return vectorstore

    vectorstore.add_documents(documents)
    vectorstore.persist()
    print(f"Added {len(documents)} documents to the vector store and persisted.")
    return vectorstore

def load_vector_store(embedding_model=None, persist_directory="chroma_db"):
    if embedding_model is None:
        embedding_model = get_embedding_model()

    if os.path.exists(persist_directory):
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
        print(f"Vector store loaded from {persist_directory}.")
        return vectorstore
    else:
        print(f"No vector store found at {persist_directory}. Initializing empty store.")
        return None

# def delete_vector_store(persist_directory="chroma_db"):
#     if os.path.exists(persist_directory):
#         shutil.rmtree(persist_directory)
#         print(f"Vector store at {persist_directory} deleted.")
#     else:
#         print(f"No vector store found at {persist_directory}.")
        
        
def delete_vector_store_collection(vectorstore):
    if vectorstore:
        vectorstore.delete_collection()
        print("Vector store collection deleted.")
    else:
        print("No vector store to delete.")
