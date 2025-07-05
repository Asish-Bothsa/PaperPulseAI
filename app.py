import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from pdf_utils import extract_text_with_fallback, split_text_into_documents
from vector_utils import (
    create_vector_store, save_vector_store, load_vector_store,
    get_embedding_model, add_documents_to_vector_store,delete_vector_store_collection
)
from dotenv import load_dotenv
from langchain_community.llms import Together

import os
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACE_API_KEY:
    raise ValueError("Please set the HUGGINGFACE_API_KEY environment variable.")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError("Please set the TOGETHER_API_KEY environment variable.")

st.set_page_config(page_title="PaperPulse AI ", page_icon="📖")

st.markdown("""
<style>
    .title {
        font-size: 48px;
        color: #333333;
        text-align: center;
        font-weight: 700;
    }
    .subtitle {
        font-size: 22px;
        text-align: center;
        margin-bottom: 20px;
    }
    body {
    background-color: #F8F8F8;
    color: #222222;
}
.sidebar .sidebar-content {
    background-color: #E0F7FA;
}
.stApp {
    background-color: #F8F8F8;
}
</style>
<div class="title">📖 PaperPulse AI</div>
<div class="subtitle">Your Personal Research Assistant</div>
""", unsafe_allow_html=True)
# st.title("PaperPulse AI📖 Your Personal Research Assistant")
with st.expander("How to use this app", expanded=False):
    st.markdown("""
    1. 📄 Upload one or more PDF documents or paste text.
    2. 📥 The system splits the content into chunks, converts them into embeddings, and stores them in a vector database.
    3. 🤖 Ask natural language questions related to the uploaded content.
    4. 📚 The AI retrieves the most relevant document chunks and answers your query using Mistral-7B.
    5. 📝 View the source documents and chat history in the sidebar.
    """)


uploaded_files = st.file_uploader("📄 Upload one or more PDFs", type="pdf", accept_multiple_files=True)
article_text = st.text_area("📄 Or paste article content:", height=200)
# Load LLM via Together
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
@st.cache_resource
def get_llm():
    return  Together(
    model=model_name,
    temperature=0.5, 
    max_tokens=512
    )

embedding_model = get_embedding_model()
llm=get_llm()
vectorstore = load_vector_store(embedding_model, persist_directory="chroma_db")

if "docs_embedded" not in st.session_state:
    st.session_state["docs_embedded"] = False

if (uploaded_files or article_text) and not st.session_state["docs_embedded"]:
    total_chunks = 0
    all_documents = []
    progress=st.progress(0, "Processing {file.name}..")

    if uploaded_files:
        for i,file in enumerate(uploaded_files):
            with st.spinner(f"🔍 Processing {file.name}..."):
                text_chunks = extract_text_with_fallback(file)
                documents = split_text_into_documents(text_chunks, file.name)
                all_documents.extend(documents)
                total_chunks += len(documents)
            progress.progress((i + 1) / len(uploaded_files))

    if article_text:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_text(article_text)
        documents = [Document(page_content=chunk, metadata={"source": "TextInput"}) for chunk in docs]
        all_documents.extend(documents)
        total_chunks += len(documents)

    if all_documents:
        if not vectorstore:
            vectorstore = create_vector_store(all_documents, embedding_model, persist_directory="chroma_db")
        else:
            vectorstore = add_documents_to_vector_store(vectorstore, all_documents)

        save_vector_store(vectorstore)
        st.success(f"✅ Embedded {total_chunks} document chunks.")
    st.session_state["docs_embedded"] = True
    progress.empty()
    
if "history" not in st.session_state:
    st.session_state["history"] = []

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer"
    )

with st.sidebar:
    st.header("💬 Chat History")
    if st.session_state["history"]:
        for i, chat in enumerate(st.session_state["history"]):
            with st.expander(f"📝 Q{i+1}: {chat['question'][:40]}..."):
                st.markdown(f"**Q:** {chat['question']}")
                st.markdown(f"**A:** {chat['answer']}")
            if st.button(f"🗑️ Delete Q{i+1}", key=f"delete_{i}"):
                st.session_state["history"].pop(i)
                st.rerun()
            st.markdown("---")
    else:
        st.write("No conversation yet.")

    if st.button("🗑️ Clear All History"):
        st.session_state.history = []
        # Also clear the memory object
        if "memory" in st.session_state:
            st.session_state.memory.clear()
        st.rerun()

    if st.sidebar.button("🧹 Reset Vector Store"):
        if vectorstore:
            delete_vector_store_collection(vectorstore)
        
        # Clear all related session state variables
        st.session_state.docs_embedded = False
        st.session_state.qa_chain = None
        st.session_state.history = []
        if "memory" in st.session_state:
            st.session_state.memory.clear()
            
        st.success("Vector Store and conversation have been reset.")
        st.rerun()
    k_value = st.sidebar.slider(
    "Number of document chunks to retrieve:", 
    min_value=1, 
    max_value=10, 
    value=5, 
    key="k_slider")
    
# Prompt
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert research assistant. Your task is to answer questions based *only* on the provided context from the documents

Follow these rules strictly:
1.  Analyze the 'Context' section below, which contains excerpts from the document.
2.  Formulate your answer based exclusively on this context.
3.  If the context does not contain the information needed to answer the question, you MUST respond with: "I'm sorry, but the answer to that question is not found in the provided document excerpts."
4.  Do not use any external knowledge or make assumptions beyond the provided text.

Context:
{context}

Question: {question}

Answer:
"""
)

# Conversational QA Chain
if vectorstore:
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k":k_value}),
        memory=st.session_state["memory"],
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt_template},  
    )
    for msg in st.session_state.get("history", []):
        with st.chat_message("user"):
            st.markdown(f"**Q:** {msg['question']}")
        with st.chat_message("assistant"):
            st.markdown(f"**A:** {msg['answer']}")
       

    user_input= st.chat_input("Type your question here...")
    if user_input:
        with st.spinner("🤖 Generating answer..."):
            result = qa_chain.invoke({"question": user_input})

            answer = result["answer"]
            source_documents = result.get("source_documents", [])

            st.session_state["history"].append({
                "question": user_input,
                "answer": answer
            })
        with st.chat_message("user"):
            st.markdown(f"**Q:** {user_input}")
        with st.chat_message("assistant"):
            st.markdown(f"**A:** {answer}")


        if source_documents:
            with st.expander("📄 Source Documents"):
                for doc in source_documents:
                    st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                    st.code(doc.page_content, language="markdown")
                    st.markdown("---")
else:
    st.info("📄 Upload PDFs or paste content to initialize your RAG system.")

st.markdown("---")


