# 📖 PaperPulse AI

A conversational RAG-powered research assistant for querying multiple PDFs and custom text content using Mistral-7B via Together API.

## 🚀 Features
- 📄 Upload one or multiple PDFs
- 💬 Conversational Q&A with document context
- 🗃️ Chroma VectorDB for efficient retrieval
- 🔍 Mistral-7B LLM via Together API
- 📝 Multi-turn conversational memory
- 🎨 Clean light-mode UI with custom colors
- 📖 Source document display for each answer

## 🛠️ Technical Highlights

- **RAG Pipeline**: Built on LangChain's RetrievalQA chain pattern
- **Vector Store**: Chroma for fast document chunk retrieval
- **Embedding Model**: Sentence Transformers ("BAAI/bge-base-en-v1.5" or  fallback to "sentence-transformers/all-mpnet-base-v2")
- **Conversational Memory**: Maintains history of interactions per session  
- **PDF Text Extraction Logic**:
  - 📑 Primary: `unstructured.partition_pdf()`
  - 🔄 Fallback: `PyMuPDF` (`fitz`) for handling multi-column, scanned, or poorly structured research PDFs
  - Automatically determines best extraction strategy per document

## 🖥️ Demo
> ![alt text](<Screenshot (195).png>)

## 🔧 Tech Stack
- LangChain
- Chroma
- Mistral-7B via Together API
- HuggingFace Transformers
- Streamlit



