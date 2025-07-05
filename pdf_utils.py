from unstructured.partition.pdf import partition_pdf
import fitz
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_text_unstructured(file, ignored_categories=None, min_chunk_length=20):
    if ignored_categories is None:
        ignored_categories = ["PageHeader", "PageFooter", "FigureCaption", "Equation"]

    try:
        elements = partition_pdf(file=file)
        content = []
        for el in elements:
            if (
                el.text 
                and el.category not in ignored_categories 
                and len(el.text.strip()) >= min_chunk_length
            ):
                content.append(el.text.strip())
        return content
    except Exception as e:
        print(f"Unstructured extraction failed: {e}")
        return []

def extract_text_pymupdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    content = []
    for page in pdf_document:
        text = page.get_text()
        if text.strip():
            content.append(text.strip())
    pdf_document.close()
    return content

# def extract_text_with_fallback(file):   
#     content = extract_text_unstructured(file)
#     if not content or len(" ".join(content)) < 500:
#         print("Unstructured insufficient — falling back to PyMuPDF.")
#         file.seek(0)
#         content = extract_text_pymupdf(file)
#     return content
def extract_text_with_fallback(file):
    # Attempt extraction with unstructured
    content = extract_text_unstructured(file)

    # Check extraction quality
    text_combined = " ".join(content)
    if not content or len(text_combined) < 500 or len(content) < 3:
        print("⚠️ Unstructured extraction insufficient.")
        file.seek(0)  # Reset file pointer

        # Attempt PyMuPDF fallback
        content_pymupdf = extract_text_pymupdf(file)
        text_pymupdf = " ".join(content_pymupdf)

        # Check if PyMuPDF extraction is better
        if len(text_pymupdf) > len(text_combined):
            print(f"✅ Fallback to PyMuPDF succeeded with {len(content_pymupdf)} chunks.")
            content = content_pymupdf
        else:
            print("⚠️ PyMuPDF did not improve extraction. Keeping original result.")

    return content

def split_text_into_documents(text_chunks, source_name, chunk_size=800, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    split_documents = []
    for chunk in text_chunks:
        if len(chunk) > chunk_size:
            splits = splitter.split_text(chunk)
            split_documents.extend([Document(page_content=s, metadata={"source": source_name}) for s in splits])
        else:
            split_documents.append(Document(page_content=chunk, metadata={"source": source_name}))
    return split_documents
