import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

def get_azure_embeddings():
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    embedding_deployment = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
    api_version = os.getenv("API_VERSION")
    return AzureOpenAIEmbeddings(
        azure_deployment=embedding_deployment,
        openai_api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        chunk_size=1
    )

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    # Path to your PDF file
    pdf_path = os.path.join("Documents", "CelcomDigi_Postpaid_Plans_Detailed.pdf")
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        exit(1)

    print(f"Extracting text from {pdf_path}...")
    extracted_text = extract_text_from_pdf(pdf_path)
    print(f"Text extracted. Length: {len(extracted_text)} characters.")

    print("Chunking text...")
    chunks = chunk_text(extracted_text)
    print(f"Total chunks: {len(chunks)}")

    print("Creating embeddings and FAISS index...")
    embedding_fn = get_azure_embeddings()
    documents = [Document(page_content=chunk) for chunk in chunks]
    faiss_db = FAISS.from_documents(
        documents=documents,
        embedding=embedding_fn
    )
    faiss_db.save_local("faissIndex")
