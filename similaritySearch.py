import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage
from pydantic import SecretStr

load_dotenv()

def get_azure_embeddings():
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    embedding_deployment = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
    api_version = os.getenv("API_VERSION")
    api_key_secret = SecretStr(azure_api_key) if azure_api_key else None
    return AzureOpenAIEmbeddings(
        azure_deployment=embedding_deployment,
        api_key=api_key_secret,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        chunk_size=1
    )

def main():
    faiss_path = "faissIndex"
    if not os.path.exists(faiss_path):
        print(f"FAISS index not found at {faiss_path}. Run pdfToFaiss.py first.")
        return
    embedding_fn = get_azure_embeddings()
    faiss_db = FAISS.load_local(
        faiss_path,
        embeddings=embedding_fn,
        allow_dangerous_deserialization=True
    )
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.strip().lower() == "exit":
            break
        results = faiss_db.similarity_search(query, k=2)
        context = "\n\n".join([doc.page_content for doc in results])
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_deployment = os.getenv("DEPLOYMENT_NAME")
        api_version = os.getenv("API_VERSION")
        api_key_secret = SecretStr(azure_api_key) if azure_api_key else None
        llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key_secret,
            azure_deployment=azure_deployment,
            api_version=api_version,
            temperature=0.1,
            streaming=False
        )
        system = SystemMessage(content="You are AI Assistant. Provide clear, accurate, and concise answers strictly based on the context provided. Ensure your responses are balanced in length—neither too brief nor overly detailed—delivering essential information effectively and efficiently. Avoid including any information not supported by the given context.")
        user = HumanMessage(content=f"Context:\n{context}\n\nUser Question: {query}\n\nAnswer using only the given context.")
        response = llm.invoke([system, user])
        answer = response.content if isinstance(response.content, str) else str(response.content)
        print(f"\nAzure OpenAI Answer:\n{answer.strip()}\n{'='*60}")

if __name__ == "__main__":
    main() 