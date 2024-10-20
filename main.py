from pathlib import Path

from dotenv import load_dotenv

from ingestion.document_loader import process_directory
from rag.generator import query_rag, setup_rag_pipeline
from rag.retriever import setup_retriever
from utils.embedding import setup_embedding_and_vectorstore

PROJECT_ROOT = Path(__file__).parent
load_dotenv()


def main():
    # Directory to process
    data_directory = PROJECT_ROOT / "test_data"

    # Process all documents in the directory and its subdirectories
    print("Processing documents...")
    all_chunks = process_directory(data_directory)
    print(f"Total number of chunks processed: {len(all_chunks)}")

    # Setup embedding and vector store
    print("Setting up embedding and vector store...")
    vectorstore = setup_embedding_and_vectorstore(all_chunks)

    # Setup retriever and RAG pipeline
    print("Setting up retriever and RAG pipeline...")
    retriever = setup_retriever(vectorstore)
    qa_chain = setup_rag_pipeline(retriever)

    # Simple interface for testing
    print("\nRAG system is ready. You can now ask questions about the documents.")
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ")
        if query.lower() == "quit":
            break

        answer, sources = query_rag(qa_chain, query)
        print(f"\nAnswer: {answer}\n")
        print("Sources:")
        for i, doc in enumerate(sources):
            print(
                f"Source {i+1}: {doc.metadata['source']} (page {doc.metadata.get('page', 'N/A')})"
            )
        print("\n")


if __name__ == "__main__":
    main()
