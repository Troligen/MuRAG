from pathlib import Path

from dotenv import load_dotenv

from ingestion.document_loader import process_document
from rag.generator import query_rag, setup_rag_pipeline
from rag.retriever import setup_retriever
from utils.embedding import setup_embedding_and_vectorstore

PROJECT_ROOT = Path(__file__).parent
load_dotenv()


def main():
    # Process documents
    test_file = (
        PROJECT_ROOT / "test_data" / "science_pdf" / "1-s2.0-S2212096323000189-main.pdf"
    )
    chunks = process_document(test_file)
    print(f"Number of chunks: {len(chunks)}")

    # Setup embedding and vector store
    vectorstore = setup_embedding_and_vectorstore(chunks)

    # Setup retriever and RAG pipeline
    retriever = setup_retriever(vectorstore)
    qa_chain = setup_rag_pipeline(retriever)

    # Simple interface for testing
    while True:
        query = input("Enter your question (or 'quit' to exit): ")
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
