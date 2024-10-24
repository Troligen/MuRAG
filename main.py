import json
from pathlib import Path
from pprint import pprint

from dotenv import load_dotenv

from ingestion.document_loader import process_directory
from rag.generator import Pipeline
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
    with open("prompts/prompts_templates.json", "r") as f:
        templates = json.load(f)
    pipeline = Pipeline(retriever, templates)
    app = pipeline.compile_graph()

    config = {"configurable": {"thread_id": "1"}}

    # Simple interface for testing
    print("\nRAG system is ready. You can now ask questions about the documents.")
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ")

        if query.lower() == "quit":
            break

        for output in app.stream({"question": query}, config):
            for key, value in output.items():
                # Node
                pprint(f"Node '{key}':")
                # Optional: print full state at each node
                # pprint(value["key"], indent=2, width=80, depth=None)
                if key == "generate":
                    print(f"\n\n\nLLM Answer: {value["generation"]}\n\n\n")
            pprint("\n---\n")


if __name__ == "__main__":
    main()
