from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (Docx2txtLoader, PyPDFLoader,
                                                  TextLoader)
from langchain_core.documents import Document


class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def load_and_split(self, file_path: Path) -> List[Document]:
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix.lower() == ".docx":
            loader = Docx2txtLoader(str(file_path))
        elif file_path.suffix.lower() in [".txt", ".md", ".py", ".js", ".html", ".css"]:
            loader = TextLoader(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        documents = loader.load()
        return self.text_splitter.split_documents(documents)


def process_directory(directory_path: Path) -> List[Document]:
    processor = DocumentProcessor()
    all_documents = []

    for file_path in directory_path.rglob("*"):
        if file_path.is_file():
            try:
                documents = processor.load_and_split(file_path)
                all_documents.extend(documents)
                print(f"Processed: {file_path}")
            except ValueError as e:
                print(f"Skipping {file_path}: {str(e)}")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

    return all_documents
