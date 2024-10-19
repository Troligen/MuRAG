from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (Docx2txtLoader, PyPDFLoader,
                                                  TextLoader)
from langchain_core.documents import Document


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def load_and_split(self, file_path: Path) -> List[Document]:
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix.lower() == ".docx":
            loader = Docx2txtLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path))

        documents = loader.load()
        return self.text_splitter.split_documents(documents)


def process_document(file_path: Path) -> List[Document]:
    processor = DocumentProcessor()
    return processor.load_and_split(file_path)
