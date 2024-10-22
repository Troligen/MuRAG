from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank


def setup_retriever(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 20}
    )
    compressor = CohereRerank(model="rerank-multilingual-v3.0")

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    return compression_retriever
