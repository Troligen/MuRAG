from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.chain_extract import \
    LLMChainExtractor
from langchain_openai import ChatOpenAI


def setup_retriever(vectorstore):
    base_retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    return compression_retriever
