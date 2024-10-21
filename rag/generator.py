from operator import itemgetter

from langchain.chains import create_history_aware_retriever
from langchain_core.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def reciprocal_rank_fusion(results: list[list], k=60):
    """
    Reciprocal_rank_fusion that takes multiple lists of ranked documents
    and an optional parameter k used in the RRF formula
    """

    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not in the fused_scores dictionary, add it with an inital score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            # previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based of their ranked score in decending order to get the final reranked result
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return reranked_results


def setup_rag_pipeline(retriever, reciprocal_rank_fusion):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)

    contextualize_q_system_prompt = """Given a chat history and the latest user question
    which might reference context in the chat history,
    formulate a standalone question which can be understood
    without the chat history. Do NOT answer the question, 
    just reformulate it if needed and otherwise return it as is."""

    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    {context}

    Question: {question}
    Helpful Answer:"""

    rag_fusion_template = """You are a helpful assistant that generates multiple search queries based of a single input query. \n
    Generate multiple search queries related to: {question} \n
    output: (4 queries):
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    prompt_rag_fusion = ChatPromptTemplate.from_template(rag_fusion_template)

    generate_queries = (
        prompt_rag_fusion
        | ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        | StrOutputParser()
        | (lambda x: x.split("\n"))
    )

    retrieval_chain_rag_fusion = (
        generate_queries | retriever.map() | reciprocal_rank_fusion
    )

    prompt = ChatPromptTemplate.from_template(template)

    qa_chain = (
        {"context": retrieval_chain_rag_fusion, "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain


def query_rag(qa_chain, query: str):
    result = qa_chain.invoke({"question": query})
    return result
