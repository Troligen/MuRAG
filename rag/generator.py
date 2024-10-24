from typing import List

from langchain.load import dumps, loads
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]


class Prompts(BaseModel):
    """
    Dictiornary Keys for the prompts:

    grader_template
    rewrite_template
    generator_template
    rag_fusion_template
    """

    grader_template: str
    rewrite_template: str
    generator_template: str
    rag_fusion_template: str


class Pipeline(StateGraph):
    def __init__(
        self,
        retriever,
        prompts: Prompts,
        state=GraphState,
        web_search_tool=None,
    ):
        super().__init__(state)

        if web_search_tool is None:
            web_search_tool = TavilySearchResults(
                max_results=5,
                search_depth="advanced",
            )

        self.retriever = retriever

        self.llm_4 = ChatOpenAI(model="gpt-4o-mini", temperature=1)
        self.llm_3 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.llm_f = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

        self.prompts = Prompts(**prompts)  # pyright: ignore
        self.web_search_tool = web_search_tool
        self.structured_llm_grader = self.llm_f.with_structured_output(GradeDocuments)

        self.grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompts.grader_template),
                (
                    "human",
                    "Retrieved document: \n\n {document} \n\n User question: {question}",
                ),
            ]
        )

        self.retrieval_grader = self.grade_prompt | self.structured_llm_grader

        self.re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompts.rewrite_template),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )

        self.prompt_rag_fusion = ChatPromptTemplate.from_template(
            self.prompts.rag_fusion_template
        )

        self.question_rewriter = self.re_write_prompt | self.llm_f | StrOutputParser()

        self.generate_queries = (
            self.prompt_rag_fusion
            | self.llm_3
            | StrOutputParser()
            | (lambda x: x.split("\n"))
        )

        self.prompt_generator = ChatPromptTemplate.from_template(
            self.prompts.generator_template
        )

        self.rag_chain = self.prompt_generator | self.llm_4 | StrOutputParser()

    def _reciprocal_rank_fusion(self, results: list[list], k=60):
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
            for doc, score in sorted(
                fused_scores.items(), key=lambda x: x[1], reverse=True
            )
        ]

        return reranked_results

    def _retrieve(self, state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        queries = self.generate_queries.invoke({"question": question})
        documents = []
        for query in queries:
            results = self.retriever.invoke(query)
            documents.append(results)
        reranked_documents = self._reciprocal_rank_fusion(documents)
        return {"documents": reranked_documents, "question": question}

    def _generate(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # RAG generation
        generation = self.rag_chain.invoke({"context": documents, "question": question})
        return {
            "documents": generation,
            "question": question,
            "generation": generation,
        }

    def _grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = self.retrieval_grader.invoke({"question": question, "document": d})
            grade = score.binary_score  # pyright: ignore
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
                continue
        return {
            "documents": filtered_docs,
            "question": question,
            "web_search": web_search,
        }

    def _transform_query(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = self.question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}

    def _web_search(self, state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        print("---WEB SEARCH---")
        question = state["question"]
        documents = state["documents"]

        # Web search
        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        documents.append(web_results)

        return {"documents": documents, "question": question}

    def _decide_to_generate(self, state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        state["question"]
        web_search = state["web_search"]
        state["documents"]

        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"

    def compile_graph(self):

        # Define the nodes
        self.add_node("retrieve", self._retrieve)  # retrieve
        self.add_node("grade_documents", self._grade_documents)  # grade documents
        self.add_node("generate", self._generate)  # generatae
        self.add_node("transform_query", self._transform_query)  # transform_query
        self.add_node("web_search_node", self._web_search)  # web search

        # Build graph
        self.add_edge(START, "retrieve")
        self.add_edge("retrieve", "grade_documents")
        self.add_conditional_edges(
            "grade_documents",
            self._decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        self.add_edge("transform_query", "web_search_node")
        self.add_edge("web_search_node", "generate")
        self.add_edge("generate", END)

        # Compile
        app = self.compile()

        return app

