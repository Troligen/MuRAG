# RAG
## Main Components
### Indexing

[Document Loading:]

- Text Representation:
  - Retrieve documents that relates
    to the input question

>       questioen -> Retriever -> [Document]
>                       |
>                 Load Documents
>                       |
>                   Documents

- Numerical Representation:
  - Numerical Representation of Document
    because it's easier to compare vectors
    with numbers then free form text
  - Text documents being compressed to
    sequens of numbers for easier searches.

> (x,y,z..) -> Cosine similarity, etc -> [(x,y,z..)]
> |
> Load Documents
> |
> (x,y,z..)
> [(x,y,z..)]

[Statical and Machine Learned Representations:]

statical Example:

    Spars vectors = Looking at the frequency of words
    and create a Spars Vector in such the vecotor location
    are a large vocabulay of possible words where each value
    represent the number of the occurences of that
    particualar word and it's spars because
    there's many 0s

> There's very good search methods over these types
> of numerical representations.

Machine Learned Example:

    Embedding methods = You take and build a compressed fixed
    length reprecentation of a document. Have been developed
    with correspondingly very strong search methods over
    embeddings.

- Loading Splitting, and Embedding

  - Splitting:

    - We take a document and split it (chunking)
      because of the limited context window of
      an embedding model
      ([mayne between 512-8000 tokens])

  - Embedding:

    - Each document is compressed into a vector,
      said vector then captures a symantic meaning
      of the document and the vector get's indexed.
      Questions can be embdedded in the exact same way
      which then allows for a numerical comparison
      in some form using a veriaty of different methods
      to fish out the relevent documents relevent to said questions

[Code Example:]

```python
"""
Example: Returning numbers of a tokens in a string

This is interesting because LLMs in general including
embedding operates on tokens so it's nice to understand
how large the documents are that u try to feed in.
Mainly because the embedding has a limited context window.
"""
import tiktoken

question = "What kind of pets do I like?"
documents = "My favorite pet is a cat"

def num_tokens_from_string(string: str, encoding_name: srt) -> inte:
    """Returns the number of tokens in a text string"""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

num_tokens_from_string(question, "cl100k_base")
#output: 8
```

```python
"""
Example: embed a question and
a document to a vector embedding
using LangChain and openAI

Interesting note here is that the length
of the vector is 1536 which is a static length
so both the document and question
are both computed to a 1536 dimensional vector.
So the 1536 vector encodes the symantics
of the text that u pass.
"""

from langchain_openai import OpenAIEmbeddings

embd = OpenAIEmbeddings()
query_result = embd.embed_query(question)
document_result = embd.embed_query(document)
len(query_result)
#output: 1536
```

```python
"""
Example: cosin simelarity to campare 2 vectors
"""
import numpy as np

def cosine_similarity(vec1: list[int], vec2: list[int]) -> int:
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = no.linalf.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

similarity = cosine_similarity(query_result, document_result)
print(f"Cosine Similarity: {similarity}")
#Output: Cosine similarity: 0.8812349768269672
```

```python
#### INDEXING ####
"""
Example: Loading documents,
         Splitting document,
         embedd document

What is happening:
Takeing each split, we're embedding it
using openAI embeddings into a vector representation,
And then store it with a link to the raw doccument
itself in our vector store
"""
import bs4
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_path=("https://lilianweng.github.cio/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
blog_docs = loader.load()


#### SPLIT ####
from langchain.text_spitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chink_size=300,
    chunk_overlap=50
)

#### Make Splits ####
splits = text_splitter.split_documents(blog_docs)

#### Index ####
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

vectorestore = Chroma.from_documents(documents=splits,
                                     embedding=OpenAIEmbeddings()

retriever = vectorstore.as_retriever()
```

### Retrieval

[note]

> The act of indexing also makes
> the document easy to retrieve

[How to think about it]

> Imagein a 3d space - we embedd the document somwere in 3d space.
> We then make a search query and embedd the search query in 3d space.
> Afterwards we make a similarity surch with the embedded query
> And this allowes os to retriewe the closest 1 - 1000000+ documents
> that matches the search query.
> A point in space is also an representation of the semantic meaning
> This means that points in the same space that are close to eachother
> naturally have a similar semantic meaning which means it likely uf not
> garanteed to be related to each other in some way
> This Idea and thechnolegy is the cornerstone for a lot of search and retrieval
> methods u see using modern vector stores.

[Why does this work]

> By embedding a chunk of a file we represent the symantic meanig of
> that chunk by giving it a position in an x dimensional space by which is the vector.
> When we then embedd the query the query in questions also get's it symantic meanig
> reprecented as a postion in the same x dimensional space. This gives us the
> possibility to compare the simelarity in the querys position in space
> with all the chunkes we've embedded position in space and then retrieve the closest
> x neibours of the querys position in space.

### Generation

[Basic workflow]
First, this introduces the notion of a prompt with placeholder keys where we later can
inject information like context, documents or user input.
A prompt can also be used to alter the the query that's used to retrieve the dcuments
that's stored in our vectorstore.

[example prompt]

```python
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
```

> A sysmtem prompt which injects the context of the retrieved documents.

```python
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
```

> A system prompt designed to alter the query used to fetch from the vectorstore
> Worth noting: If it's the first message the prompt is designed to do nothing -
> this prompt is also only meant as a middle man, it will only alter the query and not
> the actual message that will be sent to the llm
> Read more about this under [Query-Translation]

## Query Translation

[What is Query Translations?]
Query Translations is the first stage of an advanced RAG-pipeline
and the goal of query translation is to take an input user question and
then translate it in some way in order to improve retrieval

[Why Query Translation?]
A users query can be ambiguous and poorly written. This cause problems
since the goal is to do some kind of similarity search between the query
and the vectorstore and if the query is badly written the similarity search
might give the wrong documents.

[Aproches to solve this]

- High abstraction

  - Step-back question
    - Step-back promoting
    - HyDe

- Mid abstraction

  - Query rewriting
    - Multi-query
    - RAG-Fusion

- Low abstraction
  - Sub-Question
    - Least-to-most
    - IR-CoT

### Multi-Query

[Main Idea]
The main idea is that we take the user question and break it down to
diferently worded questions from different perspektives. The intuition here
is that it is possible that the way that a question is initially worded once
embedded is not well aligned or in close proximity in this High dimensional embedding
space to a document we want to retrieve. So by rewriting the question in different
ways can we then combine it with retrieval by comparing the retrieved documents
or combine them in some way that fits the question and then use generation to perform RAG.

[Example Prompt for Multi-Query]

```python
template = """
You are an AI language model assistan. Your task is to generate five
diferent versions of the given user question to retrieve relevant documents from a
vector databse. By generating multiple perspectives on the user question, your goal
is to help the user overcome some of the limitatons of the distance-based similarity search.
Provide these alternative questions separated by newlines. Original question: {question}
"""
```

[Example: How it's done]

> Ask question -> Send it to an llm -> llm generates n different version of question
> -> We do a similarity search on all n versions of questions -> We the smuch the list together -
> and remove duplicates of retrieved documents -> We then send the original question with the -
> retrieved documents to the main llm.

[Example Code Using LangChain]

```python
from langchain.load import dumps, loads
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattend_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattend_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

# Retrieve
question = "what is task decomposition for LLM agents?"
retrieval_chain = generate_queries | retrieve.map() | get_unique_union

# RAG
template = """
Answer the following question based onthis context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(temperatur=0)

final_rag_chain = (
    {"context": retrieval_chain,
     "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"quesion":question})
```

### RAG-fusion

[Main Idea]
It follows the sama main idea as Multy RAG where we take the
user query and make a model reformulate it into different versions
but we then apply a ranking step over our documents.

[RAG Fusion Prompt Example]

```python
from langchain.prompt import ChatPromptTemplate

# RAG-Fusion: Relate
template = """
You are a hekpful assistant that generates multiple search queries based of a single input query. \n
Generate multiple search queries related to: {question} \n
output: {4 queries}:
"""
prompt_rag_fusion = ChatPomptTemplate.from_tamplate(tamplate)
```

[Example Code using LangChain]

```python
from langchain_core.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langhcain.load import dumps, loads

generate_queries = (
    prompt_rag_fusion
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

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
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based of their ranked score in decending order to get the final reranked result
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rankfusion
docs = retrieval_chain_rag_fusion.invoke({"question": question})
```

[Final RAG chain]

```python
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    {"context": retrieval_chain_rag_fusion,
     "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"question": question})
```

### Least-to-most

The object is to first take a question and then decompose it to
a set of subproblems

[Example]
Problem to solve:
Last letter concatenation - The goal is to take a question
"think, machine, learning" and then get to the answer "keg"
using an least-to-most aproch

> Decompose the problem:
> Q: "think, machine, learning"
> A: "think" "think, machine" "think, machine, learning"

This is a least-to-most prompt context (decomposition) for the
last-leter-concatenation task. What it does is that an llm takes the question
which is "think, machine, learning" and decompose it to 3 different
sub problems that it will use to solve the main problem

Figuring out the answer:
The LLM will now take the different subproblems and solv them indevidually
and use the answer of the previous problem to easier solve the problem after.
(ignoring the first one since it's only 1 word)

> Q1: "think, machine"
> A1: The last letter of "think" is "k", the last letter of "machine" is "e",
> this gives us ke. So the output is "ke"
>
> Q2: "think, machine, learning"
> A2: "think, machine" outputs "ke", The last letter of "learing" is "g",
> concatenating "ke" and "g" leads to "keg". So the output is "keg"

### IR-CoT (Interleave Retrieval with Chain of Thougt)

IR-CoT interleaves chain-of-thought generation and knowledge retrieval steps
in order to guide the retrieval with CoT and vice-versa. This interleaving
allows retrievig more relevent information for later reasoning steps, compared
to standard retrieval using solely the question as the query

[Example Prompt]

```python
from langchain.prompts import ChatPromptTemplate

# Decomposition
template = """
You are a helpful assistent that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to {question} \n
ouput (3 questions)
"""
prompt_decomposition = ChatPromptTemplate.from_template(template)
```

[Example usage with LangChain]

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# llm
llm = ChatOpenAI(temperature=0)

# chain
generate_queries_decomposition = (
    prompt_decomposition
    | llm
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

# run
question = "What is the main component of an LLM-powered autonomus agent system?"
questions = generate_queries_decomposition.invoke({"question": question})

print(question)
# output of print():
# ['1. What is LLM technology and how does it work in autonomous agent systems?',
#  '2. What are the specific components that make up an LLM-powered autonomous agent system?',
#  '3. How do the main components of an LLM-powered autonomous agent system interact with each other to enable autonomous functionality?']
```

[Example: Prompt for using query decomposition]

```python
template = """
Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is the availeble background questions and answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context related to the questin:

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the questions: \n {question}
"""

decomposition_template = ChatPromptTemplate.from_template(template)
```

[Example: Using the decom_prompt]

```python
from operator import itemgetter
from langchain_core.parsers import StrOutputParser

def format_qa_paris(question, answer):
    """Format Q and A pairs"""

    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}"

    return formatted_string.strip()

# llm
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

q_a_pairs = ""
for q in question:

    rag_chain = (
        {"context": itemgetter("question") | retriever,
         "question": itemgetter("question"),
         "q_a_pair": itemgetter("q_a_pair") }
        | decomposition_prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke({"question": q, "q_a_pairs": q_a_pairs})
    q_a_pair = format_qa_pair(q,answer)
    q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair
```

[links to papers]
[arxiv](https://arxiv.org/pdf/2205.10625.pdf)
[ICLE-2023](https://arxiv.org/pdf/2205.10625)

### step-back Prompting

The idea here is that we take a question and then make an LLM abstract
more then it allready is. An example here would be taking the question:
"Jan Sindel's was born what year" and then the LLM abstract it to:
"what is Jan Sindel's personal history".
What this does is that it tries to make the query generate a more borader
knowladge så so it has a bigger chanse to actually query what the original
question is asking.

> Step-back is really good when there is a lot of conceptual knwoladge
> that u expect the users to know when the question are asked.
> For example if u need to know the history of Jan Sindel to know
> when he died because he was written of as dead in many different parts
> of his story, then the LLM needs to know the history, not the time of death.

[Example: Prompt and General]

```python
# Few Shot Examples
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
]
# We now transform these to example messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question
            to a more generic step-back question, which is easier to answer. Here are a few examples:""",
        ),
        # Few shot examples
        few_shot_prompt,
        # New question
        ("user", "{question}"),
    ]
)
```

[Example: Usage of Step-Back]

```python
generate_queries_step_back = prompt | ChatOpenAI(temperature=0) | StrOutputParser()
question = "What is task decomposition for LLM agents?"

# Response prompt
response_prompt_template = """
You are an expert of world knowledge. I am going to ask you a question.
Your response should be comprehensive and not contradicted with the following context
if they are relevant. Otherwise, ignore them if they are not relevant.

# {normal_context}
# {step_back_context}

# Original Question: {question}
# Answer:"""

response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

chain = (
    {
        # Retrieve context using the normal question
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        # Retrieve context using the step-back question
        "step_back_context": generate_queries_step_back | retriever,
        # Pass on the question
        "question": lambda x: x["question"],
    }
    | response_prompt
    | ChatOpenAI(temperature=0)
    | StrOutputParser()
)

chain.invoke({"question": question})
```

### HyDe

HyDe as a strategy lies in the assumption that a query question doesn't have enough context
to accurately retrieve the correct documents. The idea is that because the query ultimately
is shorter then the embedded document the chances for the query to get embedded next to the
desired documents to retrieve decreases.

The way HyDe tries to solve this is by taking the query and then add content to it to emulate
the structure of a document so when it's embedded for retrieval it will be closer to the desired
documents.

[Example: Prompt]

```python
from langchain.prompts import ChatPromptTemplate

# HyDE document genration
template = """
Please write a scientific paper passage to answer the question
Question: {question}
Passage:
"""
prompt_hyde = ChatPromptTemplate.from_template(template)

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

generate_docs_for_retrieval = (
    prompt_hyde | ChatOpenAI(temperature=0) | StrOutputParser()
)

# Run
question = "What is task decomposition for LLM agents?"
```

[Example: Creation of Retriever]

```python
# Retrieve
retrieval_chain = generate_docs_for_retrieval | retriever
retireved_docs = retrieval_chain.invoke({"question":question})
retireved_docs
```

[Exmaple: Querying and Answer]

```python
# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"context":retireved_docs,"question":question})

```

## Routing

Routing as a concept is as simply as it sounds like just routing, depending of the questions context, where should
we then route the query to generate the correct documents? Different types of documents require different type of storage
for optimal retrieval and it's the routers job to figure out where we should query the documents from.
There's also the idea of taking the question and then rout it to pre-embedded prompts which is another part
of the routing strategies

- Routes:

  - Logical Routing
    - giving and LLM the knowledge of what data sources
      we can retrieve from and then letting it reason
      about what data source to use depending on the question

  - Semantic Routing
    - The idea here is that this allow us to change systempromp
      based of what the question is that has been asked. So we
      don't just query documents but we also query a system prompt
      to be used for the LLM. In other words we route what prompt to
      use based of the semantic meaning of the question/query

### Logical Routing

[Example: Setting Up Routing]

```python
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

class RouteQuery(BaseModel):
    """
    We Create a structured object in this case holding
    ["python_docs", "js_docs", "golang_docs"]
    This allows to convert it to a OpenAI schema
    and then bind it to our LLM as function calls
    For example the output could be {"datasource": "js_docs"}
    """

    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )

# LLM with function call
# And then binding of RoutQuery
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = llm.with_structured_output(RouteQuery)

# Prompt
system = """You are an expert at routing a user question to the appropriate data source.

Based on the programming language the question is referring to, route it to the relevant data source."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# Define router
router = prompt | structured_llm
```

[Example: Using the Router]

```python
question = """Why doesn't the following code work:

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french")
"""

result = router.invoke({"question": question})
print(result) # RouteQuery(DataSource="python_doc")

def choose_route(result):
    if "python_docs" in result.datasource.lower():
        ### Logic here
        return "chain for python_docs"
    elif "js_docs" in result.datasource.lower():
        ### Logic here
        return "chain for js_docs"
    else:
        ### Logic here
        return "golang_docs"

from langchain_core.runnables import RunnableLambda

full_chain = router | RunnableLambda(choose_route)

full_chain.invoke({"question": question})
```

[Notes:]
As an example could u use this in an Agent made in LangGraph and use it
to decide what NODE/VEC the agent should go to next for fetching of data.

### Semantic Routing

[Example: Code]

```python
from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Two prompts
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{query}"""

math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{query}"""

# Embed prompts
embeddings = OpenAIEmbeddings()
prompt_templates = [physics_template, math_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)

# Route question to prompt
def prompt_router(input):
    # Embed question
    query_embedding = embeddings.embed_query(input["query"])
    # Compute similarity
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    # Chosen prompt
    print("Using MATH" if most_similar == math_template else "Using PHYSICS")
    return PromptTemplate.from_template(most_similar)


chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | ChatOpenAI()
    | StrOutputParser()
)

print(chain.invoke("What's a black hole"))
```

## Query Construction

The idea of taking natural language and the converting it to a domain specific language
In this case are we talking about going for natural language to meta data for filters in
vectorstores.

How it works is that we take the natural language query, construct it to metadata that
matches the structure in some capacity to the meta data that's embedded together with
the documents in the vectordb, and then append it to the original query we'll hopefully
gett better documents queried from the vectordb.

```python
from langchain_community.document_loaders import YoutubeLoader

docs = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=pbAd8O1Lvm4", add_video_info=True
).load()

docs[0].metadata

```

```python
import datetime
from typing import Literal, Optional, Tuple
from langchain_core.pydantic_v1 import BaseModel, Field

class TutorialSearch(BaseModel):
    """Search over a database of tutorial videos about a software library."""

    content_search: str = Field(
        ...,
        description="Similarity search query applied to video transcripts.",
    )
    title_search: str = Field(
        ...,
        description=(
            "Alternate version of the content search query to apply to video titles. "
            "Should be succinct and only include key words that could be in a video "
            "title."
        ),
    )
    min_view_count: Optional[int] = Field(
        None,
        description="Minimum view count filter, inclusive. Only use if explicitly specified.",
    )
    max_view_count: Optional[int] = Field(
        None,
        description="Maximum view count filter, exclusive. Only use if explicitly specified.",
    )
    earliest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Earliest publish date filter, inclusive. Only use if explicitly specified.",
    )
    latest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Latest publish date filter, exclusive. Only use if explicitly specified.",
    )
    min_length_sec: Optional[int] = Field(
        None,
        description="Minimum video length in seconds, inclusive. Only use if explicitly specified.",
    )
    max_length_sec: Optional[int] = Field(
        None,
        description="Maximum video length in seconds, exclusive. Only use if explicitly specified.",
    )

    def pretty_print(self) -> None:
        for field in self.__fields__:
            if getattr(self, field) is not None and getattr(self, field) != getattr(
                self.__fields__[field], "default", None
            ):
                print(f"{field}: {getattr(self, field)}")

```

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

system = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
Given a question, return a database query optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = llm.with_structured_output(TutorialSearch)
query_analyzer = prompt | structured_llm

```


