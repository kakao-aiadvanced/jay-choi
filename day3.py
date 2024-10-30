#! /usr/bin/env python

from tavily import TavilyClient
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
import json

llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0)
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs_list = WebBaseLoader(urls).load()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
)
retriever = vectorstore.as_retriever()

from pprint import pprint
from typing import List

from langchain_core.documents import Document
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

### Retrieval Grader
system = """
    You are a grader evaluating the relevance of retrieved documents to the user's question.
    If a document contains keywords related to the user's question, evaluate it as relevant. This doesn't need to be a strict test. The goal is to filter out irrelevant search results.
    Give a binary score of 'yes' or 'no' for whether the document is relevant to the question.
    Provide the score in JSON with only a 'relevance' key, without any explanations or preamble.
    """

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "question: {question}\n\n document: {document} "),
    ]
)

retrieval_grader = prompt | llm | JsonOutputParser()

### Generate

system = """
    You are an assistant for question-answering tasks.
    Use the provided search context to answer the question. If you don't know the answer, say so.
    Keep your response concise using maximum 3 sentences.
    """

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "question: {question}\n\n context: {context} "),
    ]
)

# Chain
rag_chain = prompt | llm | StrOutputParser()

### Hallucination Grader
system = """
    You are a grader assessing whether the response is based on or supported by the given facts. 
    Provide a binary score of 'yes' or 'no' indicating whether the response is fact-based or supported. 
    Deliver the score as a JSON with only the 'score' key, and do not include any explanations or preamble.
    """

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "documents: {documents}\n\n answer: {generation} "),
    ]
)

hallucination_grader = prompt | llm | JsonOutputParser()

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
    relevance: str
    documents: List[str]


### Nodes

def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def relevance_check(state):
    """
    Check if the retrieved documents are relevant to the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered documents and updated web_search state
    """
    print("---RELEVANCE CHECK---")
    if "web_search" in state:
        print(state["web_search"])
    else:
        print("no web_search")
    relevance = retrieval_grader.invoke(
        {"question": state["question"], "document": state["documents"]}
    )
    return {"documents": state["documents"], "question": state["question"], "relevance": relevance}

def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def hallucination_check(state):
    """
    Check if the generation is hallucinated

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): return current state
    """
    # check state has websearch key
    print("---HALLUCINATION CHECK---")
    if "web_search" in state:
        if state["web_search"] == "yes":
            print("---HALLUCINATION CHECK: SKIP FOR WEB SEARCH---")
        return state
    else:
        question = state["question"]
        documents = state["documents"]
        hallucination = hallucination_grader.invoke(
            {"documents": documents, "generation": question}
        )

        score = hallucination["score"]
        print(hallucination)
        if score.lower() == "yes":
            print("success: no hallucination")
            return state
        else:
            print("failed: hallucination")
            # print question and documents
            exit()

def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents, and insert "websearch": "yes" for relevance check
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = None
    if "documents" in state:
      documents = state["documents"]

    # Web search
    docs = tavily.search(query=question)['results']

    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents = [web_results]
    return {"documents": documents, "question": question, "web_search": "yes"}

### Edges

def decide_to_generate_or_websearch(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    relevance = state["relevance"]
    web_search = relevance['relevance'].lower() == "no"
    state["documents"]

    # 관계가 없으면 websearch 호출
    if web_search:
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.set_entry_point("retrieve")
workflow.add_node("relevance_check", relevance_check)
workflow.add_edge("retrieve", "relevance_check")
workflow.add_node("websearch", web_search)  # web search
workflow.add_edge("websearch", "relevance_check")

# relevance check가 yes인 경우인 경우 websearch 호출
workflow.add_conditional_edges(
    "relevance_check",
    decide_to_generate_or_websearch,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)

workflow.add_node("generate", generate)
workflow.add_node("hallucination_check", hallucination_check)
workflow.add_edge("generate", "hallucination_check")
# Compile
app = workflow.compile()

# import curve style
from langchain_core.runnables.graph import CurveStyle
from langchain_core.runnables.graph import NodeStyles
from langchain_core.runnables.graph import MermaidDrawMethod

import nest_asyncio

nest_asyncio.apply()  # Required for Jupyter Notebook to run async functions

# write image file
image = app.get_graph().draw_mermaid_png(
    curve_style=CurveStyle.LINEAR,
    node_colors=NodeStyles(first="#ffdfba", last="#baffc9", default="#fad7de"),
    wrap_label_n_words=9,
    output_file_path="graph.png",
    draw_method=MermaidDrawMethod.PYPPETEER,
    background_color="white",
    padding=10,
)

with open("graph.png", "wb") as f:
    f.write(image)

# display(
#     Image(
#         app.get_graph().draw_mermaid_png(
#             curve_style=CurveStyle.LINEAR,
#             node_colors=NodeStyles(first="#ffdfba", last="#baffc9", default="#fad7de"),
#             wrap_label_n_words=9,
#             output_file_path=None,
#             draw_method=MermaidDrawMethod.PYPPETEER,
#             background_color="white",
#             padding=10,
#         )
#     )
# )

# command line에서 query를 입력
import sys
query = sys.argv[1]

inputs = {"question": query}
outputs = app.stream(inputs)
for output in outputs:
    for key, value in output.items():
        pprint(f"Finished running: {key}:")

# print documents for generated answer

pprint(value["generation"])
