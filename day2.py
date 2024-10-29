#! /usr/bin/env python

import getpass
import os

# os.environ['OPENAI_API_KEY'] = getpass.getpass("Enter your OpenAI API key: ")
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# load urls with WebBaseLoader
doclist = [WebBaseLoader(url).load() for url in urls]
# loader = WebBaseLoader(
#     web_paths=urls
# )

docs = [item for sublist in doclist for item in sublist]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, collection_name="rag-chroma", embedding=OpenAIEmbeddings(model="text-embedding-3-small"))

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retriever = vectorstore.as_retriever()
retriever.search_type = "similarity"
retriever.search_kwargs["k"] = 6

query = "agent memory"
retrieved_docs = retriever.invoke(query)

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

parser = JsonOutputParser()

prompt = PromptTemplate(
    input_variables=['query', 'retrieved_chunk'],
    template=(
        "check the relevance of the following each of retrieved chunk list to the query. answer to relevance is yes or no.\n"
        "{format_instructions}\n"
        "query: {query}\n"
        "retrieved chunk: {retrieved_chunk}\n"
    ),
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser

# chunks from retrieved_docs[n].page_content
chunks = []
for doc in retrieved_docs:
    chunks.append(doc.page_content)

relevances = chain.invoke({"query": query, "retrieved_chunk": chunks})
# for relevance in relevances['relevance']:
#     print("Relevant: ", relevance['relevant'])
#     print("Chunk:", relevance["chunk"])
#     print()


# using ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "answer with chunks for relevant is only yes"),
    ("user", "chunks is {relevances}"),
])

answer_chain = chat_prompt | llm | StrOutputParser()

answer = answer_chain.invoke({"relevances": relevances})

print(answer)