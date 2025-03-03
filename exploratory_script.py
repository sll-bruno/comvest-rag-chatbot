from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import chromadb
import os
from bs4 import BeautifulSoup
from langchain_community.document_loaders import AsyncHtmlLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


CHAT_OPENAI_MODEL = "gpt-3.5-turbo"
EMBEDDINGS_MODEL_NAME = 'text-embedding-ada-002'
url = "https://www.pg.unicamp.br/norma/31879/0"

#Read the .env variables 
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

chat = ChatOpenAI(model_name = CHAT_OPENAI_MODEL, temperature = 0)
embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL_NAME)

loader = AsyncHtmlLoader([url])
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size= 800,
    chunk_overlap= 100,
)
html2text = Html2TextTransformer()

content = loader.load()
transformed_content = html2text.transform_documents(content)
splitted_content = text_splitter.split_documents(transformed_content)

vector_store = Chroma(
    collection_name = "example_collection",
    embedding_function = embeddings,
    persist_directory = "./chroma_langchain_db",
)

vector_store.add_documents(splitted_content)

retriever = vector_store.as_retriever()

PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Context: {context} 
Answer:
"""

HUMAM_TEMPLATE = """
Question:
{question}
"""

question = "Como é o processo seletivo de ingresso na unicamp?"

retrieved_contexts = retriever.invoke(question) #Buscas baseadas em similaridade semantica
joined_contexts = '\n'.join([doc.page_content for doc in retrieved_contexts])

final_prompt = ChatPromptTemplate([
    SystemMessagePromptTemplate.from_template(PROMPT_TEMPLATE),
    HumanMessagePromptTemplate.from_template(HUMAM_TEMPLATE)
])

formatted_prompt = final_prompt.format_messages(context= joined_contexts, question=question)

response = chat.invoke(formatted_prompt)

print(response.content)