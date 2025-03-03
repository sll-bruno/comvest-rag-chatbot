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

"""
This module is responsible for the chatbot that will be used to answer questions about the Comvest.
In the first version, the chatbot will be based on the OpenAI API and will not be able to consider, in the answers, the context of the questions.
In this cenário, the client will not have access and be able to modify the variables of the chatbot.
"""

SYSTEM_PROMPT_TEMPLATE = """
Você é um assistente de IA cuja função é responder perguntas sobre o vestibular da Unicamp em 2024 a partir das publicações da Comvest sobre o vestibular. Seu objetivo é responder as perguntas de forma simples e informativa 
a fim de auxiliar os estudantes que pretendem prestar o vestibular. Se você não souber a resposta, diga "Me desculpe, eu não sei a resposta para essa pergunta".
Considere o contexto dado e a pergunta para dar uma resposta adequada.

Context:
{retrieved_context}
"""

HUMAM_PROMPT_TEMPLATE = """
Pergunta: {question}
"""

#Pass this to the cliente after the development
CHAT_OPENAI_MODEL = "gpt-3.5-turbo"
EMBEDDINGS_MODEL_NAME = 'text-embedding-ada-002'
url = "https://www.pg.unicamp.br/norma/31879/0"

class ComvestChatbot:
    
    def __init__(self, chat_model = CHAT_OPENAI_MODEL, embeddings_model = EMBEDDINGS_MODEL_NAME, url = url):
        
        #ets model name and default setting
        self.chat_model = CHAT_OPENAI_MODEL
        self.embeddings_model = EMBEDDINGS_MODEL_NAME
        self.url = url
        self.temperature = 0
        
        #defines the chatbot model
        self.chat = ChatOpenAI(model_name = CHAT_OPENAI_MODEL, temperature = 0)
        self.embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL_NAME)

        #Load and transform data
        transformed_content = self.__load_data(self.url)
        splitted_content = self.__split_data(transformed_content)
        self.retriever = self.__create_vector_store_and_retriever(splitted_content)
        
        self.prompt = ChatPromptTemplate([
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE),
            HumanMessagePromptTemplate.from_template(HUMAM_PROMPT_TEMPLATE),
        ])
        
    def __load_data(self, page_url):
        #Init the data loader and transformer
        loader = AsyncHtmlLoader([page_url])
        html2text = Html2TextTransformer()
        
        #Extract the html content from page_url
        content = loader.load()
        
        #Transform the html content into text
        transformed_content = html2text.transform_documents(content)
        return transformed_content
        
    def __split_data(self, transformed_content):
        #Init the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size= 800,
            chunk_overlap= 100,
        )
        
        #Split the content and returns it
        splitted_content = text_splitter.split_documents(transformed_content)
        return splitted_content
    
    def __create_vector_store_and_retriever(self, splitted_content):
        
        vector_store = Chroma(
            collection_name = "comvest_data",
            embedding_function = self.embeddings,
            persist_directory = "./chroma_langchain_db",
        )
        
        vector_store.add_documents(splitted_content)
        retriever = vector_store.as_retriever() #init the retriever with default settings
        
        return retriever
    
    def ask_question(self, question):
        retrieved_content = self.retriever.invoke(question)
        joined_context = '\n'.join([doc.page_content for doc in retrieved_content])
        
        formatted_prompt = self.prompt.format_messages(retrieved_context= joined_context, question=question)
        
        response = self.chat.invoke(formatted_prompt)
        
        return response.content
    
    