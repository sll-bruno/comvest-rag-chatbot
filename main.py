import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os 
import re

#Read the .env variables 
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
urls = ["https://vaultresearch.com.br/","https://www.vaultcapital.com.br/?utm_source=site"]

# new
from openai import OpenAI

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

paragraphs = []
headings = []

#extract mkdwn from the urls
for url in urls:
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Erro ao acessar a página: {e}")
        continue

    soup = BeautifulSoup(response.text, "html.parser")

    headings += [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3"])]
    paragraphs += [p.get_text() for p in soup.find_all('p')]

content = headings + paragraphs

#clean the content
cleaned_content = [re.sub(r'\s+', ' ', text) for text in content]  # Remove excessive whitespace
cleaned_content = [re.sub(r'[^\w\s]', '', text) for text in cleaned_content]  # Remove special characters

def text_chunker(text, max_length=5000):
    chunks = []
    current_chunk = ""

    for paragraph in text:
        if len(current_chunk) + len(paragraph) < max_length:
            current_chunk += paragraph + " "
        else:
            chunks.append(current_chunk)
            current_chunk = paragraph + " "

    chunks.append(current_chunk)
    return chunks

def get_embeddings(chunks):
    embedding = []
    for chunk in chunks:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=chunk
        )
        embedding.append(response.data[0].embedding)
    return embedding

chunks = text_chunker(cleaned_content)
embedding = get_embeddings(chunks)
#print(embedding)

import faiss
import numpy as np

dimension = len(embedding[0])  # Get the embedding size
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embedding))

def retriver(query,chunks):
    query_embedding = client.embeddings.create(
        model="text-embedding-ada-002",
        input = query
    ).data[0].embedding

    # Perform similarity search
    distances, indices = index.search(np.array([query_embedding]), k=15)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks

query = "o que é a Vault?"

relevant_chunks = retriver(query,chunks)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system","content": "Você é um consultor de uma relevante empresa de assesoria em criptoativos brasileira chamada Vault Research. Como um importante assessor de investimentos, você conhece profundamente os processos internos da empresa e sua proposta de valor. Seu papel é instruir possiveis clientes em relação aos serviçõs prestados pela Vault. Um cliente entrou em contato com você pedindo informações sobre a empresa. Envie primeiro uma mensagem: ""Qual duvida você tem sobre a vault?"" para entender melhor a necessidade do cliente."},
        {"role": "user", "content": f"Context: {relevant_chunks}\n\nQuery: {query}"}
    ]
)
print(response.choices[0].message.content)

"""with open("extracted_text.txt", "w", encoding="utf-8") as file:
    for text in cleaned_content:
        file.write(text + "\n")"""


