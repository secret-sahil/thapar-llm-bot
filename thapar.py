# !pip install langchain
# !pip install -U langchain-community
# !pip install unstructured
# !pip install faiss-cpu
# !pip install openai
# !pip install tiktoken
# !pip install gradio
# pip install --upgrade nltk
# pip install sentence-transformers
# pip install google-generativeai

import pandas as pd
import os
import google.generativeai as genai
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
import gradio as gr
import nltk
from sentence_transformers import SentenceTransformer
from langchain.docstore.in_memory import InMemoryDocstore

nltk.download('punkt')
nltk.download('punkt_tab')


genai.configure(api_key="")


urls = pd.read_csv("sitemap_urls.csv")
urls = [url for url in urls.iloc[:, 0]]

loaders = UnstructuredURLLoader(urls=urls)
data = loaders.load()

text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(data)

texts = [doc.page_content for doc in docs]

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embedding_model.encode(texts)

index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

index_to_docstore_id_mapping = {i: str(i) for i in range(len(texts))}

vectorStore = FAISS(
    index=index,
    embedding_function=lambda text: embedding_model.encode([text])[0],  # embedding function
    index_to_docstore_id=index_to_docstore_id_mapping,
    docstore=InMemoryDocstore({str(i): doc for i, doc in enumerate(texts)})
)

def generate_gemini_response(question):
    generation_config = {"temperature": 0.9, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}
    model = genai.GenerativeModel("gemini-pro", generation_config=generation_config)
    response = model.generate_content(question)
    return response.text

chat_history = []

def answer_question(question):
    global chat_history
    result = generate_gemini_response(question)
    chat_history.append((question, result))
    return result

# Gradio interface
iface = gr.Interface(
    fn=answer_question,
    inputs="text",
    outputs="text",
    title="Chatbot",
    description="Ask me anything!"
)

iface.launch(share=True)