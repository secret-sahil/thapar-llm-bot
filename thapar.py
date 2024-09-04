import os
os.environ["OPENAI_API_KEY"] = ""

import pandas as pd
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain import OpenAI
from langchain.chains import ConversationalRetrievalChain
import gradio as gr
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')


urls = pd.read_csv("sitemap_urls.csv")
urls = [url for url in urls.iloc[:, 0]]

loaders = UnstructuredURLLoader(urls=urls)
data = loaders.load()

text_splitter = CharacterTextSplitter(separator='\n',
                                      chunk_size=1000,
                                      chunk_overlap=200)
docs = text_splitter.split_documents(data)

embeddings = OpenAIEmbeddings()
vectorStore_openAI = FAISS.from_documents(docs, embeddings)
llm=OpenAI(temperature=0)
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorStore_openAI.as_retriever())

chat_history = []

def answer_question(question):
    global chat_history
    result = chain({"question": question, "chat_history": chat_history})
    chat_history.append((question, result['answer']))
    return result['answer']

iface = gr.Interface(
    fn=answer_question,
    inputs="text",
    outputs="text",
    title="Chatbot",
    description="Ask me!"
)

iface.launch(share=True)
