# app.py
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
#from langchain.embeddings.base import Embeddings
from langchain_core.embeddings import Embeddings

from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from gtts import gTTS
import playsound
import tempfile
import time



pdf_path = "Personal_Knowledge_Base.pdf"
index_path = "personal_vector"
embedding_model_name = "BAAI/bge-base-en-v1.5"

API_KEY=st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=API_KEY)


# sentence transformer model
base_model = SentenceTransformer(embedding_model_name)



###########################################################
class EmbeddingModel(Embeddings):
    def __init__(self, model_sentence):
        self.model = model_sentence

    def embed_documents(self, texts):
        return [self.model.encode(text, show_progress_bar=False).tolist() for text in texts]

    def embed_query(self, text):
        return self.model.encode(text, show_progress_bar=False).tolist()



embedding_model = EmbeddingModel(base_model)



#loading pdf
def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text



# making chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    return splitter.split_text(text)



# making vector database FAISS
def get_faiss_index(pdf_path, index_path, embedding_model):
    if os.path.exists(index_path):
        print(" Loading existing FAISS index")
        vectorstore = FAISS.load_local(
            index_path, embeddings=embedding_model, allow_dangerous_deserialization=True
        )
    else:
        print(" Making new FAISS index...")
        text = get_pdf_text(pdf_path)
        text_chunks = get_text_chunks(text)
        vectorstore = FAISS.from_texts(text_chunks, embedding=embedding_model)
        vectorstore.save_local(index_path)
        print(" FAISS index created and saved.")
    return vectorstore



def build_vectorstore(uploaded_pdf):
    """Handles uploaded PDF file from Streamlit and builds FAISS index."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        tmp_path = tmp.name
    return get_faiss_index(tmp_path, index_path, embedding_model)



def speak_text(text):
    """Convert AI response to voice and play it (Windows-safe)."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_path = fp.name
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(temp_path)
        playsound.playsound(temp_path)
        time.sleep(0.5)
        os.remove(temp_path)
    except Exception as e:
        print(f" Voice playback failed: {e}")




'''
def generate_answer_with_gemini(query, vectorstore):
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.invoke(query)

    
    context = "\n\n".join([d.page_content for d in docs])

  
    prompt = f"""
You are an expert assistant. Use the context below to answer the question clearly and accurately.

Context:
{context}

Question:
{query}

Answer in a concise and helpful manner:
"""

    
    model = genai.GenerativeModel("gemini-2.0-flash")
    chat = model.start_chat(history=[])

   
    response = chat.send_message(prompt)

    
    return response.text

'''






def get_gemini_answer(query, vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    You are an intelligent personal assistant trained to answer questions about Sonu Kumar.

    Use the retrieved context below to provide an accurate and natural answer to the user’s question. 
    If the context does not contain enough information, clearly mention that you don’t have sufficient details to answer confidently — do not make up facts.

    Be concise, friendly, and sound like Sonu Kumar himself when appropriate.

    -----------------------------
    Retrieved Context:
    {context}

    User Query:
    {query}
    -----------------------------

    Final Answer:

    """

    model = genai.GenerativeModel("gemini-2.0-flash")
    chat = model.start_chat(history=[])
    response = chat.send_message(prompt)
    return response.text.strip()

