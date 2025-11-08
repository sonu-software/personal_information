# ui.py
import streamlit as st
from app import get_gemini_answer, get_faiss_index, embedding_model, pdf_path, index_path
from gtts import gTTS
import tempfile
import os



#################################################################################
st.set_page_config(page_title=" My Information RAG Chatbot", layout="centered")
st.title("Personal information Assistant")
st.markdown("Ask me anything")




if "messages" not in st.session_state:
    st.session_state.messages = []  # chat history


#cache of vectorstore
@st.cache_resource(show_spinner=False)
def load_vectorstore():
    return get_faiss_index(pdf_path, index_path, embedding_model)

with st.spinner(" Loading your knowledge base..."):
    vectorstore = load_vectorstore()
st.success(" Knowledge base ready!")




# chat input field
user_input = st.chat_input("Type your question...")

if user_input:
    
    st.session_state.messages.append({"role": "user", "content": user_input})

   
    with st.spinner(" Thinking..."):
        answer = get_gemini_answer(user_input, vectorstore)

    
    st.session_state.messages.append({"role": "assistant", "content": answer})

   
    try:
        tts = gTTS(text=answer, lang="en", slow=False)
        audio_path = os.path.join(tempfile.gettempdir(), "gemini_auto_answer.mp3")
        tts.save(audio_path)
        st.session_state.last_audio = audio_path
    except Exception as e:
        st.warning(f"Audio generation failed: {e}")



# Display chat messages 
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])
            



# auto play
if "last_audio" in st.session_state:
    st.audio(st.session_state.last_audio, format="audio/mp3", autoplay=True)
