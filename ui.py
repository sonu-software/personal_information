# ui.py
import streamlit as st
from app import get_gemini_answer, get_faiss_index, embedding_model, pdf_path, index_path
from gtts import gTTS
import tempfile
import os

#################################################################################
st.set_page_config(page_title="My Information RAG Chatbot", layout="centered")
st.title("Personal Information Assistant")
st.markdown("Ask me anything")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Cache vectorstore
@st.cache_resource(show_spinner=False)
def load_vectorstore():
    return get_faiss_index(pdf_path, index_path, embedding_model)

with st.spinner("Loading your knowledge base..."):
    vectorstore = load_vectorstore()
st.success("Knowledge base ready!")

# Chat input
user_input = st.chat_input("Type your question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        answer = get_gemini_answer(user_input, vectorstore)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    try:
        # Generate TTS audio
        tts = gTTS(text=answer, lang="en", slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tts.save(tmp.name)
            st.session_state.last_audio = tmp.name
    except Exception as e:
        st.warning(f"Audio generation failed: {e}")

# Display chat history
for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role):
        st.write(msg["content"])

# ✅ FIXED: Stream the audio file content as bytes
if "last_audio" in st.session_state:
    try:
        with open(st.session_state.last_audio, "rb") as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)
    except Exception as e:
        st.warning(f"Unable to play audio: {e}")
