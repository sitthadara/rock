import streamlit as st
import requests

st.set_page_config(page_title="Pet Rack", page_icon="🪨")

# --- CUSTOM CSS FOR 2-SIDE CHAT (No User Icon) ---
st.markdown("""
    <style>
        /* Hide the User Avatar completely */
        [data-testid="stChatMessageAvatarUser"] {
            display: none !important;
        }

        /* Align User messages to the right */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            flex-direction: row-reverse !important;
            width: fit-content;
        }

        /* Align Assistant (Rock) messages to the left */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
            flex-direction: row;
            text-align: left;
        }

        /* Add bubble styling */
        .stChatMessageContent {
            border-radius: 15px;
            padding: 10px;
            display: inline-block;
        }

        [data-testid="stLayoutWrapper"] {
            flex-direction: row-reverse !important;
        }

        button[kind="header"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🪨 Pet Rock")
st.markdown("It understands everything. It says nothing.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    avatar = message.get("avatar", None)

    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Say something to your rock..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🪨"):
        with st.spinner("Your rock is contemplating..."):
            try:
                res = requests.post("http://backend:8000/chat", json={"message": prompt})
                response_text = res.json().get("response")
            except:
                response_text = "Your rock is currently unreachable."
            
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text, "avatar": "🪨"})