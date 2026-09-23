import streamlit as st

from main import (
    load_resume,
    split_text,
    add_documents,
    chat_with_bot
)


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Resume Chatbot",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# Header
# ============================================================

st.title("🤖 Resume Chatbot")

st.write(
    "Ask me anything about my resume, skills, "
    "projects, education, or experience."
)


# ============================================================
# Prepare resume database
# ============================================================

if "resume_indexed" not in st.session_state:

    with st.spinner("Loading resume..."):

        try:

            resume_text = load_resume()

            documents = split_text(
                resume_text
            )

            add_documents(
                documents
            )

            st.session_state.resume_indexed = True

        except Exception as error:

            st.error(
                f"Unable to load resume: {error}"
            )

            st.stop()


# ============================================================
# Chat history
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# Chat input
# ============================================================

question = st.chat_input(
    "Ask a question about my resume..."
)


if question:

    # --------------------------------------------------------
    # User message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------------------------
    # AI response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                answer = chat_with_bot(
                    question
                )

            except Exception as error:

                answer = (
                    f"Sorry, an error occurred: {error}"
                )

            st.markdown(answer)


    # --------------------------------------------------------
    # Save response
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("Chatbot")

    st.write(
        "AI-powered resume assistant using "
        "ChromaDB, Sentence Transformers, "
        "and NVIDIA AI."
    )

    st.divider()

    if st.button("Clear Chat"):

        st.session_state.messages = []

        st.rerun()