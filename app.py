import streamlit as st
from src.web_loader import load_website
from src.rag_pipeline import  create_chunks, vec_store, create_retriever, ask_question, generate_summary, generate_suggested_questions

st.set_page_config(page_title="Web Content RAG", layout="wide")

# Session state 
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "questions" not in st.session_state:
    st.session_state.questions = []

if "latest_sources" not in st.session_state:
    st.session_state.latest_sources = []

if "current_url" not in st.session_state:
    st.session_state.current_url = ""
    
# Sidebar
with st.sidebar:
    st.title("Controls")
        
# Main Header
st.title("Web Content RAG Chatbot")
st.markdown("Enter a website URL to load content and start asking questions!")

# URL Input
url = st.text_input("Enter Website URL")

# Process Website
if st.button("🚀 Process Website"):

    if not url:
        st.warning("Please enter a URL.")

    else:
        # New URL = New Memory
        st.session_state.messages = []
        st.session_state.latest_sources = []
        st.session_state.summary = ""
        st.session_state.questions = []

        with st.status(
            "Processing Website...",
            expanded=True
        ) as status:
            st.write("🌐 Loading Website...")
            documents = load_website(url)

            st.write("✂️ Splitting Content...")
            chunks = create_chunks(documents)

            st.write("🧠 Creating Embeddings...")
            vector_store = vec_store(chunks)

            st.write("🔍 Creating Retriever...")
            retriever = create_retriever(vector_store)

            st.write("📝 Generating Summary...")
            summary = generate_summary(retriever)

            st.write(
                "💡 Generating Suggested Questions..."
            )
            questions = (
                generate_suggested_questions(retriever)
            )

            st.session_state.retriever = retriever
            st.session_state.summary = summary
            st.session_state.questions = questions
            st.session_state.current_url = url

            status.update(
                label="✅ Website Ready!",
                state="complete"
            )

# Website summary
if st.session_state.summary:
    st.subheader("📄 Website Summary")
    st.info(st.session_state.summary)


# SUGGESTED QUESTIONS
if st.session_state.questions:
    st.subheader(
        "💡 Suggested Questions"
    )

    cols = st.columns(2)

    for i, question in enumerate(
        st.session_state.questions
    ):

        with cols[i % 2]:

            if st.button(
                question,
                key=f"suggested_{i}"
            ):

                response = ask_question(
                    question,
                    st.session_state.retriever,
                    st.session_state.messages[:-1]  # Exclude current question from history
                )

                st.session_state.messages.append({
                    "role": "user",
                    "content": question
                })

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"]
                })

                st.session_state.latest_sources = (
                    response["sources"]
                )

                st.rerun()

# Chat Interface
if st.session_state.retriever:
    col1, col2 = st.columns([5,1])

    with col1:
        st.subheader("💬 Chat")

    with col2:
        if st.button("🗑️ Clear"):

            st.session_state.messages = []
            st.session_state.latest_sources = []

            st.rerun()

    # Display Previous Messages
    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):
            st.markdown(
                message["content"]
            )

    # Chat Input
    user_question = st.chat_input(
        "Ask a question about this website..."
    )

    if user_question:
        # Show User Message
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate Response

        with st.chat_message("assistant"):
            with st.spinner(
                "Thinking..."
            ):

                response = ask_question(
                    user_question,
                    st.session_state.retriever,
                    st.session_state.messages[:-1]  # Exclude current question from history
                )

                answer = response["answer"]

                st.markdown(answer)

                # Save Sources

                st.session_state.latest_sources = (
                    response["sources"]
                )

        # Save Assistant Message

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
        
# Source Panel 
if st.session_state.latest_sources:

    st.subheader("📚 Sources Used")

    for idx, source in enumerate(
        st.session_state.latest_sources,
        start=1
    ):

        with st.expander(
            f"Source {idx}: {source['title']}"
        ):

            st.write(
                f"🔗 {source['source_url']}"
            )

            st.markdown(
                source["content"]
            )
