from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Prompts 
RAG_PROMPT = PromptTemplate(
    template="""You are ContextChat AI.

Answer ONLY using the provided context.

Rules:
1. Use only retrieved context.
2. If the answer isn't available, say:
   "I could not find that information on this website."
3. Keep answers clear and concise.
4. Explain technical concepts simply.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer: """,
input_variables=["chat_history", "context", "question"]
)

SUMMARY_PROMPT = PromptTemplate(template="""
You are an expert summarizer.

Analyze the website content and provide:

1. Main topic
2. Key concepts
3. Short summary

Keep the response concise and user-friendly.

Content:
{content} """, 
input_variables=["content"])

SUGGESTED_QUESTIONS_PROMPT = PromptTemplate(template="""
Based on the website content below,
generate 5 useful questions a user may ask.

Return only the questions.

Content:
{content} """, 
input_variables=["content"])

# Chaining

rag_chain = RAG_PROMPT | llm
summary_chain = SUMMARY_PROMPT | llm
suggested_questions_chain = SUGGESTED_QUESTIONS_PROMPT | llm

# Splitting the documents
def create_chunks(documents):
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    return splitter.split_documents(documents)

# Vector store
def vec_store(chunks):
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
),
    )
    return vector_store

# Retriever
def create_retriever(vector_store):
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10}
    )
    return retriever

# Chat history 
def format_chat_history(messages):
    
    if not messages:
        return "No previous conversation!"
    
    history = []
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        history.append(
            f"{role.capitalize()}: {content}"
        )
    return "\n".join(history)

# Ask a question
def ask_question(question, retriever, chat_history):
    
    """Main RAG chat function"""
    
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    history = format_chat_history(chat_history)
    
    response = rag_chain.invoke({
        "chat_history": history,
        "context": context,
        "question": question
    })
    
    sources = []
    
    for doc in retrieved_docs:
        sources.append({
            "title" : doc.metadata.get("title", "Unknown Source"),
            "source_url" : doc.metadata.get("source_url", "Unknown URL"),
            "content" : doc.page_content
        })
    
    return {
        "answer": response.content,
        "sources": sources[:1]
    }
    
# Website summary
def generate_summary(retriever):
    """Generate a summary of the website content"""
    docs = retriever.invoke("Summarize the entire website content.")
    
    content = "\n\n".join([doc.page_content for doc in docs])
    
    response = summary_chain.invoke({
        "content": content
    })
    
    return response.content

# Suggested questions
def generate_suggested_questions(retriever):
    """Generate 5 suggested questions based on website content"""
    docs = retriever.invoke("What are some common questions users ask about this website?")
    
    content = "\n".join([doc.page_content for doc in docs])
    
    response = suggested_questions_chain.invoke({
        "content": content
    })
    
    questions = []

    for line in response.content.split("\n"):

        cleaned = (
            line.strip()
            .replace("-", "")
            .replace("*", "")
            .strip()
        )

        if cleaned:
            questions.append(cleaned)

    return questions[:5]