# Website-RAG-Chatbot

A simple yet powerful RAG (Retrieval-Augmented Generation) application that lets you chat with any website using natural language.

The idea behind this project was pretty straightforward: whenever I visit a long blog, article, or documentation page, I don't always want to read everything. Sometimes I just want to ask questions and get the information instantly.
So I built a website intelligence assistant that extracts content from a webpage, retrieves the most relevant information, and utilises Google Gemini to generate context-aware answers.

---

## Features

* Chat with any website URL
* Website content summarization
* Suggested questions generation
* Conversational memory
* Source attribution for transparency
* Semantic search using vector embeddings
* Retrieval-Augmented Generation (RAG) pipeline
* Modern Streamlit chat interface

---

## How It Works

The workflow is pretty simple:

1. Enter a website URL.
2. The website content is extracted using a web loader.
3. The content is split into chunks.
5. Relevant chunks are retrieved whenever a question is asked.
6. Gemini generates an answer using only the retrieved context.
7. Sources used for the answer are displayed to the user.

Workflow:

URL → Website Loader → Text Splitter → Embeddings → Retriever → Gemini → Answer

---

## 🛠️ Tech Stack
### Generative AI
* Google Gemini 2.5 Flash

### Frameworks
* LangChain
* Streamlit

### Embeddings
* HuggingFace Embeddings
* sentence-transformers/all-MiniLM-L6-v2

### Data Processing
* BeautifulSoup
* Requests

### Environment Management
* Python Dotenv

---

## Project Structure

```bash
ContextChat-AI/
│
├── app.py
├── .env
├── requirements.txt
├── README.md
├── website_loader.py
|── rag_pipeline.py
```

---

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd ContextChat-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
```

Run the application:

```bash
streamlit run app.py
```

---

## Example Questions

After processing a website, you can ask questions like:

* What is this website about?
* Summarize the key concepts.
* Explain this topic like I'm a beginner.
* What are the important takeaways?
* Give me a short summary.

---

## Screenshots

I'll be adding screenshots and a demo GIF soon.

---

## Future Improvements

This project is still evolving and I have a lot of ideas I want to add:

* YouTube transcript support
* PDF document chat
* Multi-website knowledge base
* Deployment on Streamlit Cloud / Render
* User authentication

The next major update currently in action is to implement multi-session conversational memory with persistent website-specific knowledge bases.

---

## What I Learned

This project helped me get hands-on experience with:

* Retrieval-Augmented Generation (RAG)
* Embeddings
* Semantic Search
* Prompt Engineering
* Streamlit Application Development
* LLM Integration using Gemini

More importantly, it helped me understand how modern GenAI applications are actually built beyond just calling an API.

---

## Author

Tanishk Sarraf

If you have suggestions, feedback, or ideas for improving the project, feel free to connect with me.

Thanks for checking out the project :)
