# 🦜🔗 LangChain & LangGraph AI Projects

A collection of GenAI agents and chatbots exploring the evolution from linear chains to stateful, autonomous systems. This repository demonstrates practical implementations of **LangGraph**, **LangChain**, **RAG (Retrieval Augmented Generation)**, and local LLMs.

---

## 🚀 Featured Project: Autonomous LangGraph Agent

A fully local, stateful AI agent that can reason, remember context, and autonomously decide when to use external tools.

**Tech Stack:** `LangGraph`, `Streamlit`, `Ollama (Qwen 2.5)`, `DuckDuckGo Search`, `AlphaVantage API`

### ✨ Key Features
* **🧠 Autonomous Routing:** The agent analyzes user intent to decide whether to answer directly or trigger a tool (e.g., "What is the price of Tesla?" triggers the Stock Tool).
* **💾 Full Persistence:** Implemented `MemorySaver` checkpointers to maintain chat history across sessions. Users can close the app and "Resume Chat" exactly where they left off.
* **🛠️ Agentic Tool Use:** Connected to live web search and financial APIs for real-time data retrieval.
* **💻 100% Local:** Optimized to run efficiently using quantized local models via Ollama.


## 📺 YouTube RAG Chatbot

A Retrieval Augmented Generation (RAG) application that allows users to "chat" with YouTube videos.

**Tech Stack:** `LangChain`, `FAISS`, `OpenAI/Ollama Embeddings`

* **How it works:** Extracts transcripts from YouTube URLs, chunks the text, creates vector embeddings, and allows users to query the video content using natural language.
* **Use Case:** Quickly summarizing long podcasts or finding specific information in tutorials without watching the whole video.

---

## 📊 Sentiment Analysis & Mini-Projects

A collection of smaller experiments and utility scripts, including:
* **Sentiment Analyser:** A LangChain-based tool that classifies text input as Positive, Negative, or Neutral using prompt engineering.

---

## 🛠️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/taher51-lang/Langchain_AI_projects.git](https://github.com/taher51-lang/Langchain_AI_projects.git)
    cd Langchain_AI_projects
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the LangGraph Agent:**
    Ensure you have [Ollama](https://ollama.com/) installed and running.
    ```bash
    streamlit run app.py
    # (Or navigate to the specific folder if organized into subdirectories)
    ```

## 🤝 Connect
If you are interested in Agentic AI or Local LLMs, feel free to connect!
* **LinkedIn:** [www.linkedin.com/in/taher-rangwala-2a4558340]
* **GitHub:** [taher51-lang](https://github.com/taher51-lang)
