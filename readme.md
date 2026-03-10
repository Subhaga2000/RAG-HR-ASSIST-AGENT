# 📚 HR RAG AI Agent using LangGraph + ChromaDB + OpenAI

An **AI-powered HR Assistant Agent** built using **LangGraph, LangChain, OpenAI, and ChromaDB** that answers HR-related questions from an **HR manual PDF** and can generate **professional HR emails with approval before sending**.

The system uses **Retrieval-Augmented Generation (RAG)** to ensure responses come directly from the HR manual instead of hallucinated knowledge.

---

# 🚀 Features

* 📄 HR manual PDF ingestion
* ✂️ Intelligent document chunking
* 🧠 OpenAI embeddings
* 🗄️ Chroma vector database
* 🔎 Semantic HR policy search
* 🤖 LangGraph agent workflow
* 🧰 Tool-based retrieval
* 📧 HR email generation
* ✅ HR Head approval workflow
* 📤 Automatic email sending via Gmail SMTP
* 💬 Interactive CLI HR assistant

---

# 🖼️ Architecture Diagram

Place your architecture diagram image inside:

```
assets/architecture.png
```

Then it will render below:

![HR RAG Agent Architecture](assets/architecture.png)

---

# 🏗️ System Architecture

```
HR Head / User
      ↓
LangGraph Agent
      ↓
OpenAI LLM
      ↓
Retriever Tool
      ↓
Chroma Vector Database
      ↓
Relevant HR Policy Sections
      ↓
LLM Generates Answer
      ↓
(Optional)
Email Draft Tool
      ↓
HR Head Approval
      ↓
SMTP Email Sender
      ↓
Employee Receives Email
```

---

# 📂 Project Structure

```
HR-RAG-AI-AGENT
│
├── rag_agent.py
├── hr_manual.pdf
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── assets
│   └── architecture.png
│
├── vector_db/        # Chroma database (auto created)
└── venv/             # Virtual environment (ignored)
```

---

# 🧰 Technologies Used

| Technology        | Purpose                      |
| ----------------- | ---------------------------- |
| Python            | Core programming language    |
| LangGraph         | Agent workflow orchestration |
| LangChain         | Tool integration             |
| OpenAI GPT        | Natural language reasoning   |
| ChromaDB          | Vector database              |
| OpenAI Embeddings | Document embeddings          |
| PyPDFLoader       | Load HR manual               |
| SMTP (Gmail)      | Email sending                |
| python-dotenv     | Environment configuration    |

---

# ⚙️ Requirements

Recommended Python version:

```
Python 3.11
```

⚠ Python **3.13+ may cause dependency issues** with some LangChain libraries.

---

# 📦 Installation

## 1️⃣ Clone the Repository

```
git clone https://github.com/Subhaga2000/hr-rag-ai-agent.git
cd hr-rag-ai-agent
```

---

## 2️⃣ Create Virtual Environment

```
py -3.11 -m venv venv
```

---

## 3️⃣ Activate Virtual Environment

### Windows

```
venv\Scripts\activate
```

### Linux / Mac

```
source venv/bin/activate
```

---

## 4️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

# 📜 requirements.txt

```
python-dotenv
langgraph
langchain
langchain-core
langchain-openai
langchain-community
langchain-text-splitters
langchain-chroma
chromadb
pypdf
tiktoken
openai
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root.

```
OPENAI_API_KEY=your_openai_api_key

EMAIL_ADDRESS=your_gmail@gmail.com
EMAIL_PASSWORD=your_gmail_app_password
```

⚠ Use a **Gmail App Password**, not your normal Gmail password.

Example `.env.example` file:

```
OPENAI_API_KEY=your_api_key
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
```

---

# 🚀 Running the Agent

Run the script:

```
python rag_agent.py
```

You will see:

```
==== RAG AGENT STARTED ====
Your Question:
```

Then start asking HR questions.

---

# 💬 Example Questions

### HR Policy Question

```
Your Question: What is the employee leave policy?
```

Example output:

```
====== ANSWER ======
Employees are entitled to annual leave according to the HR policy...
```

---

### Generate Welcome Email

```
Write a welcome email for a newly selected employee.

Company: ABC Pvt Ltd
Employee Name: John Silva
Employee Email: john@email.com
```

---

# 📧 Email Workflow

```
User Request
     ↓
Retrieve HR Policy
     ↓
Generate Email Draft
     ↓
HR Head Approval
     ↓
Send Email via SMTP
```

Example approval prompt:

```
====== EMAIL DRAFT ======

To: john@email.com
Subject: Welcome to ABC Pvt Ltd

HR Head approval - send this email? (yes/no)
```

If approved:

```
Email sent successfully
```

---

# 🧠 How the System Works

### Step 1 — Load HR Manual

```
PyPDFLoader
```

Reads the HR manual PDF.

---

### Step 2 — Split Document

```
RecursiveCharacterTextSplitter
```

Breaks the document into smaller chunks.

---

### Step 3 — Generate Embeddings

```
OpenAIEmbeddings
```

Each chunk becomes a vector representation.

---

### Step 4 — Store in Vector Database

```
ChromaDB
```

Embeddings are stored for semantic similarity search.

---

### Step 5 — Create Retriever

```
vectorstore.as_retriever()
```

Finds the most relevant HR policy sections.

---

### Step 6 — LangGraph Agent

The agent:

1. Receives HR question
2. Determines which tool to call
3. Uses retriever tool
4. Retrieves relevant HR policies
5. Generates final answer

---

# 📊 Example Retrieval Workflow

```
User: What is the probation period?

Agent
   ↓
Retriever Tool
   ↓
ChromaDB Vector Search
   ↓
Relevant HR Manual Section
   ↓
LLM Generates Answer
```

---

# 🧹 .gitignore

```
venv/
__pycache__/
.env
vector_db/
*.sqlite3
*.bin
*.log
.vscode/
```

---

# 📤 Upload to GitHub

Initialize Git:

```
git init
```

Add files:

```
git add .
```

Commit:

```
git commit -m "Initial commit - HR RAG AI Agent"
```

Rename branch:

```
git branch -M main
```

Add remote repository:

```
git remote add origin https://github.com/Subhaga2000/hr-rag-ai-agent.git
```

Push code:

```
git push -u origin main
```

---

# 📁 Files to Upload

Upload these files:

```
rag_agent.py
README.md
requirements.txt
.gitignore
.env.example
hr_manual.pdf
assets/architecture.png
```

---

# 🔮 Future Improvements

* 🌐 Streamlit Web Interface
* 📂 Multi-document HR manual support
* 💬 Slack / Teams HR assistant
* 🧠 Agent memory
* ⚡ Streaming responses
* 📊 HR analytics dashboard
* 📑 Source citations with page numbers

---

# 🧑‍💻 Author

Created by **Subhaga Hansamana**

AI / Data Science Projects

---

# ⭐ Summary

This project demonstrates a **complete RAG pipeline with AI agents**, including:

* HR manual knowledge retrieval
* Tool-based LangGraph workflow
* Vector database search
* AI-powered HR question answering
* Automated HR email generation
* Human approval workflow
* SMTP email automation

It is a strong project for learning **modern AI agent development and enterprise AI workflows**.
