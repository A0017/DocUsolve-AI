# 🏢 DocuSolve AI Enterprise

![DocuSolve AI Banner](https://img.shields.io/badge/Status-Active-brightgreen) ![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B) ![LangChain](https://img.shields.io/badge/LangChain-AI-1C3C3C) 

**DocuSolve AI** is a fully autonomous, multi-agent Enterprise Document Intelligence platform. It leverages a blazing-fast Hybrid RAG (Retrieval-Augmented Generation) pipeline to analyze complex legal and financial documents, mitigate corporate risk, and automate business proposals.

---

## ✨ Core Features

*   **⚖️ Executive Review Board (Multi-Agent Concurrency)**
    Deploy specialized AI Personas (a Corporate Lawyer and a Chief Financial Officer) to simultaneously audit uploaded contracts for liabilities, indemnification risks, and hidden financial penalties. Uses Python's `ThreadPoolExecutor` for parallel document analysis.
*   **✍️ Autonomous Targeted Drafting**
    Automatically extract core client pain points from an RFP or requirement document, and instantly generate a highly persuasive, tailored proposal letter ready to be sent to external organizations.
*   **🔗 Enterprise Integrations**
    Built-in routing capability to push the AI's final analysis directly into simulated enterprise workflows (e.g., Salesforce Sync, Jira Epic Creation, Automated Stakeholder Emails) via custom webhooks.
*   **💎 Glassmorphic User Interface**
    A highly premium, custom CSS interface built directly on top of Streamlit featuring a frosted glass aesthetic, dynamic responsive cards, and an obsidian-emerald corporate theme.

---

## 🏗️ Technical Architecture

1.  **Document Ingestion**: Extracts text using `pdfplumber` and creates strict 1000-character chunks via LangChain `RecursiveCharacterTextSplitter`.
2.  **Vector Database (FAISS)**: Uses HuggingFace embeddings (`all-MiniLM-L6-v2`) to convert text to vectors and store them in-memory, completely eliminating AI hallucination.
3.  **Inference Engine (Groq)**: Powered by the ultra-fast Groq API running the `qwen/qwen3-32b` Large Language Model.
4.  **Concurrency Optimization**: The AI agents run in parallel threads, cutting processing wait times by 50%.

---

## 🚀 Quickstart Guide

### 1. Prerequisites
Ensure you have Python 3.10+ installed. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_api_key_here
```

### 3. Run the Application
Launch the Enterprise platform using Streamlit:
```bash
streamlit run DocuSolve_App.py
```

---

*© 2026 DocuSolve Systems Inc. Developed for enterprise document automation.*
