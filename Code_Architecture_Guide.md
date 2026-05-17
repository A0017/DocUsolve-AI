# 🏗️ DocuSolve AI: Code Architecture Guide

This guide breaks down exactly how the codebase (`DocuSolve_App.py`) works so you can confidently explain the technical implementation.

---

## 1. Core Technologies
*   **Streamlit**: The web framework used for the frontend.
*   **LangChain**: The framework used to orchestrate the AI models, prompts, and vector database.
*   **FAISS (Facebook AI Similarity Search)**: The in-memory vector database used to store document data.
*   **Groq**: The ultra-fast inference engine running the `qwen/qwen3-32b` Large Language Model.
*   **HuggingFace**: Used for generating text embeddings (`all-MiniLM-L6-v2`).

---

## 2. Step-by-Step Code Execution

### Step 1: Global Styling and Setup
*   **Glassmorphism CSS**: The app injects a large block of raw CSS to override Streamlit's default look. It applies radial gradients to the background, and `backdrop-filter: blur(24px)` to the main container and sidebar to create the frosted glass effect.
*   **Model Loading (`load_models`)**: Wrapped in `@st.cache_resource`, this ensures the Groq LLM and HuggingFace Embedding models are only loaded into memory once, drastically improving performance.

### Step 2: Document Ingestion (`process_document`)
*   When a user uploads a PDF, it is temporarily saved.
*   `PDFPlumberLoader` extracts the raw text.
*   `RecursiveCharacterTextSplitter` breaks the massive text into smaller chunks (1000 characters each). This is crucial because LLMs have token limits.
*   `FAISS.from_documents` converts these chunks into numbers (embeddings) and creates the searchable vector database.

### Step 3: The Multi-Agent System (Concurrency)
*   When the user clicks **"Executive Review Board"**, the code utilizes Python's `concurrent.futures.ThreadPoolExecutor`.
*   Two functions (representing the Legal Agent and Finance Agent) are dispatched to run at the exact same time. 
*   **How they search:** Each agent queries the FAISS database for different keywords. The Legal agent searches for `"liability indemnification..."` while the Finance agent searches for `"pricing payment..."`.
*   **Synthesis**: Once both threads finish, the results are passed to an "Executive Agent" which uses a specific prompt to read both reports and output a final `APPROVE` or `REJECT` recommendation.

### Step 4: Proposal Drafting
*   The system searches the document for `"requirements problems challenges"`.
*   It feeds these pain points to a specialized "Business Developer" prompt to autonomously generate a targeted proposal.

### Step 5: Interactive Chat & Response Parsing
*   The chat system loops through `st.session_state.messages` to render history.
*   It features an interactive QA where users can type questions.
*   **`parse_response()` Helper**: Because the Qwen LLM sometimes wraps its internal reasoning inside `<think>` tags, this centralized helper function cleanly strips out the "thoughts" to ensure the user only sees the final, polished answer.

### Step 6: Enterprise Integrations (Webhook)
*   Located at the bottom of the script, this logic captures the user's selected action (like "Sync to Salesforce").
*   It packages the document name, the target ID, and the most recent AI generated data into a JSON payload.
*   It uses the `requests.post()` library to fire this payload to a local webhook (`http://localhost:5678`), simulating a real enterprise backend integration.
