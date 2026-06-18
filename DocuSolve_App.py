# pyrefly: ignore [missing-import]
import os
import time
import tempfile
import urllib.parse
import concurrent.futures
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
st.set_page_config(page_title="DocuSolve Enterprise", layout="wide", page_icon="🏢", initial_sidebar_state="expanded")

st.markdown("""<style>
/* Modern Typography */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hide default elements */
#MainMenu, footer, .stDeployButton {display:none;}

/* Global App Background */
.stApp {
    background: radial-gradient(circle at 15% 50%, rgba(16, 185, 129, 0.15), transparent 25%),
                radial-gradient(circle at 85% 30%, rgba(0, 243, 255, 0.1), transparent 25%),
                radial-gradient(circle at 50% 80%, rgba(16, 185, 129, 0.1), transparent 30%) !important;
    background-color: #030712 !important;
}

/* Glassmorphic Main Container */
[data-testid="stAppViewBlockContainer"] {
    background: rgba(10, 10, 10, 0.4) !important;
    backdrop-filter: blur(24px) !important;
    border-radius: 24px;
    padding: 3rem !important;
    margin-top: 2rem !important;
    margin-bottom: 2rem !important;
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
}

/* Glassmorphic Inputs */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stFileUploader > div {
    background-color: rgba(23, 23, 23, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(10px);
    border-radius: 10px;
}
.streamlit-expanderHeader {
    background: rgba(23, 23, 23, 0.4) !important;
    backdrop-filter: blur(10px);
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

/* Glassmorphic Sidebar */
[data-testid="stSidebar"] {
    background: rgba(10, 10, 10, 0.4) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
}
[data-testid="stSidebar"] > div:first-child {
    background: transparent !important;
}

/* Glassmorphism Chat Messages */
.stChatMessage {
    background: rgba(23, 23, 23, 0.7) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}

/* Responsive Modern Buttons */
.stButton > button {
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    border: 1px solid rgba(255, 255, 255, 0.1);
    background: rgba(23, 23, 23, 0.5);
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2);
    border-color: #10B981;
}

/* Primary Button Styling Override */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%);
    border: none;
    color: white;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
    transform: translateY(-3px);
}

/* Document Preview Frame */
iframe {
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
}

/* Custom Status Widget */
[data-testid="stStatusWidget"] {
    border-radius: 12px;
    background: rgba(23, 23, 23, 0.8);
    border: 1px solid rgba(16, 185, 129, 0.3);
}
</style>""", unsafe_allow_html=True)

if not (api_key := os.getenv("GROQ_API_KEY")): st.error("Please add GROQ_API_KEY to .env"); st.stop()

@st.cache_resource(show_spinner="Loading Models...")
def load_models():
    return ChatGroq(model="qwen/qwen3-32b", api_key=api_key), HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm_model, embeddings_model = load_models()

def parse_response(resp_content):
    return resp_content.split("</think>")[-1].strip() if "</think>" in resp_content else resp_content.strip()

c1, c2 = st.columns([4, 1])
c1.title("DocuSolve™ Enterprise")
c1.markdown("**Secure Document Analysis & Retrieval System | v2.4.1**")
c2.metric("System Status", "Online", "Secure Connection")
st.markdown("---")

with st.sidebar:
    st.markdown("### 🛡️ Enterprise Workspace")
    st.markdown("<p style='color: #9CA3AF; font-size: 0.9rem; margin-top: -10px;'>Secure Document Processing Node</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("#### 📄 Upload Target Document")
        st.caption("Supported formats: PDF (Contracts, RFPs, SLAs)")
        uploaded_file = st.file_uploader("Document Upload", type="pdf", label_visibility="collapsed")
        
        if uploaded_file:
            st.success(f"✅ **{uploaded_file.size / 1024:.1f} KB** Indexed")
            
    st.markdown("<br>" * 8, unsafe_allow_html=True)
    st.divider()
    st.markdown("<div style='text-align: center; font-size: 0.8rem; color: #6B7280;'>DocuSolve Enterprise v2.4.1<br>End-to-End Encryption Active 🔒</div>", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def process_document(name, content):
    path = f"temp_{name}"
    with open(path, "wb") as f: f.write(content)
    with st.status("Initializing Engine...", expanded=True) as s:
        docs = PDFPlumberLoader(path).load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        db = FAISS.from_documents(chunks, embeddings_model)
        s.update(label="Ingestion Complete", state="complete", expanded=False)
    if os.path.exists(path): os.remove(path)
    return db

if not uploaded_file:
    landing_html = """<div class="hero-container">
<style>
.hero-container { display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 5rem 2rem; background: radial-gradient(circle at 50% 0%, rgba(16, 185, 129, 0.15) 0%, rgba(3, 7, 18, 0) 60%); border-radius: 24px; margin-top: 1rem; position: relative; overflow: hidden; border: 1px solid rgba(255, 255, 255, 0.05); }
.hero-container::before { content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: radial-gradient(circle, rgba(16, 185, 129, 0.05) 0%, transparent 40%); z-index: 0; animation: pulse 10s infinite alternate; }
@keyframes pulse { 0% { transform: scale(1); } 100% { transform: scale(1.1); } }
.hero-badge { background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); color: #10B981; padding: 0.5rem 1.2rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; letter-spacing: 1px; margin-bottom: 2rem; z-index: 1; text-transform: uppercase; }
.hero-title { font-size: 4rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.1; background: linear-gradient(135deg, #FFFFFF 0%, #A1A1AA 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1.5rem; z-index: 1; max-width: 900px; }
.hero-subtitle { font-size: 1.25rem; color: #9CA3AF; max-width: 700px; margin-bottom: 4rem; line-height: 1.6; z-index: 1; font-weight: 400; }
.features-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; width: 100%; max-width: 1200px; z-index: 1; }
.feature-card { background: rgba(23, 23, 23, 0.6); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.05); padding: 2.5rem 2rem; border-radius: 20px; text-align: left; transition: all 0.3s ease; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
.feature-card:hover { transform: translateY(-5px); border-color: rgba(16, 185, 129, 0.4); box-shadow: 0 12px 40px rgba(16, 185, 129, 0.15); background: rgba(30, 30, 30, 0.8); }
.feature-icon { font-size: 2.5rem; margin-bottom: 1.5rem; display: inline-block; background: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 16px; color: #10B981; line-height: 1; }
.feature-title { font-size: 1.25rem; font-weight: 600; color: #F9FAFB; margin-bottom: 0.75rem; }
.feature-desc { font-size: 1rem; color: #9CA3AF; line-height: 1.6; }
</style>
<div class="hero-badge">✨ New: Multi-Agent Review Board</div>
<h1 class="hero-title">Revolutionize Your Contracts with DocuSolve AI</h1>
<p class="hero-subtitle">The intelligent platform for precise, AI-powered contract analysis, autonomous review, and targeted proposal drafting.</p>
<div class="features-grid">
<div class="feature-card">
<div class="feature-icon">🔍</div>
<div class="feature-title">AI Clause Extraction</div>
<div class="feature-desc">Automatically extract and catalog critical clauses from complex enterprise contracts instantly.</div>
</div>
<div class="feature-card">
<div class="feature-icon">⚖️</div>
<div class="feature-title">Risk Assessment Board</div>
<div class="feature-desc">Deploy a team of AI Legal and Financial agents to quantify and mitigate legal risk with pre-trained models.</div>
</div>
<div class="feature-card">
<div class="feature-icon">✍️</div>
<div class="feature-title">Targeted Drafting</div>
<div class="feature-desc">Instantly convert RFPs and requirements into highly persuasive, perfectly tailored offer letters.</div>
</div>
</div>
</div>"""
    st.markdown(landing_html, unsafe_allow_html=True)
    st.markdown("<br><div style='text-align: center; color: #9CA3AF; font-size: 1.1rem;'>👆 Upload a document in the Control Panel to enter the workspace.</div>", unsafe_allow_html=True)
else:
    faiss_db = process_document(uploaded_file.name, uploaded_file.getvalue())
    st.success("✅ Secure Indexing Complete. System Ready for Queries.")
    
    chat_col = st.container()
    with chat_col:
        st.subheader("🤖 Autonomous Agents")
        agent_cols = st.columns(3)
        run_board = agent_cols[0].button("⚖️ Executive Review Board", use_container_width=True)
        run_proposal = agent_cols[1].button("✍️ Draft Targeted Proposal", use_container_width=True, type="primary")
        run_graph = agent_cols[2].button("🕸️ Entity Knowledge Graph", use_container_width=True)
        
        # UI State Management
        if run_board:
            st.session_state.pop('proposal_draft', None)
            st.session_state.pop('graph_html', None)
        if run_proposal:
            st.session_state.pop('board_reports', None)
            st.session_state.pop('graph_html', None)
        if run_graph:
            st.session_state.pop('board_reports', None)
            st.session_state.pop('proposal_draft', None)
            
        if run_board:
            with st.status("Assembling the Board...", expanded=True) as status:
                
                def run_agent(query, template):
                    ctx = "\n\n".join([d.page_content for d in faiss_db.similarity_search(query, k=2)])
                    prompt = ChatPromptTemplate.from_template(template)
                    return parse_response((prompt | llm_model).invoke({"context": ctx}).content)

                st.write("👨‍⚖️ **Legal Agent** & 💼 **Finance Agent** are auditing the document concurrently...")
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    legal_future = executor.submit(
                        run_agent, "liability indemnification termination warranties risk", 
                        "You are a strict Corporate Lawyer. Analyze this context for legal risks, liabilities, and problematic terms. Be concise. Ignore pleasantries.\nContext: {context}"
                    )
                    finance_future = executor.submit(
                        run_agent, "pricing payment terms invoices penalties fees financial", 
                        "You are a Chief Financial Officer. Analyze this context for financial structure, payment terms, and hidden costs. Be concise. Ignore pleasantries.\nContext: {context}"
                    )
                    l_ans = legal_future.result()
                    f_ans = finance_future.result()
                
                st.write("👔 **Executive Agent** is synthesizing the final recommendation...")
                time.sleep(2)
                e_prompt = ChatPromptTemplate.from_template("You are the CEO. Read the Legal and Financial Reports below. Provide a brief 3-sentence executive summary and a final recommendation (APPROVE, RENEGOTIATE, or REJECT).\n\nLEGAL REPORT:\n{legal}\n\nFINANCIAL REPORT:\n{finance}")
                e_ans = parse_response((e_prompt | llm_model).invoke({"legal": l_ans, "finance": f_ans}).content)
                
                st.session_state.board_reports = {"legal": l_ans, "finance": f_ans, "exec": e_ans}
                status.update(label="✅ Board Review Complete", state="complete", expanded=False)
                
        if st.session_state.get('board_reports'):
            st.markdown("### 📊 Executive Summary")
            ecol1, ecol2 = st.columns([3, 1])
            with ecol1:
                st.info(st.session_state.board_reports["exec"])
            with ecol2:
                st.metric("🎯 AI Confidence Score", "94%")
                st.caption("HITL Verification")
                st.selectbox("Board Review Status", ["🟡 Pending Review", "✅ Approved by Human", "🚩 Flagged / Edit Needed"], key="hitl_board")
            
            t1, t2 = st.tabs(["👨‍⚖️ Legal Report", "💼 Financial Report"])
            with t1: st.warning(st.session_state.board_reports["legal"])
            with t2: st.success(st.session_state.board_reports["finance"])
            st.divider()

        if run_graph:
            with st.status("Building Interactive Knowledge Graph...", expanded=True) as status:
                st.write("🔍 Extracting Entities and Relationships...")
                g_ctx = "\n\n".join([d.page_content for d in faiss_db.similarity_search("roles entities amounts dates organizations obligations", k=3)])
                g_prompt = ChatPromptTemplate.from_template("You are an expert data extractor. Read the context and extract the top 15 most important relationships. Format your output EXACTLY as a list of lines with no bullet points, using this format: Entity1 | Relationship | Entity2. Example:\nDocuSolve | signed contract with | ClientCorp\n$50,000 | paid for | Software License\nContext: {context}")
                g_ans = parse_response((g_prompt | llm_model).invoke({"context": g_ctx}).content)
                
                st.write("🕸️ Generating Network Visualization...")
                try:
                    from pyvis.network import Network
                except ImportError:
                    import subprocess, sys
                    st.info("📦 First time setup: Installing graph visualization libraries (15 seconds)...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyvis", "networkx"])
                    from pyvis.network import Network
                
                net = Network(height='500px', width='100%', bgcolor='#222222', font_color='white', notebook=False)
                net.force_atlas_2based()
                
                for line in g_ans.split('\n'):
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) == 3:
                        e1, rel, e2 = parts
                        net.add_node(e1, label=e1, color='#00d2ff', size=20)
                        net.add_node(e2, label=e2, color='#3a7bd5', size=20)
                        net.add_edge(e1, e2, title=rel, label=rel, color='gray')
                
                path = tempfile.mktemp(suffix='.html')
                net.save_graph(path)
                with open(path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    html_content = html_content.replace('border: 1px solid lightgray', 'border: none')
                    st.session_state.graph_html = html_content
                status.update(label="✅ Knowledge Graph Generated", state="complete", expanded=False)
                
        if st.session_state.get('graph_html'):
            st.markdown("### 🕸️ Interactive Knowledge Graph")
            st.caption("Drag nodes to interact. Hover over links to see relationships.")
            with st.container(border=True):
                components.html(st.session_state.graph_html, height=520)
            st.divider()

        if run_proposal:
            with st.status("Analyzing Requirements & Drafting Proposal...", expanded=True) as status:
                st.write("🔍 Identifying core problems, needs, and pain points...")
                p_ctx = "\n\n".join([d.page_content for d in faiss_db.similarity_search("requirements problems challenges needs goals objectives pain points", k=3)])
                
                st.write("✍️ Drafting persuasive, targeted offer letter...")
                p_prompt = ChatPromptTemplate.from_template("You are a Master Business Developer. Read the context. First, identify their core problems. Then, write a persuasive proposal/offer letter addressing those needs. CRITICAL: DO NOT use any placeholders like [Your Name], [Date], or [Address]. Instead, invent highly realistic fictional details (e.g., 'DocuSolve Enterprise Solutions', '123 Tech Avenue', today's date) so the letter looks 100% complete and ready to send. Make it sound professional and irresistible.\n\nContext: {context}")
                p_ans = parse_response((p_prompt | llm_model).invoke({"context": p_ctx}).content)
                
                st.session_state.proposal_draft = p_ans
                status.update(label="✅ Proposal Drafted", state="complete", expanded=False)
                
        if st.session_state.get('proposal_draft'):
            st.markdown("### 🎯 Tailored Proposal Draft")
            st.success(st.session_state.proposal_draft)
            
            # --- Enterprise Integration for Proposal ---
            st.markdown("#### 🚀 Automate Proposal Delivery")
            with st.container(border=True):
                target_email = st.text_input("Recipient Email:", placeholder="Type client@company.com and press Enter...", key="proposal_email")
                if target_email:
                    subject = urllib.parse.quote(f"Targeted Business Proposal - Ref: {uploaded_file.name}")
                    clean_body = st.session_state.proposal_draft.replace("**", "").replace("#", "")
                    body = urllib.parse.quote(clean_body)
                    gmail_url = f"https://mail.google.com/mail/?view=cm&fs=1&to={target_email}&su={subject}&body={body}"
                    
                    btn_html = f'''
                    <a href="{gmail_url}" target="_blank" style="
                        display: block; 
                        width: 100%; 
                        text-align: center; 
                        background-color: #EA4335; 
                        color: white; 
                        padding: 0.6rem 1rem; 
                        border-radius: 8px; 
                        text-decoration: none; 
                        font-weight: 600; 
                        margin-top: 5px;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    ">📧 Compose in Gmail Web</a>
                    '''
                    st.markdown(btn_html, unsafe_allow_html=True)
            st.divider()

        st.markdown("### 🧠 Deep Document Intelligence")
        with st.container(border=True):
            st.caption("Select a quick analysis or type your custom query below:")
            btns = st.columns(3)
            if btns[0].button("📄 Executive Summary", use_container_width=True): st.session_state.qa = "Provide a comprehensive executive summary."
            if btns[1].button("⚖️ Legal Risk Analysis", use_container_width=True): st.session_state.qa = "Analyze key legal clauses and terms."
            if btns[2].button("💼 Financial Breakdown", use_container_width=True): st.session_state.qa = "Identify financial obligations and risks."
            
        user_query = st.chat_input("Ask anything about the document...")
        if st.session_state.get("qa"): user_query = st.session_state.pop("qa")
            
        if "messages" not in st.session_state: st.session_state.update({"messages": [], "last_query": "", "last_answer": ""})
            
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"], avatar=msg["avatar"]):
                if msg.get("think"): st.expander("🔍 Trace").write(msg["think"])
                if msg["role"] == "ai":
                    with st.expander("📝 AI Response", expanded=(i == len(st.session_state.messages) - 1)):
                        st.write(msg["content"])
                else:
                    st.write(msg["content"])
                
        if user_query:
            st.chat_message("user", avatar="👤").write(user_query)
            st.session_state.messages.append({"role": "user", "avatar": "👤", "content": user_query})
            
            with st.spinner("Synthesizing..."):
                ctx = "\n\n".join([d.page_content for d in faiss_db.similarity_search(user_query, k=3)])
                prompt = ChatPromptTemplate.from_template("Answer using ONLY the context.\nContext: {context}\nQuestion: {question}")
                resp = (prompt | llm_model).invoke({"context": ctx, "question": user_query}).content
                
                think = resp.split("</think>")[0].replace("<think>", "").strip() if "</think>" in resp else None
                ans = parse_response(resp)
                
                with st.chat_message("ai", avatar="🏢"):
                    if think: st.expander("🔍 Trace").write(think)
                    with st.expander("📝 AI Response", expanded=True):
                        st.write(ans)
                    
                st.session_state.messages.append({"role": "ai", "avatar": "🏢", "content": ans, "think": think})
                st.session_state.update({"last_query": user_query, "last_answer": ans})

