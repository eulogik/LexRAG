import streamlit as st
import sys
import os
import uuid
from datetime import datetime
import importlib

# Add root directory to path to ensure proper imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Force reload of api.rag_engine to avoid caching issues with session_id arg
import api.rag_engine as rag_engine
importlib.reload(rag_engine)
from api.rag_engine import query_rag
from api.memory import list_sessions, get_history, delete_session

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="LexRAG | By Evolucent AI", 
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GENIUS 2.2 CSS (AkashML + OpenClaw Style) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono&display=swap');
    
    :root {
        --bg-color: #000000;
        --accent-color: #ffffff;
        --secondary-bg: #0A0A0A;
        --border-color: #1A1A1A;
        --text-muted: #888888;
        --bubble-user: #111111;
        --bubble-ai: #000000;
        --highlight: #ffffff;
    }
    
    .stApp {
        background-color: var(--bg-color);
        color: var(--accent-color);
        font-family: 'Outfit', sans-serif;
    }
    
    /* Clean Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--bg-color);
        border-right: 1px solid var(--border-color);
    }
    
    /* Typography */
    h1, h2, h3 {
        font-weight: 700;
        letter-spacing: -0.04em;
        color: var(--accent-color);
    }
    
    .stMarkdown p {
        font-weight: 300;
        line-height: 1.7;
    }

    /* Intuitive Chat Bubbles */
    .chat-row {
        display: flex;
        flex-direction: column;
        margin-bottom: 2rem;
        width: 100%;
    }
    
    .user-bubble {
        align-self: flex-end;
        background-color: var(--bubble-user);
        border: 1px solid var(--border-color);
        padding: 1.2rem 1.8rem;
        border-radius: 24px 24px 4px 24px;
        max-width: 80%;
        font-size: 1.05rem;
        color: #ffffff;
    }
    
    .ai-bubble {
        align-self: flex-start;
        background-color: var(--bubble-ai);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 1.2rem 0;
        max-width: 100%;
        font-size: 1.1rem;
        color: #e0e0e0;
    }

    /* Buttons & Inputs */
    .stButton>button {
        background-color: transparent;
        color: var(--accent-color);
        border: 1px solid var(--border-color);
        border-radius: 50px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
    }
    
    .stButton>button:hover {
        background-color: #ffffff;
        color: #000000;
        border-color: #ffffff;
        transform: translateY(-1px);
    }
    
    /* Hide Default Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Mono Labels */
    .mono-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }

    /* Source Box */
    .source-container {
        margin-top: 1rem;
        padding: 1rem;
        border-left: 2px solid var(--border-color);
        background: #050505;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR: MANAGEMENT ---
with st.sidebar:
    st.markdown('<div class="mono-label">Project</div>', unsafe_allow_html=True)
    st.markdown("## LexRAG")
    st.caption("By Evolucent AI")
    
    st.markdown('<div class="mono-label" style="margin-top:2rem">Control</div>', unsafe_allow_html=True)
    if st.button("New Instance", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
        
    st.markdown('<div class="mono-label" style="margin-top:2rem">Archives</div>', unsafe_allow_html=True)
    sessions = list_sessions()
    if sessions:
        for s in sessions[:5]:
            cols = st.columns([5, 1])
            if cols[0].button(f"S-{s['session_id'][:4].upper()}", key=f"sess_{s['session_id']}", use_container_width=True):
                st.session_state.session_id = s['session_id']
                st.session_state.messages = get_history(s['session_id'])
                st.rerun()
            if cols[1].button("×", key=f"del_{s['session_id']}"):
                delete_session(s['session_id'])
                st.rerun()
    
    st.markdown('<div class="mono-label" style="margin-top:2rem">Engine</div>', unsafe_allow_html=True)
    provider = st.selectbox("Inference", ["openrouter", "groq", "ollama"], label_visibility="collapsed")
    jurisdiction = st.selectbox("Jurisdiction Focus", ["Both", "UAE", "India"])
    source_count = st.slider("Context Depth", 3, 10, 5)

# --- MAIN CHAT ---
st.markdown('<div class="mono-label">Strategic Intelligence Interface</div>', unsafe_allow_html=True)
st.markdown("# Intelligence Feed")

# Display Messages
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-row"><div class="user-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-row"><div class="ai-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
        # Handle sources if they were stored (not currently in memory, but show toggle for new ones)
        if "sources" in msg:
            with st.expander("Show Evidence"):
                for s in msg["sources"]:
                    st.caption(f"📄 {s['title']} | Score: {s['score']}")

# Query Input
query = st.chat_input("Inquire with legal or accounting precision...")

if query:
    # Append User
    st.session_state.messages.append({"role": "user", "content": query})
    st.markdown(f'<div class="chat-row"><div class="user-bubble">{query}</div></div>', unsafe_allow_html=True)
    
    with st.spinner("Decoding intelligence..."):
        try:
            result = query_rag(
                question=query, 
                jurisdiction=jurisdiction, 
                top_k=source_count, 
                provider=provider,
                session_id=st.session_state.session_id
            )
            
            # Handle Errors (like 429) inline
            if "Model error" in result["answer"] and "429" in result["answer"]:
                st.warning("⚠️ OpenRouter rate limit reached. Suggestion: Switch 'Inference' to Groq or Ollama in the sidebar.")
            
            # Append AI
            # We store sources in the message object for this session only (not in SQLite history yet for content, but we'll reflect it here)
            ai_msg = {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
            st.session_state.messages.append(ai_msg)
            
            st.markdown(f'<div class="chat-row"><div class="ai-bubble">{result["answer"]}</div></div>', unsafe_allow_html=True)
            
            # Evidence Toggle
            if result["sources"]:
                with st.expander("Show Evidence Matrix"):
                    for s in result["sources"]:
                        st.markdown(f"""
                        <div class="source-container">
                            <b>{s['title']}</b><br>
                            Jurisdiction: {s['jurisdiction']} | Confidence: {s['score']}<br>
                            {f'<a href="{s["url"]}" style="color:#888; font-size:0.7rem">View Original Registry →</a>' if s.get('url') else ''}
                        </div>
                        """, unsafe_allow_html=True)
                        
        except Exception as e:
            st.error(f"Engine Fault: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.markdown('<p style="font-size:0.6rem; color:#444">VER: 2.2 SVRN INTELLIGENCE<br>© EVOLUCENT AI</p>', unsafe_allow_html=True)