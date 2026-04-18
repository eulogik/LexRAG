import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.rag_engine import query_rag

st.set_page_config(page_title="LexRAG — UAE & India Legal AI", page_icon="⚖️", layout="wide")
st.title("⚖️ LexRAG — Legal & Accounting AI")
st.caption("UAE and Indian Law | Taxation | Case Law | Accounting Standards")

with st.sidebar:
    st.header("Model Settings")
    provider = st.selectbox(
        "LLM Provider", 
        ["openrouter", "groq", "ollama"],
        index=0,
        help="Select which LLM to use. OpenRouter is default. Groq is fast. Ollama is local."
    )
    st.markdown("---")
    st.header("Search Options")
    jurisdiction = st.selectbox("Jurisdiction", ["Both", "UAE", "India"])
    source_type = st.selectbox("Document Type", ["All", "statute", "case", "ruling"])
    top_k = st.slider("Documents to retrieve", 3, 10, 5)
    st.markdown("---")
    st.markdown("**Data Sources**")
    st.markdown("- UAE FTA: [tax.gov.ae](https://www.tax.gov.ae)")
    st.markdown("- UAE Laws: [uaelegislation.gov.ae](https://uaelegislation.gov.ae)")
    st.markdown("- India GST: [cbic-gst.gov.in](https://cbic-gst.gov.in)")
    st.markdown("- Case Law: [indiankanoon.org](https://indiankanoon.org)")

question = st.text_area("Ask a legal or accounting question", height=100,
    placeholder="e.g. What are the VAT exemptions for healthcare in UAE? / What is the penalty for late GST filing in India?")

if st.button("Ask LexRAG", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching knowledge base and generating answer..."):
            result = query_rag(question, jurisdiction, source_type if source_type != "All" else None, top_k, provider)
        
        st.subheader("Answer")
        st.markdown(result["answer"])
        
        if result["sources"]:
            st.subheader(f"Sources Used ({result['context_used']} documents)")
            for s in result["sources"]:
                with st.expander(f"📄 {s['title'][:80]} | {s['jurisdiction']} | Score: {s['score']}"):
                    st.write(f"**Source:** {s['source']}")
                    st.write(f"**Type:** {s['type']}")
                    if s["url"]:
                        st.write(f"**URL:** {s['url']}")