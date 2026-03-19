"""
DocRAG — Chat With Your Document
Clean, minimal UI. No API config visible to user.
All keys loaded from .env automatically.
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocRAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Reset & base */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    font-family: 'Sora', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
}

[data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* Main container */
.main .block-container {
    max-width: 860px !important;
    padding: 0 !important;
    margin: 0 auto !important;
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 52px 24px 32px;
    position: relative;
}

.hero-icon {
    width: 56px; height: 56px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 16px;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 26px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(99,102,241,0.4);
}

.hero h1 {
    font-size: 2.4rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.04em;
    line-height: 1.1;
    margin-bottom: 10px;
}

.hero h1 span {
    background: linear-gradient(135deg, #6366f1, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    font-size: 0.95rem;
    color: #64748b;
    font-weight: 400;
    letter-spacing: 0.01em;
}

/* ── Upload zone ── */
.upload-zone {
    margin: 0 24px 28px;
    border: 1.5px dashed #1e293b;
    border-radius: 16px;
    padding: 32px 24px;
    text-align: center;
    background: #0f1117;
    transition: border-color 0.2s;
}

.upload-zone:hover { border-color: #6366f1; }

.upload-label {
    font-size: 0.85rem;
    color: #475569;
    margin-top: 10px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── File pill ── */
.file-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 999px;
    padding: 6px 14px;
    font-size: 0.82rem;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
    margin: 0 24px 20px;
}

.file-pill .dot {
    width: 7px; height: 7px;
    background: #22c55e;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── Status banner ── */
.status-ready {
    margin: 0 24px 20px;
    background: #052e16;
    border: 1px solid #16a34a;
    border-radius: 12px;
    padding: 12px 18px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.85rem;
    color: #86efac;
}

.status-info {
    margin: 0 24px 20px;
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 12px 18px;
    font-size: 0.85rem;
    color: #475569;
}

/* ── Divider ── */
.section-divider {
    height: 1px;
    background: #1e293b;
    margin: 0 24px 24px;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 24px !important;
    margin-bottom: 20px !important;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) .stChatMessageContent {
    background: #1e1b4b !important;
    border: 1px solid #312e81 !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 12px 16px !important;
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) .stChatMessageContent {
    background: #0f1117 !important;
    border: 1px solid #1e293b !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 12px 16px !important;
}

/* ── Source expander ── */
[data-testid="stExpander"] {
    background: #0a0a0f !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    margin: 8px 24px 0 !important;
}

[data-testid="stExpander"] summary {
    color: #475569 !important;
    font-size: 0.78rem !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.src-card {
    background: #0f1117;
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.8rem;
    color: #94a3b8;
    line-height: 1.6;
}

.src-card .src-name {
    font-family: 'JetBrains Mono', monospace;
    color: #6366f1;
    font-size: 0.75rem;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #0f1117 !important;
    border: 1.5px solid #1e293b !important;
    border-radius: 16px !important;
    margin: 16px 24px !important;
    padding: 4px 8px !important;
    transition: border-color 0.2s !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}

[data-testid="stChatInput"] textarea {
    color: #e2e8f0 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.9rem !important;
    background: transparent !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #334155 !important;
}

/* ── Streamlit buttons ── */
.stButton button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 10px 20px !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
    box-shadow: 0 4px 16px rgba(99,102,241,0.3) !important;
}

.stButton button:hover { opacity: 0.88 !important; }

/* file uploader */
[data-testid="stFileUploader"] {
    background: transparent !important;
}

[data-testid="stFileUploader"] > div {
    background: #0f1117 !important;
    border: 1.5px dashed #1e293b !important;
    border-radius: 14px !important;
    padding: 28px !important;
}

[data-testid="stFileUploader"] label {
    color: #475569 !important;
    font-size: 0.85rem !important;
}

[data-testid="stFileDropzoneInstructions"] {
    color: #334155 !important;
}

/* progress bar */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
}

/* scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 2px; }

/* footer badge */
.footer-badge {
    text-align: center;
    padding: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #1e293b;
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {
    "messages":       [],
    "ingested_files": [],
    "rag":            None,
    "ingestor":       None,
    "ready":          False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Cached loaders (keys come from .env) ──────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_ingestor():
    from rag.ingestion import Ingestor
    return Ingestor(
        endee_host=os.getenv("ENDEE_HOST", "http://localhost:8080"),
        endee_api_key=os.getenv("ENDEE_API_KEY") or None,
    )

@st.cache_resource(show_spinner=False)
def build_rag():
    from rag.retriever import RAGChain
    return RAGChain(
        endee_host=os.getenv("ENDEE_HOST", "http://localhost:8080"),
        endee_api_key=os.getenv("ENDEE_API_KEY") or None,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    )


# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-icon">📄</div>
    <h1>Chat With Your <span>Document</span></h1>
    <p>Upload any PDF or text file and ask questions about it instantly.</p>
</div>
""", unsafe_allow_html=True)


# ── Upload section ─────────────────────────────────────────────────────────────
if not st.session_state.ready:
    uploaded = st.file_uploader(
        "Drop your file here",
        type=["pdf", "txt"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if uploaded:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            ingest_btn = st.button("✦ Analyse Document", use_container_width=True)

        if ingest_btn:
            with st.spinner(""):
                prog = st.progress(0, text="Reading document...")
                try:
                    ingestor = build_ingestor()
                    rag      = build_rag()
                    st.session_state.ingestor = ingestor
                    st.session_state.rag      = rag

                    prog.progress(30, text="Chunking text...")
                    suffix = Path(uploaded.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name

                    prog.progress(60, text="Embedding & indexing in Endee...")
                    ingestor.ingest_file(tmp_path, display_name=uploaded.name)
                    os.unlink(tmp_path)

                    st.session_state.ingested_files = [uploaded.name]
                    prog.progress(100, text="Done!")
                    st.session_state.ready = True
                    st.rerun()

                except Exception as e:
                    st.error(f"Something went wrong: {e}")

else:
    # ── Ready state: show file pill + chat ────────────────────────────────────
    fname = st.session_state.ingested_files[0] if st.session_state.ingested_files else "Document"

    st.markdown(f"""
    <div class="file-pill">
        <span class="dot"></span>
        {fname}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"🔍 {len(msg['sources'])} passages retrieved from Endee"):
                    for src in msg["sources"]:
                        st.markdown(
                            f'<div class="src-card">'
                            f'<div class="src-name">📄 {src["source"]} &nbsp;·&nbsp; chunk #{src["chunk_index"]}</div>'
                            f'{src["text"][:320]}{"..." if len(src["text"]) > 320 else ""}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # Upload another file
    with st.expander("📎 Upload a different document"):
        new_file = st.file_uploader(
            "Replace document",
            type=["pdf", "txt"],
            label_visibility="collapsed",
        )
        if new_file:
            if st.button("Switch to this document"):
                st.cache_resource.clear()
                st.session_state.messages       = []
                st.session_state.ingested_files = []
                st.session_state.ready          = False
                st.session_state.rag            = None
                st.session_state.ingestor       = None
                st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask anything about your document..."):
        if not st.session_state.rag:
            st.warning("Please re-ingest your document.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_text   = ""
                sources     = []

                try:
                    for token in st.session_state.rag.stream(prompt):
                        if isinstance(token, dict) and "__sources__" in token:
                            sources = token["__sources__"]
                        else:
                            full_text += token
                            placeholder.markdown(full_text + "▌")
                    placeholder.markdown(full_text)

                    if sources:
                        with st.expander(f"🔍 {len(sources)} passages retrieved from Endee"):
                            for src in sources:
                                st.markdown(
                                    f'<div class="src-card">'
                                    f'<div class="src-name">📄 {src["source"]} &nbsp;·&nbsp; chunk #{src["chunk_index"]}</div>'
                                    f'{src["text"][:320]}{"..." if len(src["text"]) > 320 else ""}'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                except Exception as e:
                    full_text = f"❌ Error: {e}"
                    placeholder.markdown(full_text)

            st.session_state.messages.append({
                "role":    "assistant",
                "content": full_text,
                "sources": sources,
            })

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-badge">
    ENDEE VECTOR DB &nbsp;·&nbsp; LANGCHAIN &nbsp;·&nbsp; HUGGINGFACE &nbsp;·&nbsp; GROQ
</div>
""", unsafe_allow_html=True)