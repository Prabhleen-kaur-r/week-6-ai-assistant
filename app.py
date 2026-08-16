"""Streamlit frontend application - AI Knowledge Base Assistant."""

import streamlit as st
import requests
import json
import os
from datetime import datetime

st.set_page_config(
    page_title="AI Knowledge Base Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external CSS (library theme)
try:
    with open("assets/styles.css", "r") as f:
        css_content = f.read()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("styles.css not found. Using default styling.")

# Session state (no top_k/reranking vars needed)
if "page" not in st.session_state:
    st.session_state.page = "Chat"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("API_KEY", "")
if "backend_url" not in st.session_state:
    st.session_state.backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = {"documents": [], "stats": {"total_documents": 0, "total_chunks": 0, "sources": []}}
if "selected_files" not in st.session_state:
    st.session_state.selected_files = []


def refresh_knowledge_base():
    try:
        headers = {"X-API-Key": st.session_state.api_key}
        docs_response = requests.get(
            f"{st.session_state.backend_url}/api/documents",
            headers=headers,
            timeout=10
        )
        if docs_response.status_code == 200:
            st.session_state.knowledge_base["documents"] = docs_response.json()
        stats_response = requests.get(
            f"{st.session_state.backend_url}/api/documents/stats",
            headers=headers,
            timeout=10
        )
        if stats_response.status_code == 200:
            st.session_state.knowledge_base["stats"] = stats_response.json()
        return True
    except Exception as e:
        st.error(f"Failed to refresh knowledge base: {str(e)}")
        return False


def get_backend_status():
    try:
        response = requests.get(f"{st.session_state.backend_url}/health", timeout=3)
        return response.status_code == 200
    except:
        return False


# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ API Configuration")
    backend_url = st.text_input("Backend URL", value=st.session_state.backend_url, key="backend_url_input")
    api_key = st.text_input("API Key", value=st.session_state.api_key, type="password", key="api_key_input")
    if st.button("Save", use_container_width=True):
        st.session_state.backend_url = st.session_state.backend_url_input
        st.session_state.api_key = st.session_state.api_key_input
        if get_backend_status():
            st.success("✅ Connected")
            refresh_knowledge_base()
        else:
            st.error("❌ Cannot reach backend")
    if get_backend_status():
        st.markdown("🟢 **Connected**")
    else:
        st.markdown("🔴 **Disconnected**")
    st.markdown("---")
    st.caption(f"v1.0.0 • {datetime.now().strftime('%H:%M')}")

# Header
st.markdown("""
<div class="app-header">
    <div>
        <div class="app-title">📚 AI <span>Knowledge Base</span></div>
        <div class="app-subtitle">Intelligent document Q&A with RAG</div>
    </div>
    <div style="display:flex; gap:0.5rem; align-items:center;">
        <span style="color:#94a3b8; font-size:0.75rem;">System</span>
        <span style="color:#4ade80; font-size:0.75rem;">● Online</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation
st.markdown('<div class="nav-container">', unsafe_allow_html=True)
nav_cols = st.columns(4)
pages = ["💬 Chat", "📄 Knowledge Base", "📊 RAG Evaluation", "⚙️ System Status"]
for idx, page in enumerate(pages):
    with nav_cols[idx]:
        if st.button(page, use_container_width=True, type="primary" if st.session_state.page == page else "secondary"):
            st.session_state.page = page
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================
# PAGE: CHAT (no settings expander)
# ============================================================
if st.session_state.page == "💬 Chat":
    st.markdown("### 💬 Chat")
    st.caption("Ask questions about your uploaded documents.")
    stats = st.session_state.knowledge_base.get("stats", {})
    if stats.get("total_documents", 0) > 0:
        st.caption(f"📚 {stats['total_documents']} documents available")
    
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">🤖</div>
                <div style="font-size:0.95rem; color:#64748b;">No messages yet</div>
                <div style="font-size:0.8rem; color:#94a3b8;">Upload documents in Knowledge Base, then ask questions.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-message-user"><strong>🧑 You</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    conf_class = f'confidence-{msg.get("confidence", "medium")}'
                    conf_text = msg.get("confidence", "medium").upper()
                    sources_html = ""
                    if msg.get("sources"):
                        source_items = "".join([f'<span class="source-item source-item-pdf">📄 {s}</span>' for s in msg["sources"][:5]])
                        sources_html = f'<div class="sources-container"><strong>Sources:</strong> {source_items}</div>'
                    key_points_html = ""
                    if msg.get("key_points"):
                        points = "".join([f'<li style="font-size:0.8rem;">{p}</li>' for p in msg["key_points"][:3]])
                        key_points_html = f'<div style="margin-top:0.2rem;"><strong>Key Points:</strong><ul style="margin:0.1rem 0; padding-left:1rem;">{points}</ul></div>'
                    st.markdown(f"""
                    <div class="chat-message-assistant">
                        <strong>🤖 Assistant</strong>
                        <div style="margin-top:0.2rem;">{msg["content"]}</div>
                        <div style="margin-top:0.3rem; display:flex; gap:0.5rem; align-items:center; flex-wrap:wrap;">
                            <span class="{conf_class}">{conf_text}</span>
                        </div>
                        {key_points_html}
                        {sources_html}
                    </div>
                    """, unsafe_allow_html=True)
    
    prompt = st.chat_input("Ask a question...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            try:
                headers = {"X-API-Key": st.session_state.api_key}
                data = {
                    "message": prompt,
                    "session_id": st.session_state.session_id,
                    "stream": True,
                    # Use default values from backend config (no user controls)
                    "use_reranking": True,      # or False – adjust as needed
                    "use_query_transformation": False,
                    "top_k": 5                  # default – adjust as needed
                }
                response = requests.post(
                    f"{st.session_state.backend_url}/api/chat/stream",
                    json=data,
                    headers=headers,
                    stream=True,
                    timeout=90
                )
                if response.status_code == 200:
                    full_answer = ""
                    sources = []
                    confidence = "medium"
                    key_points = []
                    for line in response.iter_lines():
                        if line:
                            try:
                                chunk_data = json.loads(line.decode('utf-8'))
                                if "error" in chunk_data:
                                    st.error(f"Error: {chunk_data['error']}")
                                    break
                                if "chunk" in chunk_data:
                                    full_answer += chunk_data["chunk"]
                                if chunk_data.get("complete"):
                                    sources = chunk_data.get("sources", [])
                                    confidence = chunk_data.get("confidence", "medium")
                                    key_points = chunk_data.get("key_points", [])
                                    if chunk_data.get("session_id"):
                                        st.session_state.session_id = chunk_data["session_id"]
                            except json.JSONDecodeError:
                                continue
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_answer if full_answer else "I couldn't generate a response.",
                        "sources": sources,
                        "confidence": confidence,
                        "key_points": key_points
                    })
                    st.rerun()
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    if st.button("🗑️ Clear Chat", use_container_width=False):
        st.session_state.messages = []
        st.rerun()

# ============================================================
# PAGE: KNOWLEDGE BASE
# ============================================================
elif st.session_state.page == "📄 Knowledge Base":
    st.markdown("### 📄 Knowledge Base")
    st.caption("Upload and manage your documents.")
    if not st.session_state.knowledge_base.get("documents"):
        refresh_knowledge_base()
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("#### Upload Documents")
        uploaded_files = st.file_uploader(
            "Select PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if uploaded_files is not None and len(uploaded_files) > 0:
            st.session_state.selected_files = uploaded_files
        if st.session_state.selected_files:
            st.write(f"**{len(st.session_state.selected_files)} files selected:**")
            for f in st.session_state.selected_files:
                st.markdown(f'<span class="selected-file">📎 {f.name} ({f.size / 1024:.1f} KB)</span>', unsafe_allow_html=True)
        if st.button("Process Documents", use_container_width=True, type="primary"):
            if not st.session_state.selected_files:
                st.warning("Please select files first.")
            else:
                status_placeholder = st.empty()
                progress_bar = st.progress(0)
                errors = []
                success_count = 0
                for idx, file in enumerate(st.session_state.selected_files):
                    status_placeholder.info(f"⏳ Processing: {file.name}...")
                    try:
                        files = {"file": (file.name, file.getvalue())}
                        headers = {"X-API-Key": st.session_state.api_key}
                        response = requests.post(
                            f"{st.session_state.backend_url}/api/documents/upload",
                            files=files,
                            headers=headers,
                            timeout=120
                        )
                        if response.status_code == 200:
                            success_count += 1
                            status_placeholder.success(f"✅ {file.name} processed")
                        else:
                            error_msg = response.json().get("detail", {}).get("message", "Unknown error")
                            errors.append(f"{file.name}: {error_msg}")
                            status_placeholder.error(f"❌ {file.name}: {error_msg}")
                    except Exception as e:
                        errors.append(f"{file.name}: {str(e)}")
                        status_placeholder.error(f"❌ {file.name}: {str(e)}")
                    progress_bar.progress((idx + 1) / len(st.session_state.selected_files))
                status_placeholder.empty()
                if errors:
                    st.markdown(f"""
                    <div class="processing-status error">
                        ⚠️ {success_count} of {len(st.session_state.selected_files)} processed successfully.<br>
                        <strong>Failed:</strong> {', '.join([e.split(':')[0] for e in errors[:3]])}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="processing-status success">
                        ✅ All {len(st.session_state.selected_files)} documents processed successfully!
                    </div>
                    """, unsafe_allow_html=True)
                st.session_state.selected_files = []
                refresh_knowledge_base()
                st.rerun()
        st.markdown("#### Processed Documents")
        docs = st.session_state.knowledge_base.get("documents", [])
        if docs:
            for doc in docs:
                st.markdown(f'<div class="doc-item"><span class="doc-name">📄 {doc}</span><span class="doc-status">✅ Processed</span></div>', unsafe_allow_html=True)
        else:
            st.info("No documents uploaded yet.")
    with col2:
        st.markdown("#### Statistics")
        stats = st.session_state.knowledge_base.get("stats", {})
        st.metric("📄 Documents", stats.get("total_documents", 0))
        st.metric("📝 Chunks", stats.get("total_chunks", 0))
        st.metric("📁 Sources", len(stats.get("sources", [])))
        if stats.get("sources"):
            st.markdown("**Sources:**")
            for s in stats["sources"][:5]:
                st.markdown(f"• {s}")
        st.markdown("---")
        st.markdown("#### Danger Zone")
        if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
            try:
                headers = {"X-API-Key": st.session_state.api_key}
                response = requests.delete(
                    f"{st.session_state.backend_url}/api/documents",
                    headers=headers,
                    timeout=30
                )
                if response.status_code == 200:
                    st.session_state.knowledge_base = {"documents": [], "stats": {"total_documents": 0, "total_chunks": 0, "sources": []}}
                    st.session_state.selected_files = []
                    st.success("✅ Knowledge base cleared")
                    st.rerun()
                else:
                    st.error("Failed to clear knowledge base")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ============================================================
# PAGE: RAG EVALUATION (no cluster controls, just stats)
# ============================================================
elif st.session_state.page == "📊 RAG Evaluation":
    st.markdown("### 📊 RAG Evaluation")
    st.caption("Monitor knowledge‑base health and retrieval readiness.")

    if st.button("🔄 Refresh Stats", use_container_width=False, type="primary"):
        refresh_knowledge_base()
        st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📄 Document Overview")
        stats = st.session_state.knowledge_base.get("stats", {})
        st.metric("Total Documents", stats.get("total_documents", 0))
        st.metric("Total Chunks", stats.get("total_chunks", 0))
        st.metric("Unique Sources", len(stats.get("sources", [])))
        if stats.get("sources"):
            st.markdown("**Sources:**")
            for s in stats["sources"]:
                st.markdown(f"• {s}")

    with col2:
        st.markdown("#### 📊 Retrieval Readiness")
        total_chunks = stats.get("total_chunks", 0)
        if total_chunks == 0:
            readiness = "🔴 No documents – upload some first"
            progress = 0
        elif total_chunks < 10:
            readiness = "🟡 Low – add more documents for better answers"
            progress = 30
        elif total_chunks < 50:
            readiness = "🟢 Good – ready for questions"
            progress = 70
        else:
            readiness = "✅ Excellent – high coverage"
            progress = 100
        st.markdown(f"**Readiness:** {readiness}")
        st.progress(progress / 100)
        try:
            response = requests.get(f"{st.session_state.backend_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                st.markdown(f"**LLM Model:** {data.get('model', 'Unknown')}")
                st.markdown(f"**Embedding Model:** {data.get('embedding_model', 'Unknown')}")
                st.markdown(f"**Backend Status:** {data.get('status', 'unknown').upper()}")
        except:
            st.markdown("**Backend Status:** 🔴 OFFLINE")

    st.markdown("---")
    st.markdown("#### 📁 Processed Documents")
    docs = st.session_state.knowledge_base.get("documents", [])
    if docs:
        for doc in docs:
            st.markdown(f'<div class="doc-item"><span class="doc-name">📄 {doc}</span><span class="doc-status">✅ Indexed</span></div>', unsafe_allow_html=True)
    else:
        st.info("No documents uploaded yet. Go to **Knowledge Base** to upload.")
    st.markdown("---")
    st.caption("The retrieval settings (top_k, reranking) are managed by the backend defaults.")

# ============================================================
# PAGE: SYSTEM STATUS
# ============================================================
elif st.session_state.page == "⚙️ System Status":
    st.markdown("### ⚙️ System Status")
    st.caption("Current system health and configuration.")
    try:
        response = requests.get(f"{st.session_state.backend_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            st.markdown(f"🟢 **Status:** {data.get('status', 'unknown').upper()}")
            st.markdown(f"**LLM Model:** {data.get('model', 'Unknown')}")
            st.markdown(f"**Embedding Model:** {data.get('embedding_model', 'Unknown')}")
        else:
            st.markdown("🔴 **Status:** OFFLINE")
    except:
        st.markdown("🔴 **Status:** OFFLINE")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📊 Knowledge Base")
        stats = st.session_state.knowledge_base.get("stats", {})
        st.metric("Documents", stats.get("total_documents", 0))
        st.metric("Chunks", stats.get("total_chunks", 0))
        st.metric("Sources", len(stats.get("sources", [])))
    with col2:
        st.markdown("#### 🤖 Models")
        try:
            response = requests.get(f"{st.session_state.backend_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                st.metric("LLM", data.get("model", "Not configured"))
                st.metric("Embedding", data.get("embedding_model", "Not configured"))
                st.metric("Version", data.get("version", "Unknown"))
        except:
            st.metric("LLM", "Unknown")
            st.metric("Embedding", "Unknown")
            st.metric("Version", "Unknown")
    with col3:
        st.markdown("#### 🛠️ Services")
        st.markdown("✅ FastAPI")
        st.markdown("✅ FAISS Vector Store")
        st.markdown("✅ BM25 Index")
        st.markdown("✅ Gemini API")
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")