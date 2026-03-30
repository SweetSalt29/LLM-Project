import streamlit as st
import requests
import time

# ========================
# BACKGROUND STYLE
# ========================
def set_dynamic_background():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(
                270deg,
                #ff7f00,
                #001f3f,
                #ff7f00,
                #001f3f
            );
            background-size: 400% 400%;
            animation: gradientFlow 12s ease infinite;
        }

        @keyframes gradientFlow {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* TEXT */
        h1, h2, h3, h4, h5, h6, p, div, label {
            color: white !important;
        }

        /* BUTTONS */
        .stButton > button {
            background-color: white !important;
            color: black !important;
            border-radius: 10px;
            border: none;
        }
        .stButton > button * {
            color: black !important;
        }

        /* INPUT */
        .stTextInput input {
            background-color: white !important;
            color: black !important;
            caret-color: black !important;
        }
        textarea {
            color: black !important;
            caret-color: black !important;
        }
        input::placeholder {
            color: #555 !important;
        }

        /* CHAT MESSAGES */
        .user-bubble {
            background-color: rgba(255,255,255,0.15);
            border-radius: 12px;
            padding: 10px 14px;
            margin: 6px 0;
            text-align: right;
        }
        .assistant-bubble {
            background-color: rgba(0,0,0,0.25);
            border-radius: 12px;
            padding: 10px 14px;
            margin: 6px 0;
            text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


BASE_URL = "http://127.0.0.1:8000"


# ========================
# SESSION STATE INIT
# ========================
def init_state():
    defaults = {
        "token": None,
        "page": "login",
        "doc_status": None,
        # Current active chat session
        "session_id": None,
        # In-memory message list for current chat: [{"role": ..., "content": ...}]
        "messages": [],
        # Which mode sidebar is showing
        "sidebar_mode": "rag",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()


# ========================
# API HELPERS
# ========================
def api_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}

def register(name, password):
    return requests.post(f"{BASE_URL}/auth/register", json={"name": name, "password": password})

def login(username, password):
    return requests.post(f"{BASE_URL}/auth/login", data={"username": username, "password": password})

def upload_file(file, token):
    return requests.post(
        f"{BASE_URL}/upload/",
        files={"files": (file.name, file.read())},
        headers={"Authorization": f"Bearer {token}"}
    )

def get_upload_status():
    return requests.get(f"{BASE_URL}/upload/status", headers=api_headers())

def chat_api(query: str, session_id: str, mode: str):
    return requests.post(
        f"{BASE_URL}/query/chat",
        json={"query": query, "session_id": session_id, "mode": mode},
        headers=api_headers()
    )

def get_sessions(mode: str):
    return requests.get(f"{BASE_URL}/query/sessions?mode={mode}", headers=api_headers())

def get_session_history(session_id: str):
    return requests.get(f"{BASE_URL}/query/sessions/{session_id}", headers=api_headers())

def summarize_chat(session_id: str, mode: str):
    return requests.post(
        f"{BASE_URL}/query/summarize",
        json={"session_id": session_id, "mode": mode},
        headers=api_headers()
    )


# ========================
# SIDEBAR — chat history
# ========================
def render_sidebar(mode: str):
    # Store which session was clicked — checked AFTER sidebar renders
    if "selected_session" not in st.session_state:
        st.session_state.selected_session = None

    with st.sidebar:
        st.markdown(f"### 💬 {'Document' if mode == 'rag' else 'Data'} Chats")

        if st.button("➕ New Chat", key="new_chat_btn"):
            st.session_state.session_id = None
            st.session_state.messages = []
            st.session_state.selected_session = None
            st.rerun()

        st.divider()

        res = get_sessions(mode)
        if res.status_code == 200:
            sessions = res.json().get("sessions", [])
            if not sessions:
                st.caption("No past chats yet.")
            for s in sessions:
                label = s["title"] or "Untitled Chat"
                created = s["created_at"][:10]  # YYYY-MM-DD
                # Simple flat label — no newlines or markdown inside button
                btn_label = f"📄 {label[:28]} ({created})"
                is_active = st.session_state.session_id == s["session_id"]

                if st.button(
                    btn_label,
                    key=f"sess_{s['session_id']}",
                    type="primary" if is_active else "secondary"
                ):
                    # Just store the clicked session_id — load history outside sidebar
                    st.session_state.selected_session = s["session_id"]
                    st.rerun()
        else:
            st.caption("Could not load sessions.")

        st.divider()
        if st.button("⬅ Home", key="sidebar_home"):
            st.session_state.page = "home"
            st.session_state.session_id = None
            st.session_state.messages = []
            st.session_state.selected_session = None
            st.rerun()

    # ---- Load selected session OUTSIDE sidebar, after rerun ----
    if st.session_state.selected_session:
        target = st.session_state.selected_session
        # Only reload if it's a different session than current
        if target != st.session_state.session_id:
            hist_res = get_session_history(target)
            if hist_res.status_code == 200:
                st.session_state.session_id = target
                st.session_state.messages = hist_res.json().get("messages", [])
        # Clear the trigger so it doesn't reload on every rerun
        st.session_state.selected_session = None


# ========================
# CHAT MESSAGE RENDERER
# ========================
def render_messages(messages: list, mode: str):
    for msg in messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">🧑 {msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            content = msg["content"]
            # For NL2SQL assistant messages, content is a plain string (natural_answer)
            st.markdown(
                f'<div class="assistant-bubble">🤖 {content}</div>',
                unsafe_allow_html=True
            )


# ========================
# STATUS BANNER
# ========================
def show_status_banner():
    s = st.session_state.doc_status

    if s == "processing":
        st.warning("⏳ Document is still being processed. Please wait before querying.")
        with st.spinner("Processing..."):
            time.sleep(3)
            res = get_upload_status()
            if res.status_code == 200:
                st.session_state.doc_status = res.json().get("status")
                st.rerun()

    elif s == "ready":
        st.success("✅ Document ready. You can now chat.")

    elif s and s.startswith("failed"):
        st.error(f"❌ Ingestion failed: {s}. Please re-upload your file.")


# ========================
# AUTH PAGE
# ========================
def auth_page():
    st.title("🔐 Welcome")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Login")
        name = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            if not name or not password:
                st.warning("Please enter username and password")
            else:
                try:
                    res = login(name, password)
                    if res.status_code == 200:
                        st.session_state.token = res.json()["access_token"]
                        st.session_state.page = "home"
                        st.rerun()
                    else:
                        st.error(res.json().get("detail", "Invalid credentials"))
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Is it running?")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")

    with tab2:
        st.subheader("Register")
        name = st.text_input("Username", key="reg_user")
        password = st.text_input("Password", type="password", key="reg_pass")

        if st.button("Register"):
            if not name or not password:
                st.warning("Fill all fields")
            else:
                res = register(name, password)
                if res.status_code == 200:
                    st.success("User registered")
                else:
                    st.error(res.text)


# ========================
# HOME PAGE
# ========================
def home_page():
    st.title("📊 Ask Your Files")
    st.markdown("""
Welcome! This application helps you **understand and analyze your files easily**.

### What you can do:
- 📄 **Ask About Your Document** — Upload documents and chat to explore content.
- 📊 **Analyze Your Data** — Upload CSV/Excel and get insights through conversation.
""")
    st.divider()

    tab1, tab2 = st.tabs(["📄 Ask About Your Document", "📊 Analyze Your Data"])

    with tab1:
        st.subheader("Document Intelligence")
        if st.button("Go to Document Chat"):
            st.session_state.page = "rag"
            st.session_state.sidebar_mode = "rag"
            st.session_state.session_id = None
            st.session_state.messages = []
            st.rerun()

    with tab2:
        st.subheader("Data Analysis")
        if st.button("Go to Data Analysis"):
            st.session_state.page = "sql"
            st.session_state.sidebar_mode = "sql"
            st.session_state.session_id = None
            st.session_state.messages = []
            st.rerun()

    st.divider()
    if st.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ========================
# RAG CHAT PAGE
# ========================
def rag_page():
    render_sidebar("rag")
    st.title("📄 Document Chat")

    # ---- Upload section ----
    with st.expander("📁 Upload Document", expanded=(st.session_state.doc_status is None)):
        st.caption("Accepted: PDF, DOC, DOCX, TXT")
        file = st.file_uploader(
            "Upload document file",
            type=["pdf", "doc", "docx", "txt"],
            key="rag_uploader"
        )
        if st.button("Upload", key="rag_upload_btn"):
            if file:
                res = upload_file(file, st.session_state.token)
                if res.status_code == 202:
                    data = res.json()
                    # RAG files need embedding — mark as processing
                    st.session_state.doc_status = "processing"
                    st.rerun()
                else:
                    err = res.json().get("detail", res.text)
                    st.error(f"❌ {err}")
            else:
                st.warning("Please select a file first")

    show_status_banner()

    if st.session_state.doc_status != "ready":
        return

    st.divider()

    # ---- Summarize button ----
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("📋 Summarize Chat"):
            if not st.session_state.session_id:
                st.warning("Start a conversation first.")
            else:
                with st.spinner("Summarizing..."):
                    res = summarize_chat(st.session_state.session_id, "rag")
                    if res.status_code == 200:
                        summary = res.json()["summary"]
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"**📋 Chat Summary:**\n\n{summary}"
                        })
                        st.rerun()
                    else:
                        st.error("Could not summarize.")

    # ---- Message history ----
    render_messages(st.session_state.messages, "rag")

    # ---- Chat input ----
    query = st.chat_input("Ask a question about your document...")
    if query:
        # Optimistically show user message
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Thinking..."):
            res = chat_api(query, st.session_state.session_id, "rag")

        if res.status_code == 200:
            data = res.json()
            result = data["response"]
            session_id = result.get("session_id")
            answer = result.get("answer", "")
            sources = result.get("sources", [])

            # Update session_id (set on first message)
            st.session_state.session_id = session_id

            # Format sources — one compact line, deduplicated by pipeline
            source_text = ""
            if sources:
                # Build compact labels: "file.pdf p.3" or just "file.pdf" if page is None
                labels = []
                for s in sources:
                    if not s.get("source"):
                        continue
                    label = s["source"]
                    if s.get("page") is not None:
                        label += f" (p.{s['page']})"
                    labels.append(label)
                if labels:
                    source_text = "\n\n📄 _Source: " + " · ".join(labels) + "_"

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer + source_text
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Error: {res.json().get('detail', 'Something went wrong.')}"
            })

        st.rerun()


# ========================
# NL2SQL CHAT PAGE
# ========================
def sql_page():
    render_sidebar("sql")
    st.title("📊 Data Analysis Chat")

    # ---- Upload section ----
    with st.expander("📁 Upload Data File", expanded=(st.session_state.doc_status is None)):
        st.caption("Accepted: CSV, XLSX, XLS, DB, SQL")
        file = st.file_uploader(
            "Upload data file",
            type=["csv", "xlsx", "xls", "db", "sql"],
            key="sql_uploader"
        )
        if st.button("Upload", key="sql_upload_btn"):
            if file:
                res = upload_file(file, st.session_state.token)
                if res.status_code == 202:
                    data = res.json()
                    # SQL files are ready immediately — no background embedding
                    pipeline = data.get("pipeline", "sql")
                    st.session_state.doc_status = "processing" if pipeline == "rag" else "ready"
                    st.rerun()
                else:
                    err = res.json().get("detail", res.text)
                    st.error(f"❌ {err}")
            else:
                st.warning("Please select a file first")

    show_status_banner()

    if st.session_state.doc_status != "ready":
        return

    st.divider()

    # ---- Summarize button ----
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("📋 Summarize Chat"):
            if not st.session_state.session_id:
                st.warning("Start a conversation first.")
            else:
                with st.spinner("Summarizing..."):
                    res = summarize_chat(st.session_state.session_id, "sql")
                    if res.status_code == 200:
                        summary = res.json()["summary"]
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"**📋 Chat Summary:**\n\n{summary}"
                        })
                        st.rerun()
                    else:
                        st.error("Could not summarize.")

    # ---- Message history ----
    # For SQL, assistant messages may have structured data — render specially
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">🧑 {msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            content = msg["content"]
            # Structured NL2SQL result stored as dict
            if isinstance(content, dict):
                if content.get("error"):
                    st.error(f"❌ {content['error']}")
                else:
                    st.markdown(
                        f'<div class="assistant-bubble">🤖 {content.get("natural_answer", "")}</div>',
                        unsafe_allow_html=True
                    )
                    with st.expander("🔍 SQL Query"):
                        st.code(content.get("sql_query", ""), language="sql")
                    with st.expander("📋 Data Result"):
                        st.markdown(content.get("result_markdown", "_No results._"))
            else:
                st.markdown(
                    f'<div class="assistant-bubble">🤖 {content}</div>',
                    unsafe_allow_html=True
                )

    # ---- Chat input ----
    query = st.chat_input("Ask a question about your data...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Analyzing..."):
            res = chat_api(query, st.session_state.session_id, "sql")

        if res.status_code == 200:
            data = res.json()
            result = data["response"]
            session_id = result.get("session_id")
            st.session_state.session_id = session_id

            # Store full structured result as dict for rich rendering
            st.session_state.messages.append({
                "role": "assistant",
                "content": result  # dict with sql_query, natural_answer, result_markdown
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Error: {res.json().get('detail', 'Something went wrong.')}"
            })

        st.rerun()


# ========================
# ROUTER
# ========================
set_dynamic_background()

if not st.session_state.token:
    auth_page()
else:
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "rag":
        rag_page()
    elif st.session_state.page == "sql":
        sql_page()