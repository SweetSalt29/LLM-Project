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

        h1, h2, h3, h4, h5, h6, p, div, label {
            color: white !important;
        }

        .stButton > button {
            background-color: white !important;
            color: black !important;
            border-radius: 10px;
            border: none;
        }
        .stButton > button * {
            color: black !important;
        }

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

        .user-row {
            display: flex;
            justify-content: flex-end;
            margin: 6px 0;
        }
        .assistant-row {
            display: flex;
            justify-content: flex-start;
            margin: 6px 0;
        }
        .user-bubble {
            background-color: rgba(255,255,255,0.15);
            border-radius: 18px 18px 4px 18px;
            padding: 10px 14px;
            max-width: 75%;
            text-align: left;
        }
        .assistant-bubble {
            background-color: rgba(0,0,0,0.25);
            border-radius: 18px 18px 18px 4px;
            padding: 10px 14px;
            max-width: 75%;
            text-align: left;
        }
        .library-file {
            background-color: rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 6px 10px;
            margin: 4px 0;
            font-size: 0.85em;
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
        "token":            None,
        "page":             "login",
        "session_id":       None,
        "messages":         [],
        "sidebar_mode":     "rag",
        "selected_files":   [],
        "new_chat_open":    False,
        # Incrementing key forces Streamlit to remount the file uploader,
        # clearing it after a successful upload so files can't be re-submitted.
        "uploader_key":     0,
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

def get_library(pipeline: str):
    return requests.get(f"{BASE_URL}/upload/library?pipeline={pipeline}", headers=api_headers())

def get_pending():
    return requests.get(f"{BASE_URL}/upload/pending", headers=api_headers())

def chat_api(query: str, session_id: str, mode: str, file_paths: list = None):
    payload = {"query": query, "session_id": session_id, "mode": mode}
    if file_paths is not None:
        payload["file_paths"] = file_paths
    return requests.post(f"{BASE_URL}/query/chat", json=payload, headers=api_headers())

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
# LIBRARY PANEL (top of sidebar)
# Shows all indexed files for the user. Upload new files here.
# ========================
def render_library(mode: str):
    pipeline = "rag" if mode == "rag" else "sql"
    accept   = ["pdf", "doc", "docx", "txt", "msg", "chm"] if pipeline == "rag" \
               else ["csv", "xlsx", "xls", "db", "sql"]

    with st.sidebar:
        st.markdown("### 📚 File Library")

        # Upload widget
        with st.expander("➕ Upload Files", expanded=False):
            st.caption(f"Accepted: {', '.join(accept).upper()}")
            files = st.file_uploader(
                "Choose files",
                type=accept,
                accept_multiple_files=True,
                key=f"lib_uploader_{mode}_{st.session_state.uploader_key}"
            )
            if st.button("Upload", key=f"lib_upload_btn_{mode}"):
                if files:
                    any_uploaded = False
                    for file in files:
                        res = upload_file(file, st.session_state.token)
                        if res.status_code == 202:
                            data     = res.json()
                            skipped  = data.get("skipped", [])
                            new_files = data.get("files", [])
                            if new_files:
                                st.success(f"✅ {file.name} uploaded.")
                                any_uploaded = True
                            if skipped:
                                st.info(f"⏭ {file.name} already in library — skipped.")
                        else:
                            st.error(f"❌ {file.name}: {res.json().get('detail', res.text)}")
                    # Bump key to reset the uploader widget — prevents re-submission
                    st.session_state.uploader_key += 1
                    st.rerun()
                else:
                    st.warning("Select at least one file.")

        # Pending files
        pend_res = get_pending()
        if pend_res.status_code == 200:
            pending = pend_res.json().get("pending", [])
            if pending:
                st.caption(f"⏳ Embedding {len(pending)} file(s)...")

        # Indexed files list
        lib_res = get_library(pipeline)
        if lib_res.status_code == 200:
            files_in_lib = lib_res.json().get("files", [])
            if files_in_lib:
                for f in files_in_lib:
                    date = f["uploaded_at"][:10]
                    st.markdown(
                        f'<div class="library-file">📄 {f["file_name"]} <span style="opacity:0.6">({date})</span></div>',
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No files yet. Upload above.")
        else:
            st.caption("Could not load library.")

        st.divider()


# ========================
# NEW CHAT FILE SELECTOR
# Shown inline above the chat when no session is active.
# ========================
def render_file_selector(mode: str) -> list:
    """
    Renders a multiselect to choose files for a new session.
    Returns list of selected file_paths, or empty list if none chosen.
    """
    pipeline = "rag" if mode == "rag" else "sql"
    lib_res  = get_library(pipeline)

    if lib_res.status_code != 200:
        st.error("Could not load file library.")
        return []

    files_in_lib = lib_res.json().get("files", [])

    if not files_in_lib:
        st.warning("📭 No files in your library yet. Upload files using the sidebar.")
        return []

    st.markdown("#### 🗂 Choose files for this chat")
    st.caption("These files will be locked to this session. Upload more via the sidebar.")

    file_name_to_path = {f["file_name"]: f["file_path"] for f in files_in_lib}
    all_names         = list(file_name_to_path.keys())

    # Select All checkbox
    select_all = st.checkbox("Select All", key=f"select_all_{mode}")

    default = all_names if select_all else []
    chosen_names = st.multiselect(
        "Select files",
        options=all_names,
        default=default,
        key=f"file_multiselect_{mode}"
    )

    if not chosen_names:
        return []

    return [file_name_to_path[n] for n in chosen_names]


# ========================
# SIDEBAR — chat history + library
# ========================
def render_sidebar(mode: str):
    if "selected_session" not in st.session_state:
        st.session_state.selected_session = None

    # Library panel at top
    render_library(mode)

    with st.sidebar:
        st.markdown(f"### 💬 {'Document' if mode == 'rag' else 'Data'} Chats")

        if st.button("➕ New Chat", key="new_chat_btn"):
            st.session_state.session_id     = None
            st.session_state.messages       = []
            st.session_state.selected_files = []
            st.session_state.new_chat_open  = True
            st.session_state.selected_session = None
            st.rerun()

        st.divider()

        res = get_sessions(mode)
        if res.status_code == 200:
            sessions = res.json().get("sessions", [])
            if not sessions:
                st.caption("No past chats yet.")
            for s in sessions:
                label   = s["title"] or "Untitled Chat"
                created = s["created_at"][:10]
                btn_label = f"📄 {label[:28]} ({created})"
                is_active = st.session_state.session_id == s["session_id"]

                if st.button(
                    btn_label,
                    key=f"sess_{s['session_id']}",
                    type="primary" if is_active else "secondary"
                ):
                    st.session_state.selected_session = s["session_id"]
                    st.session_state.new_chat_open    = False
                    st.session_state.selected_files   = []
                    st.rerun()
        else:
            st.caption("Could not load sessions.")

        st.divider()
        if st.button("⬅ Home", key="sidebar_home"):
            st.session_state.page           = "home"
            st.session_state.session_id     = None
            st.session_state.messages       = []
            st.session_state.selected_files = []
            st.session_state.new_chat_open  = False
            st.session_state.selected_session = None
            st.rerun()

    # Load selected session outside sidebar
    if st.session_state.selected_session:
        target = st.session_state.selected_session
        if target != st.session_state.session_id:
            hist_res = get_session_history(target)
            if hist_res.status_code == 200:
                st.session_state.session_id = target
                st.session_state.messages   = hist_res.json().get("messages", [])
        st.session_state.selected_session = None


# ========================
# CHAT MESSAGE RENDERER
# ========================
def render_messages(messages: list):
    for msg in messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-row"><div class="user-bubble">🧑 {msg["content"]}</div></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="assistant-row"><div class="assistant-bubble">🤖 {msg["content"]}</div></div>',
                unsafe_allow_html=True
            )


# ========================
# CHART RENDERER
# Renders a Plotly chart inline in the SQL chat.
# Fixed cool blue palette — no theme switching.
# chart_type: bar | line | pie | scatter
# ========================

BLUE_PALETTE  = ["#4ca3dd", "#1a6eb5", "#7ec8e3", "#0d3b6e",
                  "#2d6a9f", "#a8d8ea", "#134f8a", "#5bb3d0"]
BLUE_SINGLE   = "#4ca3dd"
BLUE_GRID     = "rgba(76,163,221,0.12)"
BLUE_PLOT_BG  = "rgba(0,0,0,0.15)"


def _render_chart(viz_config: dict, result_data: list):
    """
    Render a Plotly chart inline in the SQL chat.
    viz_config: {"chart_type": str, "x_col": str, "y_col": str, "title": str}
    result_data: list of row dicts from the DataFrame
    """
    try:
        import plotly.express as px
        import pandas as pd

        chart_type = viz_config.get("chart_type", "none")
        x_col      = viz_config.get("x_col")
        y_col      = viz_config.get("y_col")
        title      = viz_config.get("title") or "Query Result"

        if not result_data or chart_type == "none":
            st.caption("📊 No visualization for this query.")
            return

        df = pd.DataFrame(result_data)

        if x_col and x_col not in df.columns:
            st.caption("📊 No visualization for this query.")
            return
        if y_col and y_col not in df.columns:
            st.caption("📊 No visualization for this query.")
            return

        # ── Shared layout ─────────────────────────────────────────
        layout_kwargs = dict(
            title=title,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor=BLUE_PLOT_BG,
            font=dict(color="white", family="sans-serif"),
            title_font=dict(size=16, color="white"),
            margin=dict(l=40, r=40, t=55, b=40),
            xaxis=dict(
                gridcolor=BLUE_GRID,
                linecolor="rgba(255,255,255,0.2)",
                tickfont=dict(color="white"),
                title_font=dict(color="white"),
            ),
            yaxis=dict(
                gridcolor=BLUE_GRID,
                linecolor="rgba(255,255,255,0.2)",
                tickfont=dict(color="white"),
                title_font=dict(color="white"),
            ),
            legend=dict(font=dict(color="white")),
        )

        # ── Bar ───────────────────────────────────────────────────
        if chart_type == "bar":
            fig = px.bar(
                df, x=x_col, y=y_col,
                title=title,
                color=x_col,
                color_discrete_sequence=BLUE_PALETTE
            )
            fig.update_traces(marker_line_width=0)
            fig.update_layout(**layout_kwargs)

        # ── Line ──────────────────────────────────────────────────
        elif chart_type == "line":
            fig = px.line(
                df, x=x_col, y=y_col,
                title=title,
                markers=True,
                color_discrete_sequence=[BLUE_SINGLE]
            )
            fig.update_traces(
                line=dict(width=2.5, color=BLUE_SINGLE),
                marker=dict(size=7, color=BLUE_SINGLE,
                            line=dict(width=1.5, color="white"))
            )
            fig.update_layout(**layout_kwargs)

        # ── Pie / Donut ───────────────────────────────────────────
        elif chart_type == "pie":
            fig = px.pie(
                df, names=x_col,
                values=y_col if y_col else None,
                title=title,
                color_discrete_sequence=BLUE_PALETTE,
                hole=0.38
            )
            fig.update_layout(**{
                k: v for k, v in layout_kwargs.items()
                if k not in ("xaxis", "yaxis")
            })
            fig.update_traces(
                textfont_color="white",
                textfont_size=12,
                marker=dict(line=dict(color="rgba(0,0,0,0.4)", width=1.5))
            )

        # ── Scatter ───────────────────────────────────────────────
        elif chart_type == "scatter":
            fig = px.scatter(
                df, x=x_col, y=y_col,
                title=title,
                color_discrete_sequence=[BLUE_SINGLE],
                trendline="ols" if len(df) >= 5 else None
            )
            fig.update_traces(marker=dict(size=9, opacity=0.85,
                                          line=dict(width=1, color="white")))
            fig.update_layout(**layout_kwargs)

        else:
            st.caption("📊 No visualization for this query.")
            return

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.caption(f"📊 Could not render chart: {e}")


# ========================
# AUTH PAGE
# ========================
def auth_page():
    st.title("🔐 Welcome")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Login")
        name     = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            if not name or not password:
                st.warning("Please enter username and password")
            else:
                try:
                    res = login(name, password)
                    if res.status_code == 200:
                        st.session_state.token = res.json()["access_token"]
                        st.session_state.page  = "home"
                        st.rerun()
                    else:
                        st.error(res.json().get("detail", "Invalid credentials"))
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Is it running?")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")

    with tab2:
        st.subheader("Register")
        name     = st.text_input("Username", key="reg_user")
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
            st.session_state.page           = "rag"
            st.session_state.sidebar_mode   = "rag"
            st.session_state.session_id     = None
            st.session_state.messages       = []
            st.session_state.new_chat_open  = True
            st.rerun()

    with tab2:
        st.subheader("Data Analysis")
        if st.button("Go to Data Analysis"):
            st.session_state.page           = "sql"
            st.session_state.sidebar_mode   = "sql"
            st.session_state.session_id     = None
            st.session_state.messages       = []
            st.session_state.new_chat_open  = True
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

    # ── New chat: show file selector ────────────────────────────────
    if not st.session_state.session_id:
        st.info("Select files from your library to start a new chat session.")
        selected = render_file_selector("rag")
        st.session_state.selected_files = selected

        if not selected:
            return  # Can't proceed without files

        st.divider()
        st.success(f"✅ {len(selected)} file(s) selected. Ask your first question below.")

        # Check pending
        pend_res = get_pending()
        if pend_res.status_code == 200:
            pending = pend_res.json().get("pending", [])
            if pending:
                st.warning(f"⏳ {len(pending)} file(s) are still being embedded and won't be queryable yet.")

    else:
        # ── Active session: show locked files ────────────────────────
        sessions_res = get_sessions("rag")
        if sessions_res.status_code == 200:
            sessions = sessions_res.json().get("sessions", [])
            curr = next((s for s in sessions if s["session_id"] == st.session_state.session_id), None)
            if curr and curr.get("file_paths"):
                import os
                names = [os.path.basename(p) for p in curr["file_paths"]]
                st.caption(f"🔒 Knowledge base: {', '.join(names)}")

    st.divider()

    # ── Summarize button ────────────────────────────────────────────
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
                            "role":    "assistant",
                            "content": f"**📋 Chat Summary:**\n\n{summary}"
                        })
                        st.rerun()
                    else:
                        st.error("Could not summarize.")

    # ── Message history ─────────────────────────────────────────────
    render_messages(st.session_state.messages)

    # ── Chat input ──────────────────────────────────────────────────
    query = st.chat_input("Ask a question about your document(s)...")
    if query:
        is_new_session  = st.session_state.session_id is None
        file_paths_arg  = st.session_state.selected_files if is_new_session else None

        if is_new_session and not file_paths_arg:
            st.warning("Please select at least one file before chatting.")
            return

        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Thinking..."):
            res = chat_api(
                query=query,
                session_id=st.session_state.session_id,
                mode="rag",
                file_paths=file_paths_arg
            )

        if res.status_code == 200:
            data       = res.json()
            result     = data["response"]
            session_id = result.get("session_id")
            answer     = result.get("answer", "")
            sources    = result.get("sources", [])

            st.session_state.session_id    = session_id
            st.session_state.selected_files = []  # locked — clear selector state
            st.session_state.new_chat_open  = False

            # Format sources
            source_text = ""
            if sources:
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
                "role":    "assistant",
                "content": answer + source_text
            })
        else:
            st.session_state.messages.append({
                "role":    "assistant",
                "content": f"❌ Error: {res.json().get('detail', 'Something went wrong.')}"
            })

        st.rerun()


# ========================
# NL2SQL CHAT PAGE
# ========================
def sql_page():
    render_sidebar("sql")
    st.title("📊 Data Analysis Chat")

    # ── New chat: show file selector ────────────────────────────────
    if not st.session_state.session_id:
        st.info("Select data files from your library to start a new chat session.")
        selected = render_file_selector("sql")
        st.session_state.selected_files = selected

        if not selected:
            return

        st.divider()
        st.success(f"✅ {len(selected)} file(s) selected. Ask your first question below.")

    else:
        # ── Active session: show locked files ────────────────────────
        sessions_res = get_sessions("sql")
        if sessions_res.status_code == 200:
            sessions = sessions_res.json().get("sessions", [])
            curr = next((s for s in sessions if s["session_id"] == st.session_state.session_id), None)
            if curr and curr.get("file_paths"):
                import os
                names = [os.path.basename(p) for p in curr["file_paths"]]
                st.caption(f"🔒 Knowledge base: {', '.join(names)}")

    st.divider()

    # ── Summarize button ────────────────────────────────────────────
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
                            "role":    "assistant",
                            "content": f"**📋 Chat Summary:**\n\n{summary}"
                        })
                        st.rerun()
                    else:
                        st.error("Could not summarize.")

    # ── Message history ─────────────────────────────────────────────
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-row"><div class="user-bubble">🧑 {msg["content"]}</div></div>',
                unsafe_allow_html=True
            )
        else:
            content = msg["content"]
            if isinstance(content, dict):
                if content.get("error"):
                    st.error(f"❌ {content['error']}")
                else:
                    st.markdown(
                        f'<div class="assistant-row"><div class="assistant-bubble">🤖 {content.get("natural_answer", "")}</div></div>',
                        unsafe_allow_html=True
                    )
                    with st.expander("🔍 SQL Query"):
                        st.code(content.get("sql_query", ""), language="sql")
                    with st.expander("📋 Data Result"):
                        st.markdown(content.get("result_markdown", "_No results._"))

                    # ── Visualization ────────────────────────────────────
                    viz_config  = content.get("viz_config", {})
                    result_data = content.get("result_data", [])
                    chart_type  = viz_config.get("chart_type", "none")

                    if chart_type == "none" or not result_data:
                        st.caption("📊 No visualization for this query.")
                    else:
                        _render_chart(viz_config, result_data)
            else:
                st.markdown(
                    f'<div class="assistant-row"><div class="assistant-bubble">🤖 {content}</div></div>',
                    unsafe_allow_html=True
                )

    # ── Chat input ──────────────────────────────────────────────────
    query = st.chat_input("Ask a question about your data...")
    if query:
        is_new_session = st.session_state.session_id is None
        file_paths_arg = st.session_state.selected_files if is_new_session else None

        if is_new_session and not file_paths_arg:
            st.warning("Please select at least one file before chatting.")
            return

        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Analyzing..."):
            res = chat_api(
                query=query,
                session_id=st.session_state.session_id,
                mode="sql",
                file_paths=file_paths_arg
            )

        if res.status_code == 200:
            data       = res.json()
            result     = data["response"]
            session_id = result.get("session_id")

            st.session_state.session_id     = session_id
            st.session_state.selected_files  = []
            st.session_state.new_chat_open   = False

            st.session_state.messages.append({
                "role":    "assistant",
                "content": result
            })
        else:
            st.session_state.messages.append({
                "role":    "assistant",
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