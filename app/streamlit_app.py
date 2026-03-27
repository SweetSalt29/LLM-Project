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

        </style>
        """,
        unsafe_allow_html=True
    )


BASE_URL = "http://127.0.0.1:8000"

# ========================
# SESSION INIT
# ========================
if "token" not in st.session_state:
    st.session_state.token = None

if "page" not in st.session_state:
    st.session_state.page = "login"

if "doc_status" not in st.session_state:
    st.session_state.doc_status = None  # None | "processing" | "ready" | "failed:..."


# ========================
# API CALLS
# ========================
def register(name, password):
    return requests.post(
        f"{BASE_URL}/auth/register",
        json={"name": name, "password": password}
    )

def login(username, password):
    return requests.post(
        f"{BASE_URL}/auth/login",
        data={"username": username, "password": password}
    )

def upload_file(file, token):
    headers = {"Authorization": f"Bearer {token}"}
    return requests.post(
        f"{BASE_URL}/upload/",
        files={"files": (file.name, file.read())},
        headers=headers
    )

def get_upload_status(token):
    headers = {"Authorization": f"Bearer {token}"}
    return requests.get(f"{BASE_URL}/upload/status", headers=headers)

def query_api(query, token):
    headers = {"Authorization": f"Bearer {token}"}
    return requests.post(
        f"{BASE_URL}/query/",
        json={"query": query},
        headers=headers
    )


# ========================
# AUTH PAGE
# ========================
def auth_page():
    st.title("🔐 Welcome")

    tab1, tab2 = st.tabs(["Login", "Register"])

    # -------- LOGIN --------
    with tab1:
        st.subheader("Login")
        name = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            # Guard: don't hit backend with empty fields
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
                        # Show actual backend error, not a hardcoded string
                        detail = res.json().get("detail", "Invalid credentials")
                        st.error(detail)
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Is it running?")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")

    # -------- REGISTER --------
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
- 📄 **Ask About Your Document**  
  Upload documents and ask questions, get summaries, and explore content.

- 📊 **Analyze Your Data**  
  Upload structured data (CSV/Excel) and get insights like totals, averages, and trends.

Choose a mode below to get started.
""")

    st.divider()

    tab1, tab2 = st.tabs(["📄 Ask About Your Document", "📊 Analyze Your Data"])

    with tab1:
        st.subheader("Document Intelligence")
        st.write("Ask questions, summarize, and explore your documents.")
        if st.button("Go to Document Chat"):
            st.session_state.page = "rag"
            st.rerun()

    with tab2:
        st.subheader("Data Analysis")
        st.write("Get insights, calculations, and trends from your data.")
        if st.button("Go to Data Analysis"):
            st.session_state.page = "sql"
            st.rerun()

    st.divider()

    if st.button("Logout"):
        st.session_state.token = None
        st.session_state.page = "login"
        st.rerun()


# ========================
# STATUS BANNER
# Shows processing / ready / failed state clearly
# ========================
def show_status_banner():
    s = st.session_state.doc_status

    if s == "processing":
        st.warning("⏳ Document is still being processed. Please wait before querying.")

        # Poll backend every 3 seconds until ready
        with st.spinner("Processing..."):
            time.sleep(3)
            res = get_upload_status(st.session_state.token)
            if res.status_code == 200:
                new_status = res.json().get("status")
                st.session_state.doc_status = new_status
                st.rerun()

    elif s == "ready":
        st.success("✅ Document ready. You can now ask questions.")

    elif s and s.startswith("failed"):
        st.error(f"❌ Ingestion failed: {s}. Please re-upload your file.")


# ========================
# RAG PAGE
# ========================
def rag_page():
    st.title("📄 Ask About Your Document")

    st.subheader("Upload Document")
    file = st.file_uploader("Upload file", key="rag_uploader")

    if st.button("Upload Document"):
        if file:
            res = upload_file(file, st.session_state.token)
            if res.status_code == 202:
                st.session_state.doc_status = "processing"
                st.success("File uploaded! Processing in background...")
                st.rerun()
            else:
                st.error(res.text)
        else:
            st.warning("Please select a file first")

    # Always show current ingestion status
    show_status_banner()

    st.divider()

    # Only show chat UI once document is ready
    if st.session_state.doc_status == "ready":
        st.subheader("💬 Chat with Document")

        query = st.text_input("Ask a question")

        if st.button("Ask"):
            if query:
                res = query_api(query, st.session_state.token)
                if res.status_code == 200:
                    data = res.json()
                    st.write("### Answer")
                    st.write(data["response"])
                else:
                    st.error(res.json().get("detail", res.text))

        if st.button("Summarize Document"):
            res = query_api("summarize the document", st.session_state.token)
            if res.status_code == 200:
                st.write("### Summary")
                st.write(res.json()["response"])
            else:
                st.error(res.json().get("detail", res.text))

    if st.button("⬅ Back"):
        st.session_state.page = "home"
        st.rerun()


# ========================
# NL2SQL PAGE
# ========================
def sql_page():
    st.title("📊 Analyze Your Data")

    st.subheader("Upload Data File")
    file = st.file_uploader("Upload CSV / Excel", key="sql_uploader")

    if st.button("Upload Data"):
        if file:
            res = upload_file(file, st.session_state.token)
            if res.status_code == 202:
                st.session_state.doc_status = "processing"
                st.success("File uploaded! Processing in background...")
                st.rerun()
            else:
                st.error(res.text)
        else:
            st.warning("Please select a file first")

    show_status_banner()

    st.divider()

    if st.session_state.doc_status == "ready":
        st.subheader("📈 Ask Data Questions")

        query = st.text_input("e.g. total sales, average price")

        if st.button("Analyze"):
            if query:
                res = query_api(query, st.session_state.token)
                if res.status_code == 200:
                    data = res.json()
                    result = data.get("response", {})

                    # Pipeline returned an error (guardrail block, bad SQL, etc.)
                    if result.get("error"):
                        st.error(f"❌ {result['error']}")
                    else:
                        # 1. User query
                        st.markdown("### 🙋 User Query")
                        st.info(result.get("user_query", query))

                        # 2. Generated SQL
                        st.markdown("### 🔍 Generated SQL Query")
                        st.code(result.get("sql_query", ""), language="sql")

                        # 3. Natural language answer
                        st.markdown("### 💬 Answer")
                        st.success(result.get("natural_answer", ""))

                        # 4. Raw result as markdown table
                        st.markdown("### 📋 Data Result")
                        st.markdown(result.get("result_markdown", "_No results._"))
                else:
                    st.error(res.json().get("detail", res.text))

    if st.button("⬅ Back"):
        st.session_state.page = "home"
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