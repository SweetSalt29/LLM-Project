import streamlit as st
import requests

BASE_URL = "http://127.0.0.1:8000"

# ========================
# SESSION INIT
# ========================
if "token" not in st.session_state:
    st.session_state.token = None

if "page" not in st.session_state:
    st.session_state.page = "login"

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
        files={"file": file},
        headers=headers
    )

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
            try:
                res = login(name, password)
                if res.status_code == 200:
                    st.session_state.token = res.json()["access_token"]
                    st.session_state.page = "home"
                    st.rerun()
                else:
                    st.error("Invalid credentials")
            except:
                st.error("Backend not running")

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

    # -------- Description --------
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

    # -------- Tabs (TOP BAR) --------
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

    # -------- Logout --------
    if st.button("Logout"):
        st.session_state.token = None
        st.session_state.page = "login"
        st.rerun()

# ========================
# RAG PAGE
# ========================
def rag_page():
    st.title("📄 Ask About Your Document")

    st.subheader("Upload Document")
    file = st.file_uploader("Upload file")

    if st.button("Upload Document"):
        if file:
            res = upload_file(file, st.session_state.token)
            if res.status_code == 200:
                st.success("File uploaded")
            else:
                st.error(res.text)
        else:
            st.warning("Upload a file")

    st.divider()

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
                st.error(res.text)

    # -------- Summarize --------
    if st.button("Summarize Document"):
        res = query_api("summarize the document", st.session_state.token)
        if res.status_code == 200:
            st.write("### Summary")
            st.write(res.json()["response"])

    if st.button("⬅ Back"):
        st.session_state.page = "home"
        st.rerun()

# ========================
# NL2SQL PAGE
# ========================
def sql_page():
    st.title("📊 Analyze Your Data")

    st.subheader("Upload Data File")
    file = st.file_uploader("Upload CSV / Excel")

    if st.button("Upload Data"):
        if file:
            res = upload_file(file, st.session_state.token)
            if res.status_code == 200:
                st.success("File uploaded")
            else:
                st.error(res.text)
        else:
            st.warning("Upload a file")

    st.divider()

    st.subheader("📈 Ask Data Questions")

    query = st.text_input("e.g. total sales, average price")

    if st.button("Analyze"):
        if query:
            res = query_api(query, st.session_state.token)
            if res.status_code == 200:
                data = res.json()
                st.write("### Result")
                st.write(data["response"])
            else:
                st.error(res.text)

    if st.button("⬅ Back"):
        st.session_state.page = "home"
        st.rerun()

# ========================
# ROUTER
# ========================
if not st.session_state.token:
    auth_page()
else:
    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "rag":
        rag_page()
    elif st.session_state.page == "sql":
        sql_page()