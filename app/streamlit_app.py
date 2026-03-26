import streamlit as st
import requests

BASE_URL = "http://127.0.0.1:8000"

# ========================
# SESSION INIT
# ========================
if "token" not in st.session_state:
    st.session_state.token = None

if "user" not in st.session_state:
    st.session_state.user = None

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
# UI
# ========================
st.title("📊 LLM Data Assistant")

menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

# ========================
# REGISTER
# ========================
if choice == "Register":
    st.subheader("Create Account")

    name = st.text_input("Username")
    password = st.text_input("Password", type="password")

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
# LOGIN
# ========================
elif choice == "Login":
    st.subheader("Login")

    name = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        with st.spinner("Logging in..."):
            try:
                res = login(name, password)
                if res.status_code == 200:
                    st.session_state.token = res.json()["access_token"]
                    st.success("Login successful")
                else:
                    st.error("Invalid credentials")
            except:
                st.error("Backend not running")

# ========================
# MAIN APP (AFTER LOGIN)
# ========================
if st.session_state.token:

    st.sidebar.success("Logged in")

    # -------- Upload --------
    st.subheader("📂 Upload File")
    uploaded_file = st.file_uploader("Upload your file")

    if st.button("Upload"):
        if uploaded_file:
            res = upload_file(uploaded_file, st.session_state.token)
            if res.status_code == 200:
                st.success("File uploaded")
            else:
                st.error(res.text)
        else:
            st.warning("Upload a file first")

    # -------- Query --------
    st.subheader("💬 Ask Question")

    query = st.text_input("Enter your query")

    if st.button("Submit Query"):
        if not query:
            st.warning("Enter a query")
        else:
            with st.spinner("Processing..."):
                res = query_api(query, st.session_state.token)

                if res.status_code == 200:
                    data = res.json()
                    st.write("### Route:", data["route"])
                    st.write("### Response:", data["response"])
                else:
                    st.error(res.text)

    # -------- Logout --------
    if st.button("Logout"):
        st.session_state.token = None
        st.rerun()