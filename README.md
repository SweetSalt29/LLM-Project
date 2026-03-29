# 📁 Ask Your Files

An AI-powered document and data analysis platform. Upload your files and have a full conversation with them — ask questions, get insights, and explore your data through natural language.

---

## ✨ Features

- 📄 **Document Chat (RAG)** — Upload PDFs, DOCX, DOC, or TXT files and have a multi-turn conversation with your documents
- 📊 **Data Analysis Chat (NL2SQL)** — Upload CSV, Excel, or database files and query them in plain English
- 💬 **Conversation Memory** — Full chat history per session; follow-up questions work naturally
- 🗂️ **Session Sidebar** — All past chats listed like ChatGPT/Claude, click to reopen any session
- 📋 **Summarize Chat** — Generate a summary of any conversation session on demand
- 🔐 **User Authentication** — Register and login with JWT-secured sessions
- 🛡️ **SQL Guardrails** — Only SELECT queries allowed; write operations are blocked
- 🖼️ **Multimodal RAG** — Images extracted from PDFs and linked to their text chunks

---

## 🗂️ Project Structure

```
project/
│
├── backend/
│   ├── main.py                        # FastAPI app entry point
│   ├── api/
│   │   ├── auth.py                    # Register, login, JWT
│   │   ├── upload.py                  # File upload + background ingestion
│   │   └── query.py                   # Chat, summarize, session endpoints
│   │
│   ├── modules/
│   │   ├── file_handler.py            # Save uploaded files to disk
│   │   ├── chat_memory.py             # SQLite: sessions + message history
│   │   ├── rag/
│   │   │   ├── rag_loader.py          # Document loading (Docling + plain text)
│   │   │   ├── rag_pipeline.py        # RAG query + conversation memory
│   │   │   └── embeddings.py          # FAISS vector store per user
│   │   └── nl2sql/
│   │       └── nl2sql_pipeline.py     # NL2SQL query + guardrails + memory
│   │
│   ├── router/
│   │   └── dispatcher.py              # Routes query to rag / nl2sql / hybrid
│   │
│   ├── state/
│   │   └── session_manager.py         # In-memory upload state per user
│   │
│   ├── core/
│   │   └── security.py                # Password hashing, JWT creation/verification
│   │
│   └── models/
│       └── schemas.py                 # Pydantic request/response models
│
├── app.py                             # Streamlit frontend
├── requirements.txt
├── .env                               # API keys (not committed)
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo>
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
SECRET_KEY=your_jwt_secret_key_here
```

Get your OpenRouter API key at [openrouter.ai](https://openrouter.ai).

### 5. Run the backend

```bash
uvicorn backend.main:app --reload
```

Backend runs at `http://127.0.0.1:8000`.  
API docs available at `http://127.0.0.1:8000/docs`.

### 6. Run the frontend

In a separate terminal:

```bash
streamlit run app.py
```

Frontend runs at `http://localhost:8501`.

---

## 📂 Supported File Types

| Pipeline | Formats |
|----------|---------|
| RAG (Document Chat) | `.pdf`, `.doc`, `.docx`, `.txt` |
| NL2SQL (Data Analysis) | `.csv`, `.xlsx`, `.xls`, `.db`, `.sql` |

> Mixing RAG and SQL files in a single upload is not allowed. Upload them separately.

---

## 🧠 How It Works

### RAG Pipeline

1. File is uploaded and saved to `data/uploads/user_{id}/`
2. Docling parses PDFs/DOCX for text and table structure; TXT files are chunked directly
3. Images are extracted from PDFs via PyMuPDF and linked to their page's text
4. Text chunks are embedded using `sentence-transformers/all-mpnet-base-v2` and stored in a per-user FAISS index
5. On query: top-5 relevant chunks are retrieved, combined with conversation history, and sent to the LLM via OpenRouter

### NL2SQL Pipeline

1. File is uploaded and saved to disk — no embedding needed
2. On query: CSV/Excel files are loaded into an in-memory SQLite database
3. Column names are sanitized (spaces and special characters replaced with underscores)
4. LLM generates a SQLite SELECT query from the schema + conversation history
5. Guardrails validate the query (SELECT only, no forbidden keywords, no SQL comments)
6. Query is executed and results are returned as a markdown table
7. A second LLM call generates a natural language answer

### Memory

- **In-chat memory** — Full message history of the session is passed to the LLM with every turn
- **Session memory** — Sessions and messages are persisted in SQLite (`app.db`). Clicking "Summarize Chat" sends the full conversation to the LLM and stores the summary against the session

---

## 🔐 Authentication

- Passwords are hashed with bcrypt via `passlib`
- Sessions use JWT tokens signed with `python-jose`
- All API endpoints except `/auth/register` and `/auth/login` require a Bearer token

---

## 🛡️ SQL Guardrails

All LLM-generated SQL queries are validated before execution:

- Must start with `SELECT`
- Blocked keywords: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, `CREATE`, `REPLACE`, `MERGE`, `EXEC`, `EXECUTE`, `PRAGMA`, `ATTACH`, `DETACH`
- SQL comments (`--`, `/*`) are blocked
- Queries that fail validation are returned to the user with a clear error message — never executed

---

## 🗄️ Database Schema

All data is stored in `app.db` (SQLite).

| Table | Purpose |
|-------|---------|
| `users` | User accounts (id, name, hashed password) |
| `chat_sessions` | Session metadata (id, user, title, mode, summary, timestamps) |
| `chat_messages` | Per-session message history (role, content, timestamp) |
| `queries` | Legacy query log (user, query, response, route, timestamp) |
| `nl2sql_logs` | NL2SQL query log (user, session, query, generated SQL, timestamp) |

---

## 🤖 LLM

The project uses **Meta Llama 3 8B Instruct** (`meta-llama/llama-3-8b-instruct`) via [OpenRouter](https://openrouter.ai). To switch models, change the `MODEL` constant in `rag_pipeline.py` and `nl2sql_pipeline.py`.

---

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` + `uvicorn` | Backend API |
| `streamlit` | Frontend UI |
| `docling` + `langchain-docling` | PDF/DOCX parsing |
| `pymupdf` | Image extraction from PDFs |
| `langchain-huggingface` + `faiss-cpu` | Embeddings and vector search |
| `sentence-transformers` | Embedding model |
| `pandas` + `openpyxl` | CSV/Excel loading |
| `passlib` + `python-jose` | Auth and JWT |
| `requests` | OpenRouter API calls |

---

## 🚀 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/register` | Register new user |
| POST | `/auth/login` | Login, returns JWT |
| GET | `/auth/me` | Get current user |
| POST | `/upload/` | Upload files |
| GET | `/upload/status` | Check ingestion status |
| POST | `/query/chat` | Send a chat message |
| POST | `/query/summarize` | Summarize a session |
| GET | `/query/sessions` | List all sessions for sidebar |
| GET | `/query/sessions/{id}` | Load a past session's messages |