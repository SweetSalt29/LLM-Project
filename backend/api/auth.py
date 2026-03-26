from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
import sqlite3
from backend.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user
)
from backend.models.schemas import UserRegister, Token

router = APIRouter(prefix="/auth", tags=["auth"])


# ========================
# DATABASE
# ========================
def get_db():
    return sqlite3.connect("app.db")


def init_db():
    """
    Initialize users table
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT UNIQUE,
        password TEXT
    )
    """)

    conn.commit()
    conn.close()


# Initialize once
init_db()


# ========================
# REGISTER
# ========================
@router.post("/register")
def register(user: UserRegister):
    """
    Register a new user
    """
    try:
        if not user.name.strip() or not user.password.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username and password cannot be empty"
            )

        conn = get_db()
        cursor = conn.cursor()

        hashed = hash_password(user.password)

        cursor.execute(
            "INSERT INTO users (name, password) VALUES (?, ?)",
            (user.name, hashed)
        )

        conn.commit()
        conn.close()

        return {"message": "User created successfully"}

    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already exists"
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )


# ========================
# LOGIN
# ========================
@router.post("/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate user and return JWT token
    """
    try:
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, password FROM users WHERE name=?",
            (form.username,)
        )

        user = cursor.fetchone()
        conn.close()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

        user_id, hashed = user

        if not verify_password(form.password, hashed):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

        token = create_access_token({"sub": str(user_id)})

        return {
            "access_token": token,
            "token_type": "bearer"
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )


# ========================
# GET CURRENT USER
# ========================
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
import sqlite3
from backend.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user
)
from backend.models.schemas import UserRegister, Token

router = APIRouter(prefix="/auth", tags=["auth"])


# ========================
# DATABASE
# ========================
def get_db():
    return sqlite3.connect("app.db")


def init_db():
    """
    Initialize users table
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT UNIQUE,
        password TEXT
    )
    """)

    conn.commit()
    conn.close()


# Initialize once
init_db()


# ========================
# REGISTER
# ========================
@router.post("/register")
def register(user: UserRegister):
    """
    Register a new user
    """
    try:
        if not user.name.strip() or not user.password.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username and password cannot be empty"
            )

        conn = get_db()
        cursor = conn.cursor()

        hashed = hash_password(user.password)

        cursor.execute(
            "INSERT INTO users (name, password) VALUES (?, ?)",
            (user.name, hashed)
        )

        conn.commit()
        conn.close()

        return {"message": "User created successfully"}

    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already exists"
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )


# ========================
# LOGIN
# ========================
@router.post("/login", response_model=Token)
def login(form: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate user and return JWT token
    """
    try:
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, password FROM users WHERE name=?",
            (form.username,)
        )

        user = cursor.fetchone()
        conn.close()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

        user_id, hashed = user

        if not verify_password(form.password, hashed):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

        token = create_access_token({"sub": str(user_id)})

        return {
            "access_token": token,
            "token_type": "bearer"
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )


# ========================
# GET CURRENT USER
# ========================
@router.get("/me")
def get_me(user_id: int = Depends(get_current_user)):
    """
    Get current authenticated user
    """
    return {"user_id": user_id}