from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from backend.core.config import settings

# ========================
# PASSWORD HASHING CONTEXT
# ========================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ========================
# TOKEN SCHEME (Bearer)
# ========================
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ========================
# PASSWORD FUNCTIONS
# ========================
def hash_password(password: str) -> str:
    """
    Hash plain password using bcrypt
    """
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """
    Verify plain password against hashed password
    """
    return pwd_context.verify(plain, hashed)


# ========================
# TOKEN CREATION
# ========================
def create_access_token(data: dict) -> str:
    """
    Create JWT access token with expiration
    """
    to_encode = data.copy()

    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )

    to_encode.update({"exp": expire})

    return jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )


# ========================
# TOKEN DECODE + USER EXTRACTION
# ========================
def get_current_user(token: str = Depends(oauth2_scheme)) -> int:
    """
    Extract and validate user from JWT token

    Returns:
        user_id (int)

    Raises:
        HTTPException (401) if token invalid
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )

        user_id: str = payload.get("sub")

        if user_id is None:
            raise credentials_exception

        return int(user_id)

    except JWTError:
        raise credentials_exception