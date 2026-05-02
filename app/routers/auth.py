import os
from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_user, get_db_session
from app.models import User
from app.schemas import (
    RefreshTokenRequest,
    TokenPairResponse,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
)

router = APIRouter(prefix="/auth", tags=["auth"])
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-me")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))


def create_token(subject: str, token_type: str, expires_delta: timedelta) -> str:
    expire_at = datetime.now(timezone.utc) + expires_delta
    return jwt.encode(
        {
            "sub": subject,
            "exp": expire_at,
            "type": token_type,
        },
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


def create_token_pair(user_id: UUID) -> TokenPairResponse:
    subject = str(user_id)
    access_token = create_token(
        subject=subject,
        token_type="access",
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    refresh_token = create_token(
        subject=subject,
        token_type="refresh",
        expires_delta=timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    )
    return TokenPairResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    payload: UserRegisterRequest,
    db_session: AsyncSession = Depends(get_db_session),
) -> UserResponse:
    existing_user = await db_session.scalar(select(User).where(User.email == payload.email))
    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    new_user = User(
        email=payload.email,
        hashed_password=pwd_context.hash(payload.password),
        name=payload.name,
        surname=payload.surname,
        patronymic=payload.patronymic,
    )
    db_session.add(new_user)
    await db_session.commit()
    await db_session.refresh(new_user)
    return new_user


@router.post("/login", response_model=TokenPairResponse)
async def login_user(
    payload: UserLoginRequest,
    db_session: AsyncSession = Depends(get_db_session),
) -> TokenPairResponse:
    user = await db_session.scalar(select(User).where(User.email == payload.email))
    if user is None or not pwd_context.verify(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    return create_token_pair(user.id)


@router.post("/refresh", response_model=TokenPairResponse)
async def refresh_tokens(
    payload: RefreshTokenRequest,
    db_session: AsyncSession = Depends(get_db_session),
) -> TokenPairResponse:
    try:
        decoded_token = jwt.decode(payload.refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        subject = decoded_token.get("sub")
        token_type = decoded_token.get("type")
        if subject is None or token_type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )
        user_id = UUID(subject)
    except (JWTError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        ) from None

    user = await db_session.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    return create_token_pair(user.id)


@router.get("/profile", response_model=UserResponse)
async def get_auth_profile(current_user: User = Depends(get_current_user)) -> UserResponse:
    return current_user
