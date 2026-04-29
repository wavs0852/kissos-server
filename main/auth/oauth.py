import os
from pathlib import Path
import time

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / '.env')
from typing import Any, Literal

import httpx
import jwt
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


Provider = Literal["google"]

KISSOS_AUTH_SECRET = os.getenv("KISSOS_AUTH_SECRET", "dev-kissos-auth-secret-change-me")
KISSOS_AUTH_ISSUER = os.getenv("KISSOS_AUTH_ISSUER", "kissos")
KISSOS_AUTH_AUDIENCE = os.getenv("KISSOS_AUTH_AUDIENCE", "kissos-extension")
TOKEN_TTL_SECONDS = int(os.getenv("KISSOS_AUTH_TOKEN_TTL", "21600"))

PROVIDER_CONFIG: dict[str, dict[str, str]] = {
    "google": {
        "client_id": os.getenv("GOOGLE_CLIENT_ID", ""),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET", ""),
        "token_url": "https://oauth2.googleapis.com/token",
        "userinfo_url": "https://openidconnect.googleapis.com/v1/userinfo",
    },
}

router = APIRouter(prefix="/api/auth")


class ExchangeRequest(BaseModel):
    provider: Provider
    code: str
    code_verifier: str
    redirect_uri: str


def _require_provider_config(provider: str) -> dict[str, str]:
    config = PROVIDER_CONFIG[provider]
    if not config["client_id"]:
        raise HTTPException(status_code=500, detail=f"{provider} client id is not configured")
    return config


def _normalize_user(provider: str, userinfo: dict[str, Any]) -> dict[str, str | None]:
    return {
        "id": str(userinfo.get("sub", "")),
        "email": userinfo.get("email"),
        "name": userinfo.get("name"),
    }


def issue_kissos_token(provider: str, user: dict[str, str | None]) -> tuple[str, int]:
    expires_at = int(time.time()) + TOKEN_TTL_SECONDS
    payload = {
        "iss": KISSOS_AUTH_ISSUER,
        "aud": KISSOS_AUTH_AUDIENCE,
        "sub": f"{provider}:{user['id']}",
        "provider": provider,
        "email": user.get("email"),
        "name": user.get("name"),
        "iat": int(time.time()),
        "exp": expires_at,
    }
    token = jwt.encode(payload, KISSOS_AUTH_SECRET, algorithm="HS256")
    return token, expires_at


def verify_kissos_token(token: str) -> dict[str, Any]:
    return jwt.decode(
        token,
        KISSOS_AUTH_SECRET,
        algorithms=["HS256"],
        audience=KISSOS_AUTH_AUDIENCE,
        issuer=KISSOS_AUTH_ISSUER,
    )



@router.get("/providers")
async def get_oauth_providers():
    return {
        provider: {
            "label": provider.title(),
            "clientId": config["client_id"],
            "authorizationEndpoint": "https://accounts.google.com/o/oauth2/v2/auth",
            "scope": "openid email profile",
        }
        for provider, config in PROVIDER_CONFIG.items()
    }
@router.post("/exchange")
async def exchange_oauth_code(request: ExchangeRequest):
    config = _require_provider_config(request.provider)
    token_payload = {
        "grant_type": "authorization_code",
        "client_id": config["client_id"],
        "code": request.code,
        "redirect_uri": request.redirect_uri,
        "code_verifier": request.code_verifier,
    }
    if config["client_secret"]:
        token_payload["client_secret"] = config["client_secret"]

    async with httpx.AsyncClient(timeout=10.0) as client:
        token_response = await client.post(config["token_url"], data=token_payload)
        if token_response.status_code >= 400:
            raise HTTPException(status_code=401, detail=token_response.text)

        provider_tokens = token_response.json()
        provider_access_token = provider_tokens.get("access_token")
        if not provider_access_token:
            raise HTTPException(status_code=401, detail="provider access token was missing")

        user_response = await client.get(
            config["userinfo_url"],
            headers={"Authorization": f"Bearer {provider_access_token}"},
        )
        if user_response.status_code >= 400:
            raise HTTPException(status_code=401, detail=user_response.text)

    user = _normalize_user(request.provider, user_response.json())
    if not user["id"]:
        raise HTTPException(status_code=401, detail="provider user id was missing")

    access_token, expires_at = issue_kissos_token(request.provider, user)
    return {
        "provider": request.provider,
        "accessToken": access_token,
        "user": user,
        "expiresAt": expires_at * 1000,
    }

