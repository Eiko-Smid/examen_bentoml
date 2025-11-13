from __future__ import annotations
import os, time

import numpy as np

import bentoml
from bentoml.models import BentoModel

from pydantic import BaseModel, Field, ConfigDict
from typing import Annotated, Tuple
from fastapi import Header, HTTPException

import jwt  # PyJWT

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from dotenv import load_dotenv

#________________________________________________________________________________________________________
# Definitions
#________________________________________________________________________________________________________

# Load environment variables from .env file
load_dotenv()

MODEL_NAME = "student_admissions_predictor"
MODEL_VERSION = ":latest"

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALG = "HS256"
JWT_TTL_SECONDS = 30 * 60  # 30 min

#________________________________________________________________________________________________________
# Utils
#________________________________________________________________________________________________________

def load_model(
        name: str,
        version: str = ":latest"
) -> Tuple[object, BentoModel]:
    '''
    Loads latest model by given name and returns model and the Bento model.
    '''
    tag = name + version
    model_info = bentoml.sklearn.get(tag)
    model = model_info.load_model()
    print(f"\nLoaded model: {model_info.tag}")
    return model, model_info


#________________________________________________________________________________________________________
# API security
#________________________________________________________________________________________________________

class JWTAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request, call_next):
        # public endpoints (health + login)
        if request.url.path in {"/healthz", "/livez", "/readyz", "/login"}:
            return await call_next(request)

        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            return JSONResponse({"error": "Missing Bearer token"}, status_code=401)

        token = auth.split(" ", 1)[1].strip()
        try:
            jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        except jwt.ExpiredSignatureError:
            return JSONResponse({"error": "Token expired"}, status_code=401)
        except jwt.InvalidTokenError:
            return JSONResponse({"error": "Invalid token"}, status_code=401)

        return await call_next(request)
    

#________________________________________________________________________________________________________
# Schemas
#________________________________________________________________________________________________________

class InputModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    gre: float = 0.389
    toefl: float = 0.602
    uni_rating: float = -0.098
    sop: float = 0.126
    lor: float = 0.564
    cgpa: float = 0.415
    res: float = 0.895


#________________________________________________________________________________________________________
# Define Bento ML API
#________________________________________________________________________________________________________


@bentoml.service  # optional: (resources={"cpu": "1"})
class StudAdmService:
    def __init__(self) -> None:
        self.model, self.model_info = load_model(
            name=MODEL_NAME,
            version=MODEL_VERSION
        )
        print(f"\nLoaded model: {self.model_info.tag}\n")
    

    # ---- LOGIN: returns a JWT on valid creds ----
    @bentoml.api(route="/login")
    def login(self, username: str, password: str) -> dict:
        # DEMO ONLY — use a user DB / IdP in production
        users = {"alice": "s3cret", "bob": "hunter2"}
        if users.get(username) != password:
            return {"error": "Invalid credentials"}

        now = int(time.time())
        payload = {
            "sub": username,
            "iat": now,
            "exp": now + JWT_TTL_SECONDS,
            "scope": ["predict"],  # example scope
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
        return {"access_token": token, "token_type": "bearer", "expires_in": JWT_TTL_SECONDS}


    # Prediction endpoint
    @bentoml.api(route="/predict")
    def prediction(self, input_data: InputModel) -> dict:
        input_series = np.array([
            input_data.gre, input_data.toefl, input_data.uni_rating,
            input_data.sop, input_data.lor, input_data.cgpa,
            input_data.res
        ])

        # Predict model
        pred = self.model.predict(input_series.reshape(1, -1))

        return {
            "prediction": pred.tolist()
        }


# ---------- Attach middleware (class-level, 1.4 style) ----------
StudAdmService.add_asgi_middleware(JWTAuthMiddleware)
