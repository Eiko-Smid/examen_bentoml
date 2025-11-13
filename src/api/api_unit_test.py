import os
import time
import jwt
import pytest
import requests

from typing import Any, Dict


#_________________________________________________________________________________________________________
# Definitions
#_________________________________________________________________________________________________________

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALG = "HS256"

TOKEN = os.getenv("JWT_SECRET")

PORT = 3000
LOGIN_URL = f"http://localhost:{PORT}/login"
LOGIN_HEADER = {
    "Content-Type": "application/json",
}
TEST_USER = {
    "username":"alice",
    "password":"s3cret"
}
WRONG_USER = {
    "username":"alice",
    "password":"wrongpassword"
}

LOGIN_TIMEOUT = 10

PREDICT_URL = f"http://localhost:{PORT}/predict"
PREDICT_HEADER = {
    "accept": "application/json",
    "Content-Type": "application/json",
}
PREDICT_HEADER_WITHOUT_TOKEN = {
    "accept": "application/json",
    "Content-Type": "application/json",
}
MODEL_INPUT: Dict[str, Any] = {
     "input_data": {
        "gre": 0.389,
        "toefl": 0.602,
        "uni_rating": -0.098,
        "sop": 0.126,
        "lor": 0.564,
        "cgpa": 0.415,
        "res": 0.895
        }
}

INVALID_MODEL_INPUT: Dict[str, Any] = {
     "input_data": {
        "gre": "wrong",
        "toefl": 0.602,
        "uni_rating": -0.098,
        "sop": 0.126,
        "lor": 0.564,
        "cgpa": 0.415,
        "res": 0.895
        }
}


PREDICT_TIMEOUT = 10


#_________________________________________________________________________________________________________
# Fixtures and helpers
#_________________________________________________________________________________________________________

@pytest.fixture(scope="session")
def login_ok_token() -> str:
    """Obtain a real JWT from /login using valid credentials."""
    r = requests.post(
        url=LOGIN_URL,
        headers=LOGIN_HEADER,
        json=TEST_USER,
        timeout=LOGIN_TIMEOUT,
    )
    assert r.status_code == 200, f"/login failed: {r.status_code} {r.text}"
    data = r.json()
    assert "access_token" in data, f"/login no token: {data}"
    return data["access_token"]


@pytest.fixture(scope="session")
def expired_token() -> str:
    """Creates expired token."""
    now = int(time.time())
    payload = {"sub": "alice", "iat": now - 4000, "exp": now - 3600, "scope": ["predict"]}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


@pytest.fixture(scope="session")
def invalid_token() -> str:
    """Create invalid token based on wrong secret key."""
    now = int(time.time())
    payload = {"sub": "alice", "iat": now, "exp": now + 3600, "scope": ["predict"]}
    # sign with wrong secret
    return jwt.encode(payload, "wrong-secret", algorithm=JWT_ALG)


def update_predict_header(token: str, token_key: str="Authorization") -> None:
    global PREDICT_HEADER
    PREDICT_HEADER[token_key] = f"Bearer {token}"
    

# _________________________________________________________________________________________________________
# Token authorization tests
# _________________________________________________________________________________________________________

def test_auth_missing_token():
    r = requests.post(
        url=PREDICT_URL,
        headers=PREDICT_HEADER_WITHOUT_TOKEN,
        json=MODEL_INPUT,
        timeout=PREDICT_TIMEOUT
    )

    assert r.status_code == 401, f"Expected 401 for missing token, got {r.status_code} {r.text}"


def test_auth_invalid_token(invalid_token: str):
    # Define header with invalid token
    update_predict_header(
        token=invalid_token
    )
    r = requests.post(
        url=PREDICT_URL,
        headers=PREDICT_HEADER,
        json=MODEL_INPUT,
        timeout=PREDICT_TIMEOUT
    )
    assert r.status_code == 401, f"Expected 401 for invalid token, got {r.status_code} {r.text}"


def test_auth_expired_token(expired_token: str):
    update_predict_header(
        token=expired_token
    )
    r = requests.post(
        url=PREDICT_URL,
        headers=PREDICT_HEADER,
        json=MODEL_INPUT,
        timeout=PREDICT_TIMEOUT,
    )
    assert r.status_code == 401, f"Expected 401 for expired token, got {r.status_code} {r.text}"



def test_auth_valid_token(login_ok_token: str):
    update_predict_header(
        token=login_ok_token
    )
    r = requests.post(
        url=PREDICT_URL,
        headers=PREDICT_HEADER,
        json=MODEL_INPUT,
        timeout=PREDICT_TIMEOUT,
    )
    # Either a 200 with prediction or a 422 if your service validates differently
    assert r.status_code == 200, f"Expected 200 with valid token, got {r.status_code} {r.text}"
    data = r.json()
    assert "prediction" in data, f"Missing 'prediction' key: {data}"
    assert isinstance(data["prediction"], list)


#_________________________________________________________________________________________________________
# Login API tests
#_________________________________________________________________________________________________________

def test_login_success_returns_token():
    r = requests.post(
        url=LOGIN_URL,
        headers=LOGIN_HEADER,
        json=TEST_USER,
        timeout=LOGIN_TIMEOUT,
    )
    assert r.status_code == 200, f"/login should be 200 on success, got {r.status_code} {r.text}"
    data = r.json()
    assert "access_token" in data and data.get("token_type") == "bearer"


def test_login_bad_creds():
    r = requests.post(
        url=LOGIN_URL,
        headers=LOGIN_HEADER,
        json=WRONG_USER,
        timeout=LOGIN_TIMEOUT,
    )
    # If you changed login to return 401 on bad creds, assert 401:
    assert r.status_code == 401, f"Expected 401 for bad creds, got {r.status_code} {r.text}"



#_________________________________________________________________________________________________________
# Login API tests
#_________________________________________________________________________________________________________


def test_predict_with_valid_token_returns_prediction(login_ok_token: str):
    update_predict_header(
        token=login_ok_token
    )
    r = requests.post(
        url=PREDICT_URL,
        headers=PREDICT_HEADER,
        json=MODEL_INPUT,
        timeout=PREDICT_TIMEOUT,
    )
    assert r.status_code == 200, f"Prediction failed: {r.status_code} {r.text}"
    data = r.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], list)



def test_predict_invalid_input_422(login_ok_token: str):
    update_predict_header(
        token=login_ok_token
    )

    r = requests.post(
        url=PREDICT_URL,
        headers=PREDICT_HEADER,
        json=INVALID_MODEL_INPUT,
        timeout=PREDICT_TIMEOUT,
    )
    assert r.status_code in (400, 422), f"Expected validation error, got {r.status_code} {r.text}"
