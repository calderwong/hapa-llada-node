from fastapi import HTTPException, Request


def verify_request_token(request: Request, expected_token: str) -> None:
    token = None

    auth = request.headers.get("authorization")
    if auth:
        parts = auth.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]

    if token is None:
        token = request.query_params.get("token")

    if not token or token != expected_token:
        raise HTTPException(status_code=401, detail="Unauthorized")
