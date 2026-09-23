"""HTTP entry point for MewHelp."""

from fastapi import FastAPI


app = FastAPI(title="MewHelp")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
