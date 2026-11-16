"""ASGI entrypoint: `uvicorn autolearn.main:app`."""

from .api import create_app

app = create_app()
