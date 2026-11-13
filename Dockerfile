# Base images pinned by digest, so a rebuild months later uses exactly the same Python and uv
FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3

COPY --from=ghcr.io/astral-sh/uv:0.13.4@sha256:9f130421e930bfbdc563c00740466603f8723158e90297d07957c2294640262b /uv /uvx /bin/

ENV PATH="/code/.venv/bin:$PATH" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /code

# Dependencies first, in their own layer: rebuilt only when pyproject.toml or uv.lock change
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

# Then the package itself and the trained model
COPY src/ src/
COPY models/ models/
RUN uv sync --locked --no-dev

EXPOSE 9696
ENTRYPOINT ["uvicorn", "diabetes_readmission.predict:app", "--host", "0.0.0.0", "--port", "9696"]
