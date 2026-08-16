FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev --no-group benchmark \
    && useradd --create-home --uid 10001 rta \
    && mkdir /state \
    && chown rta:rta /state

USER rta
VOLUME ["/state"]

ENTRYPOINT ["reliable-task-agent"]
CMD ["--help"]
