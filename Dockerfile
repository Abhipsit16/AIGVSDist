FROM rust:1.90-bookworm AS builder
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-dev && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /buildenv
ENV PATH="/buildenv/bin:$PATH"
WORKDIR /build
COPY Cargo.toml Cargo.lock pyproject.toml LICENSE ./
COPY src ./src
RUN pip install 'maturin==1.15.0' && maturin build --release --locked --out /wheels

FROM python:3.11-slim-bookworm
RUN useradd --uid 10001 --create-home aigvs
COPY --from=builder /wheels /wheels
COPY requirements.lock /wheels/requirements.lock
RUN pip install --no-cache-dir -c /wheels/requirements.lock /wheels/*.whl && rm -rf /wheels
USER 10001:10001
ENV PYTHONUNBUFFERED=1 RAYON_NUM_THREADS=2
EXPOSE 8000
ENTRYPOINT ["aigvs"]
CMD ["serve"]
