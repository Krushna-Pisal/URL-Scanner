FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && adduser --disabled-password --gecos "" --uid 10001 scanner \
    && mkdir -p /app/data \
    && chown -R scanner:scanner /app

COPY --chown=scanner:scanner scanner ./scanner
COPY --chown=scanner:scanner web ./web

ENV URL_SCANNER_DB_PATH=/app/data/scans.sqlite3
EXPOSE 8000
USER scanner

CMD ["python", "-m", "uvicorn", "scanner.api:app", "--host", "0.0.0.0", "--port", "8000"]
