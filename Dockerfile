FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
COPY scripts/ ./scripts/
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--app-dir", "src"]
