FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY constraints.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=120 --retries 10 -c constraints.txt -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/
COPY data/processed/selected_features.json ./data/processed/selected_features.json

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]