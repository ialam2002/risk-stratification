FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY artifacts/ artifacts/
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "risk_strat.api:app", "--host", "0.0.0.0", "--port", "8000"]
