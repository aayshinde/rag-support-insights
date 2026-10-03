# API image. The FAISS index and models are not baked in: build the index once
# (python -m src.ingest) and mount ./index, or let docker-compose do it for you.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY api.py app.py ./
ENV PROVIDER=ollama CHAT_MODEL=llama3.2:3b EMBED_MODEL=nomic-embed-text
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
