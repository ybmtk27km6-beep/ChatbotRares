FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY templates ./templates
COPY static ./static

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} --worker-class gthread --workers 1 --threads 8 --timeout 180 app:app"]
