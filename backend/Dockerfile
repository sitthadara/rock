FROM python:3.13-slim

WORKDIR /app

# 1. Install dependencies
COPY backend/requirements.txt .
RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    --no-cache-dir -r requirements.txt

# 2. Copy the backend contents INTO the current WORKDIR (/app)
# This puts main.py exactly where uvicorn expects it
COPY backend/ .

# 3. Now 'main:app' will work because main.py is in /app/
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]