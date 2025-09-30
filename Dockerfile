# Use a Python 3.10 base image so pandas wheels are available
FROM python:3.10.12-slim

# prevent creation of .pyc files and ensure stdout/stderr are not buffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system build tools (safe, small)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Start the app (Render will use this command)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
