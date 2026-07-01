FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Mock demo by default — no API key required
CMD ["python", "demo.py", "--mock", "--yes"]
