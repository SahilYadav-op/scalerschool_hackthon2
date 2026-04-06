FROM python:3.11-slim

WORKDIR /app

# Ensure we're installing from the right place
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything from env root
COPY . .

# Set PYTHONPATH to include the app root so models/ is findable
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
