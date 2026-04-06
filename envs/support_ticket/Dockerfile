FROM python:3.11-slim

# Create a non-root user for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

WORKDIR /app

# Copy requirements first for better caching
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the application to /app
COPY --chown=user . .

# Set PYTHONPATH to /app so 'import models' and 'import server.X' work
ENV PYTHONPATH=/app

# Expose port 8000 for consistency with openenv.yaml
EXPOSE 8000

# Start the FastAPI server using the specified port
# Start the FastAPI server on port 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
