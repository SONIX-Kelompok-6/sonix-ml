# 1. Base Image: Use a lightweight Python version to minimize image size
FROM python:3.11-slim

# 2. Environment Setup: Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 3. Working Directory: Set the operational folder inside the container
WORKDIR /app

# 4. Dependency Caching: Copy requirements file first
# This allows Docker to cache installed packages if requirements.txt hasn't changed
COPY requirements.txt .

# 5. Installation: Install dependencies
# --no-cache-dir reduces the final image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Source Code: Copy the application logic and model artifacts
COPY src/ src/
COPY model_artifacts/ model_artifacts/

# 7. Network: Expose port 8000 (Standard for FastAPI/Uvicorn)
EXPOSE 8000

# 8. Environment Variables: Set any necessary environment variables for the application
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# 9. Execution: Launch the application
# --host 0.0.0.0 is mandatory for containerized environments to accept external traffic
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]