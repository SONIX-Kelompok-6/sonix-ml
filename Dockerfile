# 1. Base Image
FROM python:3.11-slim

# 2. Environment Setup
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 3. Working Directory
WORKDIR /app

# 4. Dependency Caching
COPY requirements.txt .

# 5. Installation
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Source Code (Cukup ngopi 2 folder utama ini aja)
COPY src/ src/
COPY model_artifacts/ model_artifacts/

# 7. Network (Udah disamain ke 7860)
EXPOSE 7860

# 8. Environment Variables (Warning PYTHONPATH udah dibenerin di sini)
ENV PYTHONPATH="/app/src"

# 9. Execution
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "src.main:app", "--bind", "0.0.0.0:7860"]