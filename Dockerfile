FROM python:3.13-slim
WORKDIR /app
ENV HF_HOME=/opt/hf PIP_NO_CACHE_DIR=1

COPY requirements-api.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-api.txt

COPY api.py pipeline.py ./
COPY static static
RUN python -c "from pipeline import VerificationPipeline; VerificationPipeline()"

ENV HF_HUB_OFFLINE=1
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
