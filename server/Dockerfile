# Fits a 512MB free tier. Measured 325MB RSS.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the model into the image. Free hosts have ephemeral disk, so without
# this every cold start re-downloads 111MB from HF and the first request
# after an idle spin-down takes ~30s instead of ~2s.
ENV HF_HOME=/app/.hf
RUN python -c "from huggingface_hub import hf_hub_download as d; \
    d('Xenova/siglip-base-patch16-224','onnx/text_model_int8.onnx'); \
    d('Xenova/siglip-base-patch16-224','tokenizer.json')"

COPY . .
ENV ORT_THREADS=2 PORT=8000
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
