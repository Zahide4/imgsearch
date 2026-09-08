# HF Spaces free tier: 2 vCPU, 16GB RAM, no credit card, Docker SDK.
# This is the production FastAPI service -- same code you'd run on a
# €12/mo Hetzner box, just hosted somewhere free.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Bake the text tower into the image so cold starts don't download it
RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-16-SigLIP', pretrained='webli'); open_clip.get_tokenizer('ViT-B-16-SigLIP')"

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
