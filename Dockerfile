FROM python:3.11-slim
WORKDIR /app

# PyPI's Linux torch wheel bundles ~6GB of CUDA libraries a GPU-less instance
# can never use; PyTorch's cpu index serves the same version without them.
# (PyPI's Windows wheel is already CPU-only, which is why local installs are
# small.) Pinned to the version running locally so the container matches, and
# placed before requirements.txt so the layer survives dependency changes.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
COPY scripts/ ./scripts/
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--app-dir", "src"]
