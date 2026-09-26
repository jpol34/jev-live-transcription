FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
COPY src ./src

# --break-system-packages: this image's Python is PEP 668 externally-managed (Debian base), but
# the container is dedicated to this one app -- there's no system package manager state to protect.
RUN uv pip install --system --break-system-packages --no-cache -e .

# Pre-bakes both GLiNER checkpoints into the image's HF cache so the pod's first tick doesn't pay
# a Hub download. map_location="cpu" is correct here regardless of config.GLINER_DEVICE -- the
# build container has no GPU, and these loaded objects are discarded immediately after populating
# the cache.
RUN python -c "\
from gliner import GLiNER; \
from jev_live_transcription import gliner_pipeline as g; \
GLiNER.from_pretrained(g.PII_MODEL_NAME, map_location='cpu'); \
GLiNER.from_pretrained(g.ZERO_SHOT_MODEL_NAME, map_location='cpu')"

# Copied last: neither affects package installation or the checkpoint cache above, so changing
# scripts or regenerating the corpus doesn't invalidate those expensive layers.
COPY scripts ./scripts
COPY output ./output
