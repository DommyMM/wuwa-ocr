# =======================
# Builder stage
# =======================
FROM python:3.13.14-slim-trixie AS builder

WORKDIR /build

# Copy requirements and install as user (for easy copying later)
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# =======================
# Runtime stage
# =======================
FROM python:3.13.14-slim-trixie

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    ca-certificates \
    curl \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Debian's tesseract-ocr-eng ships tessdata_FAST (4.1 MB, integer-quantised LSTM),
# the least accurate of the three official models. Every accuracy number the OCR
# code was validated against came from tessdata_BEST (15.4 MB, float LSTM), which
# is what a local Windows install provides. Benchmarked on 300 r2-backup cards the
# two differ on 24% of watermark UID reads (fast produced a valid 9-digit UID on
# only 228/300; best on 300/300), and std/best agree 300/300, so this is the model
# the service must run. Pinned to the commit that introduced the file -- it has
# not changed since -- and checksum-verified so a build cannot silently regress.
ARG TESSDATA_BEST_COMMIT=9ddc24e750eec0994223a9edc3fcb434a2244f3b
ARG TESSDATA_BEST_ENG_SHA256=8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba
RUN curl -fsSL -o /usr/share/tesseract-ocr/5/tessdata/eng.traineddata \
      "https://github.com/tesseract-ocr/tessdata_best/raw/${TESSDATA_BEST_COMMIT}/eng.traineddata" \
    && echo "${TESSDATA_BEST_ENG_SHA256}  /usr/share/tesseract-ocr/5/tessdata/eng.traineddata" | sha256sum -c -

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application files
COPY Data /app/Data
COPY assets /app/assets
COPY *.py /app/

# Set Python path
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# OpenCV headless mode
ENV OPENCV_HEADLESS=1

# The service runs multiple OCR workers in parallel; keep each Tesseract call
# single-threaded to avoid OpenMP oversubscription.
ENV OMP_THREAD_LIMIT=1

EXPOSE 5000

ENTRYPOINT ["python", "server.py"]
