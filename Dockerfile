FROM python:3.13.14-slim-trixie AS builder

WORKDIR /build

# --user install so the runtime stage can copy /root/.local whole
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.13.14-slim-trixie

WORKDIR /app

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

# Debian's tesseract-ocr-eng ships tessdata_fast, but the OCR code was validated on tessdata_best (the Windows install)
# Over 300 cards fast read a valid 9-digit UID on 228 while best read all 300
# Pinned to the commit that added the file and checksum-verified so a build can't silently regress
ARG TESSDATA_BEST_COMMIT=9ddc24e750eec0994223a9edc3fcb434a2244f3b
ARG TESSDATA_BEST_ENG_SHA256=8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba
RUN curl -fsSL -o /usr/share/tesseract-ocr/5/tessdata/eng.traineddata \
      "https://github.com/tesseract-ocr/tessdata_best/raw/${TESSDATA_BEST_COMMIT}/eng.traineddata" \
    && echo "${TESSDATA_BEST_ENG_SHA256}  /usr/share/tesseract-ocr/5/tessdata/eng.traineddata" | sha256sum -c -

COPY --from=builder /root/.local /root/.local

COPY Data /app/Data
COPY assets /app/assets
COPY *.py /app/

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

ENV OPENCV_HEADLESS=1

# OCR workers run in parallel, so each Tesseract call stays single-threaded to avoid OpenMP oversubscription
ENV OMP_THREAD_LIMIT=1

EXPOSE 5000

ENTRYPOINT ["python", "server.py"]
