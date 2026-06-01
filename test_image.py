"""
Test script: mimics what the frontend does — crops the image into each region
and POSTs each crop to /api/ocr concurrently.
"""
import base64
import sys
import json
import asyncio
import aiohttp
import cv2
import numpy as np

BASE_URL = "http://localhost:5000"

REGIONS = {
    "character": (0.0328, 0.0074, 0.3021, 0.0833),
    "watermark":  (0.0073, 0.0741, 0.1304, 0.1370),
    "forte":      (0.4057, 0.0222, 0.7422, 0.5917),
    "sequences":  (0.0703, 0.4787, 0.3318, 0.5843),
    "weapon":     (0.7542, 0.3843, 0.9828, 0.5843),
    "echo1":      (0.0125, 0.6019, 0.2042, 0.9843),
    "echo2":      (0.2057, 0.6019, 0.3974, 0.9843),
    "echo3":      (0.4016, 0.6019, 0.5938, 0.9843),
    "echo4":      (0.5969, 0.6019, 0.7891, 0.9843),
    "echo5":      (0.7911, 0.6019, 0.9833, 0.9843),
}

def crop_region(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    px1, py1 = round(x1 * w), round(y1 * h)
    px2, py2 = round(x2 * w), round(y2 * h)
    crop = img[py1:py2, px1:px2]
    _, buf = cv2.imencode(".png", crop)
    return base64.b64encode(buf).decode()

async def send_region(session, name, b64):
    headers = {"X-OCR-Region": name, "Content-Type": "application/json"}
    payload = {"image": b64}
    async with session.post(f"{BASE_URL}/api/ocr", json=payload, headers=headers) as resp:
        return name, resp.status, await resp.json()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python test_image.py <image-path>")
        sys.exit(2)

    image_path = sys.argv[1]
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: could not read {image_path}")
        sys.exit(1)
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    crops = {name: crop_region(img, x1, y1, x2, y2) for name, (x1, y1, x2, y2) in REGIONS.items()}

    async with aiohttp.ClientSession() as session:
        tasks = [send_region(session, name, b64) for name, b64 in crops.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for item in results:
        if isinstance(item, Exception):
            print(f"ERROR: {item}")
            continue
        name, status, body = item
        print(f"\n{'='*40}")
        print(f"Region: {name}  (HTTP {status})")
        print(json.dumps(body, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())
