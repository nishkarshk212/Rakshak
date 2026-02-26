FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# tesseract-ocr is used by the bot for OCR
# libgl1-mesa-glx and libglib2.0-0 are often needed for OpenCV (even headless in some environments)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "bot.py"]
