# Rakshak

Telegram NSFW/Drugs guard bot with:
- NudeNet detection and skin exposure heuristics
- OCR on stickers and text normalization for obfuscations
- Profile photo and name-policy enforcement
- Admin notifications and configurable thresholds

Environment variables and tokens should be stored in a local `.env` and are ignored from git.

Setup
- Python 3.11+
- Install Python deps:

  ```
  pip install -r requirements.txt
  ```

- Optional: OCR needs the Tesseract binary installed on the system. If not installed, sticker text OCR is skipped gracefully.

Run
- Set TELEGRAM_BOT_TOKEN in .env
- Start the bot:

  ```
  python bot.py
  ```
