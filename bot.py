import logging
import os
import cv2
try:
    import pytesseract  # optional; OCR will be disabled if unavailable
except Exception:
    class _PT:
        @staticmethod
        def image_to_string(*args, **kwargs):
            return ""
    pytesseract = _PT()
import re
import unicodedata
from telegram import Update, ChatPermissions
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, CommandHandler, filters
from nudenet import NudeDetector
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN") or "YOUR_BOT_TOKEN_HERE"
NSFW_THRESHOLD = float(os.getenv("NSFW_THRESHOLD", "0.6"))
SKIN_THRESHOLD = float(os.getenv("SKIN_THRESHOLD", "0.28"))
MAX_STRIKES = int(os.getenv("MAX_STRIKES", "3"))
BAD_WORDS = [w.strip() for w in os.getenv("BAD_WORDS", "sex,nude,porn,xxx,18+,nsfw,boobs,tits,ass,anal,blowjob,bj,cum,cumming,ejaculate,deepthroat,handjob,jerk,orgasm,strip,hooker,escort,prostitute,slut,whore,dick,cock,penis,balls,scrotum,pussy,vagina,clit,clitoris,twerk,69,fellatio,cunnilingus,camgirl,onlyfans,hardcore,softcore,bdsm,rape,incest,pedo,pedophile,loli,hentai,cam,sexchat,camshow,erotic,erotica,drug,drugs,weed,marijuana,cannabis,pot,ganja,hash,hashish,thc,420,coke,cocaine,crack,heroin,meth,methamphetamine,ice,crystal,lsd,acid,ketamine,mdma,ecstasy,molly,shrooms,psilocybin,opium,opioid,oxy,oxycodone,xanax,adderall,fentanyl,lean,codeine,promethazine,dope,stoned,high,speed,smack").split(",") if w.strip()]
WARN_ADMINS = os.getenv("WARN_ADMINS", "1") not in ("0", "false", "False", "")
GROUP_WARN_TEXT = os.getenv("GROUP_WARN_TEXT", "nikal yha se")
STICKER_SET_BLACKLIST = [s.strip() for s in os.getenv("STICKER_SET_BLACKLIST", "Shiva1234422_by_fStikBot").split(",") if s.strip()]
STICKER_SET_SUFFIX_BLACKLIST = [s.strip() for s in os.getenv("STICKER_SET_SUFFIX_BLACKLIST", "_by_fStikBot").split(",") if s.strip()]
NAME_WARN_ENABLED = os.getenv("NAME_WARN_ENABLED", "1") not in ("0", "false", "False", "")
NAME_WARN_ONCE = os.getenv("NAME_WARN_ONCE", "0") in ("1", "true", "True")
NAME_GROUP_WARN_TEXT = os.getenv("NAME_GROUP_WARN_TEXT", "chup ")
PFP_CHECK_ENABLED = os.getenv("PFP_CHECK_ENABLED", "1") not in ("0", "false", "False", "")
PFP_GROUP_WARN_TEXT = os.getenv("PFP_GROUP_WARN_TEXT", "niklo ji")
PFP_NSFW_THRESHOLD = float(os.getenv("PFP_NSFW_THRESHOLD", "0.45"))
PFP_SKIN_THRESHOLD = float(os.getenv("PFP_SKIN_THRESHOLD", "0.22"))
PHOTO_GROUP_WARN_TEXT = os.getenv("PHOTO_GROUP_WARN_TEXT", "photo accha daal")
BOTTOM_SKIN_THRESHOLD = float(os.getenv("BOTTOM_SKIN_THRESHOLD", "0.35"))
PFP_BOTTOM_SKIN_THRESHOLD = float(os.getenv("PFP_BOTTOM_SKIN_THRESHOLD", "0.30"))
DRUG_KEYWORDS = [w.strip() for w in os.getenv("DRUG_KEYWORDS", "weed,ganja,cocaine,heroin,lsd,mdma,meth,opium,charas,hash,drug,narcotic,smack,brown sugar,ecstasy").split(",") if w.strip()]
NSFW_LABELS = [
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "BUTTOCKS_EXPOSED",
]

logging.basicConfig(level=logging.INFO)
detector = NudeDetector()
user_strikes = {}
_name_warned = set()
_pfp_blocked = set()


def get_max_score(result):
    score = 0.0
    for item in result or []:
        try:
            s = float(item.get("score", 0.0))
        except Exception:
            s = 0.0
        if s > score:
            score = s
    return score


def has_nsfw_label(result):
    for item in result or []:
        try:
            lbl = str(item.get("label") or item.get("class") or "")
            sc = float(item.get("score", 0.0))
        except Exception:
            lbl = ""
            sc = 0.0
        if lbl in NSFW_LABELS and sc >= NSFW_THRESHOLD:
            return True, lbl, sc
    return False, None, 0.0


_LEET_MAP = str.maketrans({
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "8": "b",
    "@": "a",
    "$": "s",
})


def _normalize_variants(s: str):
    x = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii", "ignore")
    x = x.lower().translate(_LEET_MAP)
    a = x
    b = re.sub(r"[^a-z0-9]+", "", x)
    return a, b


def contains_bad_text(text: str | None):
    if not text:
        return False
    t1, t2 = _normalize_variants(text)
    for w in BAD_WORDS:
        w1 = w.lower()
        w2 = re.sub(r"[^a-z0-9]+", "", w1.translate(_LEET_MAP))
        if re.search(rf"\b{re.escape(w1)}\b", t1):
            return True
        if w2 and w2 in t2:
            return True
    return False


def contains_drug_content(text: str | None):
    if not text:
        return False
    a, b = _normalize_variants(text)
    for w in DRUG_KEYWORDS:
        w1 = w.lower()
        w2 = re.sub(r"[^a-z0-9]+", "", w1.translate(_LEET_MAP))
        if re.search(rf"\b{re.escape(w1)}\b", a):
            return True
        if w2 and w2 in b:
            return True
    return False


def skin_ratio_from_bgr(img):
    if img is None or img.size == 0:
        return 0.0
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lower = (0, 133, 77)
    upper = (255, 173, 127)
    mask = cv2.inRange(ycrcb, lower, upper)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    total = img.shape[0] * img.shape[1]
    if total <= 0:
        return 0.0
    ratio = float(cv2.countNonZero(mask)) / float(total)
    return ratio


def bottom_skin_ratio_from_bgr(img, frac=0.4):
    if img is None or img.size == 0:
        return 0.0
    h = img.shape[0]
    y0 = int(h * (1.0 - frac))
    roi = img[y0:h, :]
    return skin_ratio_from_bgr(roi)


async def get_user_profile_pic(context: ContextTypes.DEFAULT_TYPE, user_id: int, msg_id: int):
    try:
        photos = await context.bot.get_user_profile_photos(user_id, limit=1)
        if photos.total_count == 0:
            return None
        file = await context.bot.get_file(photos.photos[0][-1].file_id)
        path = f"profile_{msg_id}.jpg"
        await file.download_to_drive(path)
        return path
    except Exception:
        return None


async def warn_admins(update: Update, context: ContextTypes.DEFAULT_TYPE, caption: str, pfp_path: str | None):
    if not WARN_ADMINS:
        return
    try:
        admins = await context.bot.get_chat_administrators(update.effective_chat.id)
    except Exception:
        return
    for adm in admins:
        if adm.user.is_bot:
            continue
        try:
            if pfp_path and os.path.exists(pfp_path):
                with open(pfp_path, "rb") as fp:
                    await context.bot.send_photo(adm.user.id, photo=fp, caption=caption)
            else:
                await context.bot.send_message(adm.user.id, caption)
        except Exception:
            continue


async def punish_user(update: Update, context: ContextTypes.DEFAULT_TYPE, reason: str | None = None):
    user = update.effective_user
    chat_id = update.effective_chat.id
    user_strikes[user.id] = user_strikes.get(user.id, 0) + 1
    strikes = user_strikes[user.id]
    pfp = await get_user_profile_pic(context, user.id, update.message.message_id)
    caption = (
        f"🚨 NSFW Violation Detected\n\n"
        f"👤 Name: {user.full_name}\n"
        f"🔗 Username: @{user.username if user.username else 'None'}\n"
        f"🆔 ID: {user.id}\n"
        f"⚠️ Strikes: {strikes}/{MAX_STRIKES}"
    )
    if reason:
        caption = caption + f"\n📝 Reason: {reason}"
    # Group: send short warning only
    try:
        await context.bot.send_message(chat_id, GROUP_WARN_TEXT)
    except Exception:
        pass
    if WARN_ADMINS:
        try:
            await warn_admins(update, context, caption, pfp)
        except Exception:
            pass
    if pfp and os.path.exists(pfp):
        try:
            os.remove(pfp)
        except Exception:
            pass


async def maybe_warn_for_name(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not NAME_WARN_ENABLED:
        return False
    user = update.effective_user
    chat_id = update.effective_chat.id
    key = (chat_id, user.id)
    if NAME_WARN_ONCE and key in _name_warned:
        return False
    parts = []
    if user.full_name:
        parts.append(user.full_name)
    if user.username:
        parts.append(user.username)
    s = " ".join(parts)
    if contains_bad_text(s):
        if NAME_WARN_ONCE:
            _name_warned.add(key)
        try:
            await update.message.delete()
        except Exception:
            pass
        try:
            await context.bot.restrict_chat_member(chat_id, user.id, permissions=ChatPermissions(can_send_messages=False))
        except Exception:
            pass
        try:
            await context.bot.send_message(chat_id, NAME_GROUP_WARN_TEXT)
        except Exception:
            pass
        if WARN_ADMINS:
            cap = f"🚨 NSFW Name Detected\n\n👤 Name: {user.full_name}\n🔗 Username: @{user.username if user.username else 'None'}\n🆔 ID: {user.id}"
            try:
                await warn_admins(update, context, cap, None)
            except Exception:
                pass
        return True
    return False
    if strikes >= MAX_STRIKES:
        try:
            await context.bot.restrict_chat_member(chat_id, user.id, permissions=ChatPermissions(can_send_messages=False))
        except Exception as e:
            logging.warning("Mute error: %s", e)


async def maybe_enforce_pfp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not PFP_CHECK_ENABLED:
        return False
    user = update.effective_user
    chat_id = update.effective_chat.id
    if (chat_id, user.id) in _pfp_blocked:
        try:
            await update.message.delete()
        except Exception:
            pass
        try:
            await context.bot.restrict_chat_member(chat_id, user.id, permissions=ChatPermissions(can_send_messages=False))
        except Exception:
            pass
        try:
            await context.bot.send_message(chat_id, PFP_GROUP_WARN_TEXT)
        except Exception:
            pass
        return True
    path = await get_user_profile_pic(context, user.id, update.message.message_id)
    if not path or not os.path.exists(path):
        return False
    try:
        img = cv2.imread(path)
        up_path = None
        if img is not None:
            h, w = img.shape[:2]
            if max(h, w) < 256:
                scale = 512.0 / float(max(1, max(h, w)))
                new_w = int(w * scale)
                new_h = int(h * scale)
                up = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                up_path = f"{path}.up.jpg"
                cv2.imwrite(up_path, up)
        target_path = up_path if up_path else path
        res = detector.detect(target_path)
        sc = get_max_score(res)
        hit, hit_lbl, hit_sc = has_nsfw_label(res)
        img2 = cv2.imread(target_path)
        sr = skin_ratio_from_bgr(img2)
        bsr = bottom_skin_ratio_from_bgr(img2)
        if hit or sc >= PFP_NSFW_THRESHOLD or sr >= PFP_SKIN_THRESHOLD or bsr >= PFP_BOTTOM_SKIN_THRESHOLD:
            _pfp_blocked.add((chat_id, user.id))
            try:
                await update.message.delete()
            except Exception:
                pass
            try:
                await context.bot.restrict_chat_member(chat_id, user.id, permissions=ChatPermissions(can_send_messages=False))
            except Exception:
                pass
            try:
                await context.bot.send_message(chat_id, PFP_GROUP_WARN_TEXT)
            except Exception:
                pass
            if WARN_ADMINS:
                cap = f"🚨 NSFW Profile Photo\n\n👤 Name: {user.full_name}\n🔗 Username: @{user.username if user.username else 'None'}\n🆔 ID: {user.id}"
                try:
                    await warn_admins(update, context, cap, target_path)
                except Exception:
                    pass
            return True
    finally:
        try:
            if os.path.exists(path):
                os.remove(path)
            if 'up_path' in locals() and up_path and os.path.exists(up_path):
                os.remove(up_path)
        except Exception:
            pass
    return False


async def check_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    user = update.effective_user
    member = await context.bot.get_chat_member(update.effective_chat.id, user.id)
    if member.status in ["administrator", "creator"]:
        return
    if await maybe_warn_for_name(update, context):
        return
    if await maybe_enforce_pfp(update, context):
        return
    if contains_bad_text(getattr(message, "caption", None)) or contains_drug_content(getattr(message, "caption", None)):
        await message.delete()
        await punish_user(update, context, "Photo caption text")
        return
    file = await message.photo[-1].get_file()
    path = f"photo_{message.message_id}.jpg"
    await file.download_to_drive(path)
    try:
        result = detector.detect(path)
        score = get_max_score(result)
        hit, hit_lbl, hit_sc = has_nsfw_label(result)
        img = cv2.imread(path)
        sr = skin_ratio_from_bgr(img)
        bsr = bottom_skin_ratio_from_bgr(img)
        if hit or score >= NSFW_THRESHOLD or sr >= SKIN_THRESHOLD or bsr >= BOTTOM_SKIN_THRESHOLD:
            await message.delete()
            try:
                await context.bot.send_message(update.effective_chat.id, PHOTO_GROUP_WARN_TEXT)
            except Exception:
                pass
            return
    finally:
        if os.path.exists(path):
            os.remove(path)


async def check_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    user = update.effective_user
    member = await context.bot.get_chat_member(update.effective_chat.id, user.id)
    if member.status in ["administrator", "creator"]:
        return
    if await maybe_warn_for_name(update, context):
        return
    if await maybe_enforce_pfp(update, context):
        return
    if contains_bad_text(getattr(message, "caption", None)) or contains_drug_content(getattr(message, "caption", None)):
        await message.delete()
        await punish_user(update, context, "Video caption text")
        return
    file = await message.video.get_file()
    path = f"video_{message.message_id}.mp4"
    await file.download_to_drive(path)
    try:
        cap = cv2.VideoCapture(path)
        frame_count = 0
        nsfw_detected = False
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 30 == 0:
                temp = f"frame_{frame_count}.jpg"
                try:
                    cv2.imwrite(temp, frame)
                    result = detector.detect(temp)
                    score = get_max_score(result)
                    hit, hit_lbl, hit_sc = has_nsfw_label(result)
                    sr = skin_ratio_from_bgr(frame)
                    bsr = bottom_skin_ratio_from_bgr(frame)
                    if hit or score >= NSFW_THRESHOLD or sr >= SKIN_THRESHOLD or bsr >= BOTTOM_SKIN_THRESHOLD:
                        nsfw_detected = True
                        reason_local = []
                        if hit:
                            reason_local.append(f"Label={hit_lbl}({hit_sc:.2f})")
                        if score >= NSFW_THRESHOLD:
                            reason_local.append(f"NSFW score={score:.2f}")
                        if sr >= SKIN_THRESHOLD:
                            reason_local.append(f"Skin ratio={sr:.2f}")
                        if bsr >= BOTTOM_SKIN_THRESHOLD:
                            reason_local.append(f"Bottom skin ratio={bsr:.2f}")
                        trigger_reason = ", ".join(reason_local) if reason_local else "NSFW"
                        break
                finally:
                    if os.path.exists(temp):
                        os.remove(temp)
            frame_count += 1
        cap.release()
        if nsfw_detected:
            await message.delete()
            try:
                await punish_user(update, context, trigger_reason)
            except Exception:
                await punish_user(update, context, "NSFW")
    finally:
        if os.path.exists(path):
            os.remove(path)


async def check_sticker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    user = update.effective_user
    member = await context.bot.get_chat_member(update.effective_chat.id, user.id)
    if member.status in ["administrator", "creator"]:
        return
    if await maybe_warn_for_name(update, context):
        return
    if await maybe_enforce_pfp(update, context):
        return
    set_name = getattr(message.sticker, "set_name", None)
    if set_name and (set_name in STICKER_SET_BLACKLIST or any(set_name.endswith(suf) for suf in STICKER_SET_SUFFIX_BLACKLIST)):
        await message.delete()
        await punish_user(update, context, f"Sticker set {set_name}")
        return
    file = await message.sticker.get_file()
    ext = ".webp"
    if getattr(message.sticker, "is_video", False):
        ext = ".webm"
    elif getattr(message.sticker, "is_animated", False):
        ext = ".tgs"
    path = f"sticker_{message.message_id}{ext}"
    await file.download_to_drive(path)
    try:
        jpg_path = path.rsplit(".", 1)[0] + ".jpg"
        if getattr(message.sticker, "is_video", False):
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return
            cv2.imwrite(jpg_path, frame)
        elif getattr(message.sticker, "is_animated", False):
            return
        else:
            try:
                from PIL import Image
                im = Image.open(path)
                if im.mode in ("RGBA", "LA"):
                    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                    bg.paste(im, mask=im.split()[-1])
                    im = bg.convert("RGB")
                else:
                    im = im.convert("RGB")
                im.save(jpg_path, "JPEG")
            except Exception:
                img_fallback = cv2.imread(path)
                if img_fallback is None:
                    return
                cv2.imwrite(jpg_path, img_fallback)

        img = cv2.imread(jpg_path)
        if img is None:
            return
        sr = skin_ratio_from_bgr(img)
        bsr = bottom_skin_ratio_from_bgr(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        enlarged = cv2.resize(th, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        inv = cv2.bitwise_not(enlarged)
        try:
            text1 = pytesseract.image_to_string(enlarged, config="--psm 6")
        except Exception:
            text1 = ""
        try:
            text2 = pytesseract.image_to_string(inv, config="--psm 6")
        except Exception:
            text2 = ""
        text = f"{text1} {text2}".strip()
        nuderes = detector.detect(jpg_path)
        nscore = get_max_score(nuderes)
        hit, hit_lbl, hit_sc = has_nsfw_label(nuderes)
        if contains_bad_text(text) or contains_drug_content(text) or hit or sr >= SKIN_THRESHOLD or nscore >= NSFW_THRESHOLD or bsr >= BOTTOM_SKIN_THRESHOLD:
            await message.delete()
            rs = []
            if contains_bad_text(text):
                rs.append("Sticker text")
            if contains_drug_content(text):
                rs.append("Drug text")
            if hit:
                rs.append(f"Label={hit_lbl}({hit_sc:.2f})")
            if nscore >= NSFW_THRESHOLD:
                rs.append(f"NSFW score={nscore:.2f}")
            if sr >= SKIN_THRESHOLD:
                rs.append(f"Skin ratio={sr:.2f}")
            if bsr >= BOTTOM_SKIN_THRESHOLD:
                rs.append(f"Bottom skin ratio={bsr:.2f}")
            await punish_user(update, context, ", ".join(rs) if rs else "NSFW")
    finally:
        if os.path.exists(path):
            os.remove(path)
        jp = path.rsplit(".", 1)[0] + ".jpg"
        if os.path.exists(jp):
            os.remove(jp)


async def check_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    user = update.effective_user
    member = await context.bot.get_chat_member(update.effective_chat.id, user.id)
    if member.status in ["administrator", "creator"]:
        return
    if await maybe_warn_for_name(update, context):
        return
    if await maybe_enforce_pfp(update, context):
        return
    # Skip media messages; handled by other callbacks
    if getattr(message, "photo", None) or getattr(message, "video", None) or getattr(message, "sticker", None):
        return
    txt = message.text or message.caption
    if contains_bad_text(txt) or contains_drug_content(txt):
        await message.delete()
        await punish_user(update, context)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🛡️ ULTRA++ NSFW Guard Active\nPhotos • Videos • Sticker Text protected.")


def main():
    if not BOT_TOKEN or BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN or BOT_TOKEN environment variable")
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, check_photo))
    app.add_handler(MessageHandler(filters.VIDEO, check_video))
    app.add_handler(MessageHandler(filters.Sticker.ALL, check_sticker))
    app.add_handler(MessageHandler((filters.TEXT & ~filters.COMMAND) | filters.CaptionRegex(".*"), check_text))
    app.run_polling()


if __name__ == "__main__":
    main()
