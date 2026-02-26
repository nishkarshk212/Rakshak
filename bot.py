import logging
import os
import cv2
import sys
from collections import namedtuple
try:
    import uvloop
    uvloop.install()
except Exception:
    pass
try:
    import pytesseract  # optional; OCR will be disabled if unavailable
except Exception:
    class _PT:
        @staticmethod
        def image_to_string(*args, **kwargs):
            return ""
    pytesseract = _PT()
import re
import asyncio
import unicodedata
import json
import sqlite3
from telegram import Update, ChatPermissions, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, CommandHandler, filters, CallbackQueryHandler
from nudenet import NudeDetector
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN") or "YOUR_BOT_TOKEN_HERE"
NSFW_THRESHOLD = float(os.getenv("NSFW_THRESHOLD", "0.75"))
SKIN_THRESHOLD = float(os.getenv("SKIN_THRESHOLD", "0.45"))
MAX_STRIKES = int(os.getenv("MAX_STRIKES", "5"))
BAD_WORDS = [w.strip() for w in os.getenv("BAD_WORDS", "sex,nude,porn,xxx,18+,nsfw,boobs,tits,ass,anal,blowjob,bj,cum,cumming,ejaculate,deepthroat,handjob,jerk,orgasm,strip,hooker,escort,prostitute,slut,whore,dick,cock,penis,balls,scrotum,pussy,vagina,clit,clitoris,twerk,69,fellatio,cunnilingus,camgirl,onlyfans,hardcore,softcore,bdsm,rape,incest,pedo,pedophile,loli,hentai,cam,sexchat,camshow,erotic,erotica,drug,drugs,weed,marijuana,cannabis,pot,ganja,hash,hashish,thc,420,coke,cocaine,crack,heroin,meth,methamphetamine,ice,crystal,lsd,acid,ketamine,mdma,ecstasy,molly,shrooms,psilocybin,opium,opioid,oxy,oxycodone,xanax,adderall,fentanyl,lean,codeine,promethazine,dope,stoned,high,speed,smack").split(",") if w.strip()]

HINDI_NSFW_WORDS = [ 
     "नंगा", "नंगी", "नग्न", "सेक्स", "संभोग", "चुदाई", "चोद", 
     "लंड", "लौड़ा", "लौंडा", "गांड", "गांड़", "चूत", "चूतिया", 
     "रंडी", "वेश्या", "पोर्न", "अश्लील", "कामुक", "हॉट" 
 ] 
 
HINGLISH_NSFW_WORDS = [ 
     "nanga", "nangi", "sex", "chudai", "chod", "lund", 
     "gaand", "gand", "chut", "randi", "porn", "ashleel", "hot" 
 ] 

WARN_ADMINS = os.getenv("WARN_ADMINS", "1") not in ("0", "false", "False", "")
# Stylish Font Mapping Helper
def to_stylish(text):
    # Mapping for a Bold Sans-Serif style
    mapping = {
        'A': '𝐀', 'B': '𝐁', 'C': '𝐂', 'D': '𝐃', 'E': '𝐄', 'F': '𝐅', 'G': '𝐆', 'H': '𝐇', 'I': '𝐈', 'J': '𝐉', 'K': '𝐊', 'L': '𝐋', 'M': '𝐌', 'N': '𝐍', 'O': '𝐎', 'P': '𝐏', 'Q': '𝐐', 'R': '𝐑', 'S': '𝐒', 'T': '𝐓', 'U': '𝐔', 'V': '𝐕', 'W': '𝐖', 'X': '𝐗', 'Y': '𝐘', 'Z': '𝐙',
        'a': '𝐚', 'b': '𝐛', 'c': '𝐜', 'd': '𝐝', 'e': '𝐞', 'f': '𝐟', 'g': '𝐠', 'h': '𝐡', 'i': '𝐢', 'j': '𝐣', 'k': '𝐤', 'l': '𝐥', 'm': '𝐦', 'n': '𝐧', 'o': '𝐨', 'p': '𝐩', 'q': '𝐪', 'r': '𝐫', 's': '𝐬', 't': '𝐭', 'u': '𝐮', 'v': '𝐯', 'w': '𝐰', 'x': '𝐱', 'y': '𝐲', 'z': '𝐳',
        '0': '𝟎', '1': '𝟏', '2': '𝟐', '3': '𝟑', '4': '𝟒', '5': '𝟓', '6': '𝟔', '7': '𝟕', '8': '𝟖', '9': '𝟗'
    }
    return "".join(mapping.get(c, c) for c in text)


def get_start_buttons(bot_username):
    # Reverting to regular URL buttons for Telegram links to fix Button_url_invalid error.
    # Telegram deep links (t.me) are not allowed as Web App URLs.
    keyboard = [
        [
            InlineKeyboardButton(f"{to_stylish('Channel')}", url="https://t.me/Tele_212_bots"),
        ],
        [
            InlineKeyboardButton(f"{to_stylish('Add to Group')}", url=f"https://t.me/{bot_username}?startgroup=true")
        ],
        [
            InlineKeyboardButton(f"{to_stylish('Settings')}", callback_data="open_settings"),
            InlineKeyboardButton(f"{to_stylish('Help')}", callback_data="help_menu")
        ],
    ]
    return InlineKeyboardMarkup(keyboard)

GROUP_WARN_TEXT = to_stylish(os.getenv("GROUP_WARN_TEXT", "NIKAL YHA SE"))
NAME_GROUP_WARN_TEXT = to_stylish(os.getenv("NAME_GROUP_WARN_TEXT", "CHUP KR RE"))
PFP_GROUP_WARN_TEXT = to_stylish(os.getenv("PFP_GROUP_WARN_TEXT", "NIKLO JI"))
PHOTO_GROUP_WARN_TEXT = to_stylish(os.getenv("PHOTO_GROUP_WARN_TEXT", "PHOTO ACCHA DAAL"))
FLOOD_WARN_TEXT = to_stylish(os.getenv("FLOOD_WARN_TEXT", "SPAM MAT KAR"))

STICKER_SET_BLACKLIST = [s.strip() for s in os.getenv("STICKER_SET_BLACKLIST", "").split(",") if s.strip()]
STICKER_SET_SUFFIX_BLACKLIST = [s.strip() for s in os.getenv("STICKER_SET_SUFFIX_BLACKLIST", "").split(",") if s.strip()]
STICKER_WHITELIST = [s.strip() for s in os.getenv("STICKER_WHITELIST", "sunshine_fvrt_by_fStikBot").split(",") if s.strip()]
NAME_WARN_ENABLED = os.getenv("NAME_WARN_ENABLED", "1") not in ("0", "false", "False", "")
NAME_WARN_ONCE = os.getenv("NAME_WARN_ONCE", "1") in ("1", "true", "True")
PFP_CHECK_ENABLED = os.getenv("PFP_CHECK_ENABLED", "1") not in ("0", "false", "False", "")
PFP_NSFW_THRESHOLD = float(os.getenv("PFP_NSFW_THRESHOLD", "0.45"))
PFP_SKIN_THRESHOLD = float(os.getenv("PFP_SKIN_THRESHOLD", "0.22"))
BOTTOM_SKIN_THRESHOLD = float(os.getenv("BOTTOM_SKIN_THRESHOLD", "0.35"))
PFP_BOTTOM_SKIN_THRESHOLD = float(os.getenv("PFP_BOTTOM_SKIN_THRESHOLD", "0.30"))
DRUG_KEYWORDS = [w.strip() for w in os.getenv("DRUG_KEYWORDS", "weed,ganja,cocaine,heroin,lsd,mdma,meth,opium,charas,hash,drug,narcotic,smack,brown sugar,ecstasy").split(",") if w.strip()]
RECENT_MSG_LIMIT = int(os.getenv("RECENT_MSG_LIMIT", "300"))
NSFW_LABELS = [
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "BUTTOCKS_EXPOSED",
]
DELETE_SERVICE_MESSAGES = os.getenv("DELETE_SERVICE_MESSAGES", "1") not in ("0", "false", "False", "")
FLOOD_WINDOW_SECONDS = int(os.getenv("FLOOD_WINDOW_SECONDS", "10"))
FLOOD_MAX_MESSAGES = int(os.getenv("FLOOD_MAX_MESSAGES", "10"))
MUTE_ON_FLOOD = os.getenv("MUTE_ON_FLOOD", "0") in ("1", "true", "True")
SELF_DELETE_SECONDS = int(os.getenv("SELF_DELETE_SECONDS", "10"))
STRICT_MODE = os.getenv("STRICT_MODE", "0") in ("1", "true", "True")

logging.basicConfig(level=logging.INFO)
detector = NudeDetector()
user_strikes = {}

# Recommendation: Use a structured Key Pair for identifying Group-User relationships
GroupUserKey = namedtuple("GroupUserKey", ["chat_id", "user_id"])

_pfp_blocked = set()
_name_warned = set()
_scanned_users = set() # Store as GroupUserKey
_admin_cache = {} # GroupUserKey -> (is_admin, timestamp)
_recent_msgs = {} # GroupUserKey -> list of message IDs
_message_times = {} # GroupUserKey -> list of timestamps
_known_groups = set()  # Track groups the bot is in
user_last_action = {}


def is_spam(user_id):
    import time
    now = time.time()
    last = user_last_action.get(user_id, 0)
    user_last_action[user_id] = now
    return now - last < 2  # 2 sec spam window


_group_settings = {}
SETTINGS_FILE = "group_settings.json"

def load_settings():
    global _group_settings, _known_groups
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                data = json.load(f)
                # Convert keys back to int
                _group_settings = {int(k): v for k, v in data.items()}
                # Also track these as known groups
                for gid in _group_settings.keys():
                    _known_groups.add(gid)
        except Exception as e:
            logging.error(f"Error loading settings: {e}")
            _group_settings = {}

def save_settings():
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(_group_settings, f)
    except Exception as e:
        logging.error(f"Error saving settings: {e}")

def get_setting(chat_id, key, default):
    chat_settings = _group_settings.get(chat_id, {})
    return chat_settings.get(key, default)

def toggle_setting(chat_id, key):
    if chat_id not in _group_settings:
        _group_settings[chat_id] = {
            "nsfw": True,
            "drugs": True,
            "flood": True,
            "warn_admins": True,
            "edits": True,
            "voice_chat": True,
            "service_msgs": True,
            "pfp_check": True,
            "name_warn": True,
            "flood_limit": 10,
            "strict_mode": False,
            "sticker_check": True,
            "max_strikes": 5,
            "delete_delay": 10
        }
    
    if key == "flood_inc":
        _group_settings[chat_id]["flood_limit"] = _group_settings[chat_id].get("flood_limit", 10) + 1
    elif key == "flood_dec":
        limit = _group_settings[chat_id].get("flood_limit", 10)
        if limit > 1:
            _group_settings[chat_id]["flood_limit"] = limit - 1
    elif key.startswith("flood_val_"):
        try:
            val = int(key.split("_")[-1])
            _group_settings[chat_id]["flood_limit"] = val
        except Exception:
            pass
    elif key == "strikes_inc":
        _group_settings[chat_id]["max_strikes"] = _group_settings[chat_id].get("max_strikes", 5) + 1
    elif key == "strikes_dec":
        val = _group_settings[chat_id].get("max_strikes", 5)
        if val > 1:
            _group_settings[chat_id]["max_strikes"] = val - 1
    elif key == "delete_inc":
        _group_settings[chat_id]["delete_delay"] = _group_settings[chat_id].get("delete_delay", 10) + 5
    elif key == "delete_dec":
        val = _group_settings[chat_id].get("delete_delay", 10)
        if val > 5:
            _group_settings[chat_id]["delete_delay"] = val - 5
    else:
        _group_settings[chat_id][key] = not _group_settings[chat_id].get(key, True)
    
    save_settings()

load_settings()


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


def is_nsfw_safe(result, chat_id=None):
    for item in result or []:
        try:
            lbl = str(item.get("label") or item.get("class") or "")
            sc = float(item.get("score", 0.0))
        except Exception:
            continue
        
        # Use group-specific strict mode if available
        is_strict = get_setting(chat_id, "strict_mode", False) if chat_id else STRICT_MODE
        
        if lbl in NSFW_LABELS:
            if is_strict:
                if sc >= 0.85:
                    return True
            else:
                if sc >= 0.75:
                    return True
    return False


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


def contains_hindi_nsfw(text: str): 
    if not text: 
        return False 
    text_lower = text.lower() 
    # check hindi words 
    for word in HINDI_NSFW_WORDS: 
        if word in text: 
            return True 
    # check hinglish 
    for word in HINGLISH_NSFW_WORDS: 
        if word in text_lower: 
            return True 
    return False 


def contains_bad_text(text: str | None):
    if not text:
        return False
    if contains_hindi_nsfw(text):
        return True
    
    t1, _ = _normalize_variants(text)
    for w in BAD_WORDS:
        w_norm = w.lower().translate(_LEET_MAP)
        # Use regex to ensure the bad word is not just letters inside another word
        # \b ensures word boundaries (start/end of word, space, punctuation)
        pattern = rf"(?i)\b{re.escape(w_norm)}\b"
        if re.search(pattern, t1):
            return True
    return False


def contains_drug_content(text: str | None):
    if not text:
        return False
    a, _ = _normalize_variants(text)
    for w in DRUG_KEYWORDS:
        w_norm = w.lower().translate(_LEET_MAP)
        # Use regex to ensure the drug word is matched as a whole word, not letters within
        pattern = rf"(?i)\b{re.escape(w_norm)}\b"
        if re.search(pattern, a):
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


def _track_user_message(update: Update):
    try:
        chat = update.effective_chat
        if chat.type in ["group", "supergroup"]:
            _known_groups.add(chat.id)
        chat_id = chat.id
        user_id = update.effective_user.id
        key = GroupUserKey(chat_id, user_id)
        arr = _recent_msgs.get(key)
        if arr is None:
            arr = []
            _recent_msgs[key] = arr
        arr.append(update.effective_message.message_id)
        if len(arr) > RECENT_MSG_LIMIT:
            del arr[0: len(arr) - RECENT_MSG_LIMIT]
    except Exception:
        pass


def _track_user_time(update: Update, ts: float):
    try:
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id
        key = GroupUserKey(chat_id, user_id)
        arr = _message_times.get(key)
        if arr is None:
            arr = []
            _message_times[key] = arr
        arr.append(ts)
        cutoff = ts - float(FLOOD_WINDOW_SECONDS)
        while arr and arr[0] < cutoff:
            arr.pop(0)
    except Exception:
        pass


async def maybe_enforce_flood(update: Update, context: ContextTypes.DEFAULT_TYPE):
    import time
    ts = time.time()
    _track_user_time(update, ts)
    try:
        chat_id = update.effective_chat.id
        # Respect group-specific setting
        if not get_setting(chat_id, "flood", True):
            return False
            
        user_id = update.effective_user.id
        key = GroupUserKey(chat_id, user_id)
        arr = _message_times.get(key, [])
        
        # Use group-specific limit or fallback to global default
        limit = get_setting(chat_id, "flood_limit", FLOOD_MAX_MESSAGES)
        
        if len(arr) > int(limit):
            try:
                # Delete only the current message that triggered the flood
                await update.effective_message.delete()
            except Exception:
                pass
            
            # Optionally delete only the very recent messages from this user
            # but NOT the entire history (limiting to last 5 messages to clear the screen)
            try:
                mids = _recent_msgs.get(key, [])
                for mid in list(mids)[-5:]: # Only clear the last 5 messages
                    try:
                        await context.bot.delete_message(chat_id, mid)
                    except Exception:
                        continue
                # Keep the older history in _recent_msgs for other purposes
                # but prevent them from being double-deleted later
                _recent_msgs[key] = mids[:-5]
            except Exception:
                pass
            if MUTE_ON_FLOOD:
                try:
                    await context.bot.restrict_chat_member(chat_id, user_id, permissions=ChatPermissions(can_send_messages=False))
                except Exception:
                    pass
            try:
                await send_temp_message(context, chat_id, FLOOD_WARN_TEXT)
            except Exception:
                pass
            return True
    except Exception:
        return False
    return False


async def warn_admins(update: Update, context: ContextTypes.DEFAULT_TYPE, caption: str, pfp_path: str | None):
    chat_id = update.effective_chat.id
    if not WARN_ADMINS or not get_setting(chat_id, "warn_admins", True):
        return
    try:
        # Get group title for better DM context
        chat = await context.bot.get_chat(chat_id)
        group_name = chat.title
        enhanced_caption = f"📍 {to_stylish('Group')}: {group_name}\n\n{caption}"

        admins = await context.bot.get_chat_administrators(chat_id)
        for admin in admins:
            # Prioritize the owner (creator) or notify all admins if that's the current behavior
            # The user specifically requested the owner get the DM
            if admin.status != "creator":
                continue
                
            try:
                if pfp_path and os.path.exists(pfp_path):
                    with open(pfp_path, "rb") as f:
                        await context.bot.send_photo(admin.user.id, photo=f, caption=enhanced_caption)
                else:
                    await context.bot.send_message(admin.user.id, enhanced_caption)
            except Exception:
                # Owner likely hasn't started the bot in DM
                pass
    except Exception:
        pass


async def send_temp_message(context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str, delay: int | None = None):
    # Use group-specific delay or fallback to provided/global delay
    group_delay = get_setting(chat_id, "delete_delay", SELF_DELETE_SECONDS)
    d = group_delay if delay is None else int(delay)
    
    try:
        m = await context.bot.send_message(chat_id, text)
    except Exception:
        return
    async def _del():
        try:
            await asyncio.sleep(d)
            await context.bot.delete_message(chat_id, m.message_id)
        except Exception:
            pass
    try:
        asyncio.create_task(_del())
    except Exception:
        pass

async def punish_user(update: Update, context: ContextTypes.DEFAULT_TYPE, reason: str | None = None):
    user = update.effective_user
    chat_id = update.effective_chat.id
    user_strikes[user.id] = user_strikes.get(user.id, 0) + 1
    strikes = user_strikes[user.id]
    
    # Get group-specific max strikes
    group_max_strikes = get_setting(chat_id, "max_strikes", MAX_STRIKES)
    
    pfp = await get_user_profile_pic(context, user.id, update.effective_message.message_id)
    
    # Stylish headers
    caption = (
        f"🚨 {to_stylish('NSFW Violation Detected')}\n\n"
        f"👤 {to_stylish('Name')}: {user.full_name}\n"
        f"🔗 {to_stylish('Username')}: @{user.username if user.username else 'None'}\n"
        f"🆔 {to_stylish('ID')}: {user.id}\n"
        f"⚠️ {to_stylish('Strikes')}: {strikes}/{group_max_strikes}"
    )
    if reason:
        caption = caption + f"\n📝 {to_stylish('Reason')}: {reason}"
    # Group: send short warning only
    try:
        await send_temp_message(context, chat_id, GROUP_WARN_TEXT)
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
    
    if strikes >= group_max_strikes:
        try:
            await context.bot.restrict_chat_member(chat_id, user.id, permissions=ChatPermissions(can_send_messages=False))
        except Exception as e:
            logging.warning("Mute error: %s", e)


async def maybe_warn_for_name(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not NAME_WARN_ENABLED:
        return False
    user = update.effective_user
    chat_id = update.effective_chat.id
    
    # Skip admins for name/pfp checks to avoid deleting their every message
    if await is_user_admin(chat_id, user.id, context):
        return False
        
    if not get_setting(chat_id, "name_warn", True):
        return False
        
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
            await update.effective_message.delete()
        except Exception:
            pass
        try:
            await context.bot.restrict_chat_member(chat_id, user.id, permissions=ChatPermissions(can_send_messages=False))
        except Exception:
            pass
        try:
            await send_temp_message(context, chat_id, NAME_GROUP_WARN_TEXT)
        except Exception:
            pass
        if WARN_ADMINS:
            cap = f"🚨 {to_stylish('NSFW Name Detected')}\n\n👤 {to_stylish('Name')}: {user.full_name}\n🔗 {to_stylish('Username')}: @{user.username if user.username else 'None'}\n🆔 {to_stylish('ID')}: {user.id}"
            try:
                await warn_admins(update, context, cap, None)
            except Exception:
                pass
        return True
    return False


async def maybe_enforce_pfp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not PFP_CHECK_ENABLED:
        return False
    user = update.effective_user
    chat_id = update.effective_chat.id
    key = (chat_id, user.id)
    
    if key in _scanned_users:
        return False
        
    # Skip admins for name/pfp checks to avoid deleting their every message
    if await is_user_admin(chat_id, user.id, context):
        _scanned_users.add(key)
        return False
        
    if not get_setting(chat_id, "pfp_check", True):
        return False
        
    if key in _pfp_blocked:
        return False
    
    path = await get_user_profile_pic(context, user.id, update.effective_message.message_id)
    if not path or not os.path.exists(path):
        # If no PFP, we can't scan it, so mark as scanned to avoid repeated attempts
        _scanned_users.add(key)
        return False
    
    # Mark as scanned before starting the actual scan to prevent concurrent scans
    _scanned_users.add(key)
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
        res = await _optimized_detect(target_path)
        sc = get_max_score(res)
        hit, hit_lbl, hit_sc = has_nsfw_label(res)
        hit_v2 = is_nsfw_safe(res, chat_id)
        img2 = cv2.imread(target_path)
        sr = skin_ratio_from_bgr(img2)
        bsr = bottom_skin_ratio_from_bgr(img2)
        if hit or hit_v2 or sc >= PFP_NSFW_THRESHOLD or sr >= PFP_SKIN_THRESHOLD or bsr >= PFP_BOTTOM_SKIN_THRESHOLD:
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
                cap = f"🚨 {to_stylish('NSFW Profile Photo')}\n\n👤 {to_stylish('Name')}: {user.full_name}\n🔗 {to_stylish('Username')}: @{user.username if user.username else 'None'}\n🆔 {to_stylish('ID')}: {user.id}"
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


async def _optimized_detect(path):
    """Resizes large images before scanning to boost detection speed by up to 3x."""
    try:
        img = cv2.imread(path)
        if img is None:
            return detector.detect(path)
        h, w = img.shape[:2]
        if max(h, w) > 512:
            scale = 512.0 / float(max(h, w))
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            resized_path = f"{path}.optimized.jpg"
            cv2.imwrite(resized_path, resized)
            try:
                res = detector.detect(resized_path)
                return res
            finally:
                if os.path.exists(resized_path):
                    os.remove(resized_path)
        return detector.detect(path)
    except Exception:
        return detector.detect(path)


async def check_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    if not message:
        return
    user = update.effective_user
    # Admin/Owner check removed to support "check also editted message by user/admin/owner"
    _track_user_message(update)
    if await maybe_enforce_flood(update, context):
        return
    if await maybe_warn_for_name(update, context):
        return
    if await maybe_enforce_pfp(update, context):
        return
    
    chat_id = update.effective_chat.id
    nsfw_enabled = get_setting(chat_id, "nsfw", True)
    drugs_enabled = get_setting(chat_id, "drugs", True)

    if nsfw_enabled or drugs_enabled:
        if contains_bad_text(getattr(message, "caption", None)) or contains_drug_content(getattr(message, "caption", None)):
            # Double check with exact flags if needed, but caption usually implies nsfw/drugs
            await message.delete()
            await punish_user(update, context, "Photo caption text")
            return

    if not nsfw_enabled:
        return

    file = await message.photo[-1].get_file()
    path = f"photo_{message.message_id}.jpg"
    await file.download_to_drive(path)
    try:
        result = await _optimized_detect(path)
        score = get_max_score(result)
        hit, hit_lbl, hit_sc = has_nsfw_label(result)
        hit_v2 = is_nsfw_safe(result, chat_id)
        img = cv2.imread(path)
        sr = skin_ratio_from_bgr(img)
        bsr = bottom_skin_ratio_from_bgr(img)
        if hit or hit_v2 or score >= NSFW_THRESHOLD or sr >= SKIN_THRESHOLD or bsr >= BOTTOM_SKIN_THRESHOLD:
            await message.delete()
            try:
                await send_temp_message(context, update.effective_chat.id, PHOTO_GROUP_WARN_TEXT)
            except Exception:
                pass
            return
    finally:
        if os.path.exists(path):
            os.remove(path)


async def check_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    if not message:
        return
    user = update.effective_user
    # Admin/Owner check removed to support "check also editted message by user/admin/owner"
    _track_user_message(update)
    if await maybe_enforce_flood(update, context):
        return
    if await maybe_warn_for_name(update, context):
        return
    if await maybe_enforce_pfp(update, context):
        return
    
    chat_id = update.effective_chat.id
    nsfw_enabled = get_setting(chat_id, "nsfw", True)
    drugs_enabled = get_setting(chat_id, "drugs", True)

    if nsfw_enabled or drugs_enabled:
        if contains_bad_text(getattr(message, "caption", None)) or contains_drug_content(getattr(message, "caption", None)):
            await message.delete()
            await punish_user(update, context, "Video caption text")
            return

    if not nsfw_enabled:
        return

    file = await message.video.get_file()
    path = f"video_{message.message_id}.mp4"
    await file.download_to_drive(path)
    try:
        cap = cv2.VideoCapture(path)
        frame_count = 0
        tasks = []
        frames_to_scan = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 30 == 0:
                frames_to_scan.append((frame_count, frame.copy()))
            frame_count += 1
        cap.release()

        async def scan_frame(f_idx, f_img):
            temp = f"frame_{message.message_id}_{f_idx}.jpg"
            try:
                cv2.imwrite(temp, f_img)
                res = await _optimized_detect(temp)
                score = get_max_score(res)
                hit, hit_lbl, hit_sc = has_nsfw_label(res)
                hit_v2 = is_nsfw_safe(res, chat_id)
                sr = skin_ratio_from_bgr(f_img)
                bsr = bottom_skin_ratio_from_bgr(f_img)
                
                if hit or hit_v2 or score >= NSFW_THRESHOLD or sr >= SKIN_THRESHOLD or bsr >= BOTTOM_SKIN_THRESHOLD:
                    reason_local = []
                    if hit: reason_local.append(f"Label={hit_lbl}({hit_sc:.2f})")
                    if score >= NSFW_THRESHOLD: reason_local.append(f"NSFW score={score:.2f}")
                    if sr >= SKIN_THRESHOLD: reason_local.append(f"Skin ratio={sr:.2f}")
                    if bsr >= BOTTOM_SKIN_THRESHOLD: reason_local.append(f"Bottom skin ratio={bsr:.2f}")
                    return True, ", ".join(reason_local) if reason_local else "NSFW"
                return False, None
            finally:
                if os.path.exists(temp):
                    os.remove(temp)

        # Boost: Scan multiple frames in parallel
        results = await asyncio.gather(*[scan_frame(idx, img) for idx, img in frames_to_scan])
        
        for nsfw_detected, trigger_reason in results:
            if nsfw_detected:
                await message.delete()
                try:
                    await punish_user(update, context, trigger_reason)
                except Exception:
                    await punish_user(update, context, "NSFW")
                return
    finally:
        if os.path.exists(path):
            os.remove(path)


async def check_sticker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    if not message:
        return
    user = update.effective_user
    # Admin/Owner check removed to support "check also editted message by user/admin/owner"
    _track_user_message(update)
    if await maybe_enforce_flood(update, context):
        return
    if await maybe_warn_for_name(update, context):
        return
    if await maybe_enforce_pfp(update, context):
        return
    
    chat_id = update.effective_chat.id
    nsfw_enabled = get_setting(chat_id, "nsfw", True)
    drugs_enabled = get_setting(chat_id, "drugs", True)
    sticker_enabled = get_setting(chat_id, "sticker_check", True)

    if not sticker_enabled:
        return

    set_name = getattr(message.sticker, "set_name", None)
    if set_name in STICKER_WHITELIST:
        return

    if set_name and (set_name in STICKER_SET_BLACKLIST or any(set_name.endswith(suf) for suf in STICKER_SET_SUFFIX_BLACKLIST)):
        if nsfw_enabled: # Assume sticker sets are mostly NSFW related
            await message.delete()
            await punish_user(update, context, f"Sticker set {set_name}")
            return
    
    if not nsfw_enabled and not drugs_enabled:
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
            text1 = pytesseract.image_to_string(enlarged, config="--psm 6", lang="eng+hin")
        except Exception:
            text1 = ""
        try:
            text2 = pytesseract.image_to_string(inv, config="--psm 6", lang="eng+hin")
        except Exception:
            text2 = ""
        text = f"{text1} {text2}".strip()
        nuderes = await _optimized_detect(jpg_path)
        nscore = get_max_score(nuderes)
        hit, hit_lbl, hit_sc = has_nsfw_label(nuderes)
        hit_v2 = is_nsfw_safe(nuderes, chat_id)
        if contains_bad_text(text) or contains_drug_content(text) or hit or hit_v2 or sr >= SKIN_THRESHOLD or nscore >= NSFW_THRESHOLD or bsr >= BOTTOM_SKIN_THRESHOLD:
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
    message = update.effective_message
    if not message:
        return
    user = update.effective_user
    # Admin/Owner check removed to support "check also editted message by user/admin/owner"
    _track_user_message(update)
    if await maybe_enforce_flood(update, context):
        return
    if await maybe_warn_for_name(update, context):
        return
    if await maybe_enforce_pfp(update, context):
        return
    # Skip media messages; handled by other callbacks
    if getattr(message, "photo", None) or getattr(message, "video", None) or getattr(message, "sticker", None):
        return
    txt = message.text or message.caption
    
    chat_id = update.effective_chat.id
    nsfw_enabled = get_setting(chat_id, "nsfw", True)
    drugs_enabled = get_setting(chat_id, "drugs", True)
    
    is_bad = contains_bad_text(txt) if nsfw_enabled else False
    is_drug = contains_drug_content(txt) if drugs_enabled else False
    
    if is_bad or is_drug:
        try:
            await message.delete()
            reason = "NSFW Text" if is_bad else "Drug Content"
            await punish_user(update, context, reason)
        except Exception as e:
            logging.error(f"Error deleting bad text: {e}")


async def delete_service_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not DELETE_SERVICE_MESSAGES:
        return
    
    chat_id = update.effective_chat.id
    if not get_setting(chat_id, "service_msgs", True):
        return

    try:
        await update.effective_message.delete()
    except Exception:
        pass


async def check_voice_chat_invite(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    if not message or not message.video_chat_participants_invited:
        return
    
    chat_id = update.effective_chat.id
    if not get_setting(chat_id, "voice_chat", True):
        return
    
    # Check if any invited user has an NSFW name
    for user in message.video_chat_participants_invited.users:
        name_parts = []
        if user.full_name:
            name_parts.append(user.full_name)
        if user.username:
            name_parts.append(user.username)
        full_name = " ".join(name_parts)
        
        if contains_bad_text(full_name):
            try:
                await message.delete()
                # Optionally warn admins about the user
                if WARN_ADMINS:
                    cap = f"🚨 {to_stylish('NSFW Name in Voice Chat Invite')}\n\n👤 {to_stylish('Name')}: {user.full_name}\n🔗 {to_stylish('Username')}: @{user.username if user.username else 'None'}\n🆔 {to_stylish('ID')}: {user.id}"
                    await warn_admins(update, context, cap, None)
                return
            except Exception:
                pass


async def delete_edited_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.edited_message
    if not message:
        return
    
    chat_id = update.effective_chat.id
    if not get_setting(chat_id, "edits", True):
        return

    try:
        await message.delete()
    except Exception:
        pass


async def is_user_admin(chat_id: int, user_id: int, context: ContextTypes.DEFAULT_TYPE):
    import time
    now = time.time()
    key = (chat_id, user_id)
    if key in _admin_cache:
        is_admin, ts = _admin_cache[key]
        if now - ts < 300: # Cache for 5 minutes
            return is_admin
            
    try:
        member = await context.bot.get_chat_member(chat_id, user_id)
        res = member.status in ["administrator", "creator"]
        _admin_cache[key] = (res, now)
        return res
    except Exception:
        return False


async def mute_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    user = update.effective_user
    if not await is_user_admin(chat.id, user.id, context):
        return
    
    target_msg = update.message.reply_to_message
    if not target_msg:
        await update.message.reply_text("❗ Please reply to the user you want to mute.")
        return
    
    target_user = target_msg.from_user
    try:
        await context.bot.restrict_chat_member(chat.id, target_user.id, permissions=ChatPermissions(can_send_messages=False))
        await update.message.reply_text(f"🔇 {to_stylish(target_user.full_name)} has been muted.")
    except Exception as e:
        await update.message.reply_text(f"❌ Error muting user: {e}")


async def unmute_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    user = update.effective_user
    if not await is_user_admin(chat.id, user.id, context):
        return
    
    target_msg = update.message.reply_to_message
    if not target_msg:
        await update.message.reply_text("❗ Please reply to the user you want to unmute.")
        return
    
    target_user = target_msg.from_user
    try:
        await context.bot.restrict_chat_member(chat.id, target_user.id, permissions=ChatPermissions(
            can_send_messages=True,
            can_send_audios=True,
            can_send_documents=True,
            can_send_photos=True,
            can_send_videos=True,
            can_send_video_notes=True,
            can_send_voice_notes=True,
            can_send_polls=True,
            can_send_other_messages=True,
            can_add_web_page_previews=True,
            can_change_info=True,
            can_invite_users=True,
            can_pin_messages=True
        ))
        await update.message.reply_text(f"🔊 {to_stylish(target_user.full_name)} has been unmuted.")
    except Exception as e:
        await update.message.reply_text(f"❌ Error unmuting user: {e}")


async def ban_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    user = update.effective_user
    if not await is_user_admin(chat.id, user.id, context):
        return
    
    target_msg = update.message.reply_to_message
    if not target_msg:
        await update.message.reply_text("❗ Please reply to the user you want to ban.")
        return
    
    target_user = target_msg.from_user
    try:
        await context.bot.ban_chat_member(chat.id, target_user.id)
        await update.message.reply_text(f"🚫 {to_stylish(target_user.full_name)} has been banned.")
    except Exception as e:
        await update.message.reply_text(f"❌ Error banning user: {e}")


async def unban_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    user = update.effective_user
    if not await is_user_admin(chat.id, user.id, context):
        return
    
    # Check if replied to a message or if ID provided
    target_user_id = None
    target_name = "User"
    if update.message.reply_to_message:
        target_user_id = update.message.reply_to_message.from_user.id
        target_name = update.message.reply_to_message.from_user.full_name
    elif context.args:
        try:
            target_user_id = int(context.args[0])
        except ValueError:
            await update.message.reply_text("❗ Please provide a valid User ID or reply to a message.")
            return
    else:
        await update.message.reply_text("❗ Please reply to a message or provide a User ID to unban.")
        return
    
    try:
        await context.bot.unban_chat_member(chat.id, target_user_id, only_if_banned=True)
        await update.message.reply_text(f"✅ {to_stylish(target_name)} has been unbanned.")
    except Exception as e:
        await update.message.reply_text(f"❌ Error unbanning user: {e}")


async def settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    
    # Find groups where this user is the creator
    owned_groups = []
    for gid in list(_known_groups):
        try:
            member = await context.bot.get_chat_member(gid, user_id)
            if member.status == "creator":
                chat = await context.bot.get_chat(gid)
                owned_groups.append((gid, chat.title))
        except Exception:
            continue
            
    if not owned_groups:
        await query.edit_message_text("❌ You don't appear to be the owner of any groups I'm in.")
        return

    keyboard = []
    for gid, title in owned_groups:
        keyboard.append([InlineKeyboardButton(f"📝 {title}", callback_data=f"set_custom_{gid}")])
    
    keyboard.append([InlineKeyboardButton(f"{to_stylish('Back')}", callback_data="back_to_start")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text("Select a group to customize settings:", reply_markup=reply_markup)


async def set_custom_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data
    await query.answer()
    
    # Check if this is a toggle action or just opening the menu
    if data.startswith("toggle_"):
        parts = data.split("_")
        gid = int(parts[-1])
        key = "_".join(parts[1:-1])
        toggle_setting(gid, key)
    else:
        gid = int(data.split("_")[-1])
    
    nsfw = get_setting(gid, "nsfw", True)
    drugs = get_setting(gid, "drugs", True)
    flood = get_setting(gid, "flood", True)
    warn_admins = get_setting(gid, "warn_admins", True)
    edits = get_setting(gid, "edits", True)
    voice_chat = get_setting(gid, "voice_chat", True)
    service_msgs = get_setting(gid, "service_msgs", True)
    pfp_check = get_setting(gid, "pfp_check", True)
    name_warn = get_setting(gid, "name_warn", True)
    flood_limit = get_setting(gid, "flood_limit", 10)
    strict_mode = get_setting(gid, "strict_mode", False)
    sticker_check = get_setting(gid, "sticker_check", True)
    max_strikes = get_setting(gid, "max_strikes", 5)
    delete_delay = get_setting(gid, "delete_delay", 10)
    
    try:
        chat = await context.bot.get_chat(gid)
        title = chat.title
    except Exception:
        title = f"Group {gid}"

    keyboard = [
        [InlineKeyboardButton(f"{'✅' if nsfw else '❌'} Anti-NSFW", callback_data=f"toggle_nsfw_{gid}")],
        [InlineKeyboardButton(f"{'✅' if drugs else '❌'} Anti-Drugs", callback_data=f"toggle_drugs_{gid}")],
        [InlineKeyboardButton(f"{'✅' if sticker_check else '❌'} Sticker Scan", callback_data=f"toggle_sticker_check_{gid}")],
        [InlineKeyboardButton(f"{'✅' if strict_mode else '❌'} AI Strict Mode", callback_data=f"toggle_strict_mode_{gid}")],
        [InlineKeyboardButton(f"{'✅' if flood else '❌'} Flood Protection", callback_data=f"toggle_flood_{gid}")],
        [
            InlineKeyboardButton(f"{'📍' if flood_limit == 5 else ''} 5", callback_data=f"toggle_flood_val_5_{gid}"),
            InlineKeyboardButton(f"{'📍' if flood_limit == 10 else ''} 10", callback_data=f"toggle_flood_val_10_{gid}"),
            InlineKeyboardButton(f"{'📍' if flood_limit == 15 else ''} 15", callback_data=f"toggle_flood_val_15_{gid}"),
            InlineKeyboardButton(f"{'📍' if flood_limit == 20 else ''} 20", callback_data=f"toggle_flood_val_20_{gid}"),
            InlineKeyboardButton(f"{'📍' if flood_limit == 25 else ''} 25", callback_data=f"toggle_flood_val_25_{gid}"),
        ],
        [
            InlineKeyboardButton("➖", callback_data=f"toggle_flood_dec_{gid}"),
            InlineKeyboardButton(f"Spam Limit: {flood_limit}", callback_data="none"),
            InlineKeyboardButton("➕", callback_data=f"toggle_flood_inc_{gid}")
        ],
        [
            InlineKeyboardButton("➖", callback_data=f"toggle_strikes_dec_{gid}"),
            InlineKeyboardButton(f"Max Strikes: {max_strikes}", callback_data="none"),
            InlineKeyboardButton("➕", callback_data=f"toggle_strikes_inc_{gid}")
        ],
        [
            InlineKeyboardButton("➖", callback_data=f"toggle_delete_dec_{gid}"),
            InlineKeyboardButton(f"Delete Delay: {delete_delay}s", callback_data="none"),
            InlineKeyboardButton("➕", callback_data=f"toggle_delete_inc_{gid}")
        ],
        [InlineKeyboardButton(f"{'✅' if edits else '❌'} Auto-Delete Edits", callback_data=f"toggle_edits_{gid}")],
        [InlineKeyboardButton(f"{'✅' if voice_chat else '❌'} Voice Chat Protection", callback_data=f"toggle_voice_chat_{gid}")],
        [InlineKeyboardButton(f"{'✅' if service_msgs else '❌'} Service Msg Delete", callback_data=f"toggle_service_msgs_{gid}")],
        [InlineKeyboardButton(f"{'✅' if pfp_check else '❌'} Profile Pic Check", callback_data=f"toggle_pfp_check_{gid}")],
        [InlineKeyboardButton(f"{'✅' if name_warn else '❌'} Name Warning", callback_data=f"toggle_name_warn_{gid}")],
        [InlineKeyboardButton(f"{'✅' if warn_admins else '❌'} Admin Notifications", callback_data=f"toggle_warn_admins_{gid}")],
        [InlineKeyboardButton(f"{to_stylish('Back')}", callback_data="open_settings")]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    text = f"⚙️ **Settings for {title}**\n\nClick a button to toggle the protection status."
    
    await query.edit_message_text(text, reply_markup=reply_markup, parse_mode="Markdown")


async def group_setting(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    user = update.effective_user
    
    if chat.type == "private":
        return await start(update, context)
        
    # Check if user is owner
    try:
        member = await context.bot.get_chat_member(chat.id, user.id)
        if member.status != "creator":
            await send_temp_message(context, chat.id, "❌ Only the group owner can use this command.", SELF_DELETE_SECONDS)
            return
    except Exception:
        return

    # Tracking group for DM logic
    _known_groups.add(chat.id)
    
    bot_username = (await context.bot.get_me()).username
    # Deep link to open settings in DM for this specific group
    url = f"https://t.me/{bot_username}?start=settings_{chat.id}"
    keyboard = [[InlineKeyboardButton(f"{to_stylish('Open Settings in DM')}", url=url)]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        f"Hello {user.first_name}! Please click the button below to manage settings for **{chat.title}** in our private chat.",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    user = update.effective_user
    bot = await context.bot.get_me()
    
    if chat.type == "private":
        # Check for deep linking parameters
        if context.args and context.args[0].startswith("settings_"):
            target_gid = context.args[0].replace("settings_", "")
            try:
                target_gid = int(target_gid)
                # Verify ownership of this specific group
                member = await context.bot.get_chat_member(target_gid, user.id)
                if member.status == "creator":
                    group_chat = await context.bot.get_chat(target_gid)
                    keyboard = [[InlineKeyboardButton(f"📝 {to_stylish('Configure')} {group_chat.title}", callback_data=f"set_custom_{target_gid}")]]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    await update.message.reply_text(
                        f"Welcome! You can now configure settings for **{group_chat.title}**.",
                        reply_markup=reply_markup,
                        parse_mode="Markdown"
                    )
                    return
                else:
                    await update.message.reply_text("❌ You are not the owner of that group.")
            except Exception:
                await update.message.reply_text("❌ Could not verify group ownership.")

        # Get bot profile pic
        pfp = None
        try:
            photos = await context.bot.get_user_profile_photos(bot.id, limit=1)
            if photos.total_count > 0:
                pfp = photos.photos[0][-1].file_id
        except Exception:
            pass

        hello_lbl = to_stylish("Hello")
        guard_lbl = to_stylish("I am ULTRA++ NSFW Guard").replace("+", "\\+")
        welcome_text = (
            f"👋 {hello_lbl}, {user.mention_markdown_v2()}\!\n\n"
            f"🛡️ {guard_lbl}\n\n"
            f"I can protect your groups from NSFW content, drugs, and spam using advanced AI detection\."
        )

        reply_markup = get_start_buttons(bot.username)

        if pfp:
            await context.bot.send_photo(
                chat_id=chat.id,
                photo=pfp,
                caption=welcome_text,
                reply_markup=reply_markup,
                parse_mode="MarkdownV2",
                has_spoiler=True
            )
        else:
            await update.message.reply_text(
                welcome_text,
                reply_markup=reply_markup,
                parse_mode="MarkdownV2"
            )
    else:
        await send_temp_message(context, chat.id, "🛡️ ULTRA++ NSFW Guard Active\nPhotos • Videos • Sticker Text protected.", SELF_DELETE_SECONDS)


async def help_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    help_text = (
        f"📖 {to_stylish('Help Menu')}\n\n"
        f"🛡️ {to_stylish('Basic Commands')}:\n"
        f"• /start \- Start the bot\n"
        f"• /mute \- Mute a user \(Admin only\)\n"
        f"• /unmute \- Unmute a user \(Admin only\)\n"
        f"• /ban \- Ban a user \(Admin only\)\n"
        f"• /unban \- Unban a user \(Admin only\)\n\n"
        f"✨ {to_stylish('Features')}:\n"
        f"• Anti\-NSFW Image/Video/Sticker detection\n"
        f"• Anti\-Drug text detection\n"
        f"• Hindi/Hinglish NSFW detection\n"
        f"• Flood/Spam protection\n"
        f"• Profile Picture NSFW enforcement\n"
        f"• Automatically deletes any edited messages\n"
        f"• Automatically deletes service messages"
    )
    
    keyboard = [[InlineKeyboardButton(f"{to_stylish('Back')}", callback_data="back_to_start")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_caption(
        caption=help_text,
        reply_markup=reply_markup,
        parse_mode="MarkdownV2"
    ) if query.message.caption else await query.edit_message_text(
        text=help_text,
        reply_markup=reply_markup,
        parse_mode="MarkdownV2"
    )


async def back_to_start_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user = update.effective_user
    bot = await context.bot.get_me()
    
    hello_lbl = to_stylish("Hello")
    guard_lbl = to_stylish("I am ULTRA++ NSFW Guard").replace("+", "\\+")
    welcome_text = (
        f"👋 {hello_lbl}, {user.mention_markdown_v2()}\!\n\n"
        f"🛡️ {guard_lbl}\n\n"
        f"I can protect your groups from NSFW content, drugs, and spam using advanced AI detection\."
    )

    reply_markup = get_start_buttons(bot.username)
    
    if query.message.caption:
        await query.edit_message_caption(
            caption=welcome_text,
            reply_markup=reply_markup,
            parse_mode="MarkdownV2"
        )
    else:
        await query.edit_message_text(
            text=welcome_text,
            reply_markup=reply_markup,
            parse_mode="MarkdownV2"
        )


def main():
    if not BOT_TOKEN or BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN or BOT_TOKEN environment variable")
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("setting", group_setting))
    app.add_handler(CommandHandler("mute", mute_user))
    app.add_handler(CommandHandler("unmute", unmute_user))
    app.add_handler(CommandHandler("ban", ban_user))
    app.add_handler(CommandHandler("unban", unban_user))
    app.add_handler(CallbackQueryHandler(help_callback, pattern="^help_menu$"))
    app.add_handler(CallbackQueryHandler(back_to_start_callback, pattern="^back_to_start$"))
    app.add_handler(CallbackQueryHandler(settings_callback, pattern="^open_settings$"))
    app.add_handler(CallbackQueryHandler(set_custom_callback, pattern="^(set_custom_|toggle_)"))
    if DELETE_SERVICE_MESSAGES:
        app.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS | filters.StatusUpdate.LEFT_CHAT_MEMBER, delete_service_message))
    
    # Feature: check voice/video chat invited users for NSFW content
    app.add_handler(MessageHandler(filters.StatusUpdate.VIDEO_CHAT_PARTICIPANTS_INVITED, check_voice_chat_invite))
    
    # Feature: delete any edited message
    app.add_handler(MessageHandler(filters.UpdateType.EDITED_MESSAGE, delete_edited_message))

    app.add_handler(MessageHandler(filters.PHOTO, check_photo))
    app.add_handler(MessageHandler(filters.VIDEO, check_video))
    app.add_handler(MessageHandler(filters.Sticker.ALL, check_sticker))
    
    text_filter = (filters.TEXT) | filters.CaptionRegex(".*")
    app.add_handler(MessageHandler(text_filter, check_text))
    app.run_polling()


if __name__ == "__main__":
    main()
