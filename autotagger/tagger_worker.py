import os
import time
import base64
import csv
from io import BytesIO
from urllib.parse import urljoin, quote

import numpy as np
import requests
import onnxruntime as rt
from PIL import Image


# -------------------- ENV --------------------

OXI_BASE_URL = os.getenv("OXI_BASE_URL", "http://server:6666").rstrip("/") + "/"
OXI_USER = os.getenv("OXI_USER")
OXI_TOKEN = os.getenv("OXI_TOKEN")
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "15"))

# If you want “accept 60% sure tags” set defaults to 0.60
GENERAL_THRESHOLD = float(os.getenv("GENERAL_THRESHOLD", "0.60"))
CHAR_THRESHOLD = float(os.getenv("CHAR_THRESHOLD", "0.60"))

USE_MCUT_GENERAL = os.getenv("USE_MCUT_GENERAL", "1") == "1"
USE_MCUT_CHARACTER = os.getenv("USE_MCUT_CHARACTER", "0") == "1"

FORCE_RETAG = os.getenv("FORCE_RETAG", "0") == "1"
CLEAR_OLD_TAGS = os.getenv("CLEAR_OLD_TAGS", "0") == "1"

MODEL_DIR = "/app/model"
ONNX_URL = "https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/model.onnx"
TAGS_URL = "https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/selected_tags.csv"
ONNX_PATH = os.path.join(MODEL_DIR, "model.onnx")
TAGS_PATH = os.path.join(MODEL_DIR, "selected_tags.csv")

MARKER_TAG = os.getenv("MARKER_TAG", "auto_tagged_wd3")

# Categories must already exist in Oxibooru
CAT_RATING = "rating"
CAT_CHARACTER = "character"
CAT_GENERAL = "general"

# Request tuning
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30"))
IMG_TIMEOUT = float(os.getenv("IMG_TIMEOUT", "90"))
USER_AGENT = os.getenv("USER_AGENT", "oxibooru-autotagger/1.0")


# -------------------- AUTH --------------------

def token_headers(json=True):
    raw = f"{OXI_USER}:{OXI_TOKEN}".encode("utf-8")
    b64 = base64.b64encode(raw).decode("ascii")
    h = {
        "Authorization": f"Token {b64}",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    if json:
        h["Content-Type"] = "application/json"
    return h


# -------------------- SAFE HTTP JSON --------------------

def request_json(method, url, *, headers=None, params=None, json=None, timeout=HTTP_TIMEOUT):
    """
    Like requests.request(...).json() but with:
    - status/body debug if response isn't JSON
    - raises with useful info
    """
    r = requests.request(method, url, headers=headers, params=params, json=json, timeout=timeout)

    # Fast path: ok + json content type
    ct = (r.headers.get("Content-Type") or "").lower()
    if "application/json" in ct:
        try:
            return r.json()
        except Exception:
            pass

    # Non-JSON or bad JSON -> print debug
    body_preview = (r.text or "")[:300].replace("\n", "\\n")
    raise RuntimeError(f"Non-JSON response from {url} -> {r.status_code} CT={ct!r} BODY={body_preview!r}")


# -------------------- DEMO LOGIC --------------------

def mcut_threshold(probs):
    sorted_probs = probs[probs.argsort()[::-1]]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    t = difs.argmax()
    return (sorted_probs[t] + sorted_probs[t + 1]) / 2


def prepare_image(image: Image.Image, target_size: int):
    # 1) RGBA -> composite on white
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    canvas = Image.new("RGBA", image.size, (255, 255, 255))
    canvas.alpha_composite(image)
    image = canvas.convert("RGB")

    # 2) pad to square
    w, h = image.size
    max_dim = max(w, h)
    pad_left = (max_dim - w) // 2
    pad_top = (max_dim - h) // 2
    padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded.paste(image, (pad_left, pad_top))

    # 3) resize
    if max_dim != target_size:
        padded = padded.resize((target_size, target_size), Image.BICUBIC)

    # 4) RGB -> BGR float32, NHWC
    arr = np.asarray(padded, dtype=np.float32)
    arr = arr[:, :, ::-1]
    return np.expand_dims(arr, axis=0)


# -------------------- MODEL --------------------

def ensure_model_files():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(ONNX_PATH):
        print("Downloading ONNX model...", flush=True)
        r = requests.get(ONNX_URL, timeout=180)
        r.raise_for_status()
        with open(ONNX_PATH, "wb") as f:
            f.write(r.content)

    if not os.path.exists(TAGS_PATH):
        print("Downloading tag list...", flush=True)
        r = requests.get(TAGS_URL, timeout=180)
        r.raise_for_status()
        with open(TAGS_PATH, "wb") as f:
            f.write(r.content)


def load_labels_no_pandas():
    tag_names = []
    rating_indexes = []
    general_indexes = []
    character_indexes = []

    with open(TAGS_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header

        # v3: tag_id, name, category, count
        for idx, row in enumerate(reader):
            if len(row) < 3:
                continue
            name = row[1].strip()
            category = int(row[2])
            tag_names.append(name)

            if category == 9:
                rating_indexes.append(idx)
            elif category == 0:
                general_indexes.append(idx)
            elif category == 4:
                character_indexes.append(idx)

    return tag_names, rating_indexes, general_indexes, character_indexes


def process_post_image(session, img: Image.Image, tag_data):
    tag_names, rating_idxs, general_idxs, char_idxs = tag_data

    # Input shape is typically [1, H, W, 3] in this model
    shape = session.get_inputs()[0].shape
    target_size = int(shape[1])

    image_input = prepare_image(img, target_size)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # WD v3 ONNX already outputs sigmoid probabilities
    preds = session.run([output_name], {input_name: image_input})[0][0].astype(float)

    # Ratings
    ratings = [(tag_names[i], preds[i]) for i in rating_idxs]
    best_rating, best_rating_p = max(ratings, key=lambda x: x[1])

    # General
    general = [(tag_names[i], preds[i]) for i in general_idxs]
    gen_thresh = GENERAL_THRESHOLD
    if USE_MCUT_GENERAL:
        gen_probs = np.array([p for _, p in general], dtype=float)
        gen_thresh = mcut_threshold(gen_probs)
    final_general = [(t, p) for (t, p) in general if p > gen_thresh]
    final_general.sort(key=lambda x: x[1], reverse=True)

    # Character
    chars = [(tag_names[i], preds[i]) for i in char_idxs]
    char_thresh = CHAR_THRESHOLD
    if USE_MCUT_CHARACTER:
        char_probs = np.array([p for _, p in chars], dtype=float)
        char_thresh = mcut_threshold(char_probs)
        char_thresh = max(0.15, char_thresh)
    final_chars = [(t, p) for (t, p) in chars if p > char_thresh]
    final_chars.sort(key=lambda x: x[1], reverse=True)

    return (best_rating, best_rating_p), final_chars, final_general


# -------------------- OXIBOORU API --------------------

def list_posts(limit=25, offset=0):
    query = quote("sort:id,desc")
    url = f"{OXI_BASE_URL}posts/"
    params = {"offset": offset, "limit": limit, "query": query}
    return request_json("GET", url, headers=token_headers(json=False), params=params, timeout=HTTP_TIMEOUT)


def get_post(post_id):
    url = f"{OXI_BASE_URL}post/{post_id}"
    return request_json("GET", url, headers=token_headers(json=False), timeout=HTTP_TIMEOUT)


def update_post(post_id, version, tags):
    url = f"{OXI_BASE_URL}post/{post_id}"
    payload = {"version": version, "tags": list(tags)}
    return request_json("PUT", url, headers=token_headers(json=True), json=payload, timeout=HTTP_TIMEOUT)


def tag_get(name):
    url = f"{OXI_BASE_URL}tag/{quote(name)}"
    r = requests.get(url, headers=token_headers(json=False), timeout=HTTP_TIMEOUT)
    if r.status_code == 200:
        try:
            return True, r.json()
        except Exception:
            return True, {}
    return False, None


def tag_create(name, category):
    url = f"{OXI_BASE_URL}tags"
    payload = {"names": [name], "category": category}
    r = requests.post(url, headers=token_headers(json=True), json=payload, timeout=HTTP_TIMEOUT)
    if r.status_code in (200, 201):
        print(f"Created tag {name} -> {category}", flush=True)
    elif r.status_code == 409:
        pass
    else:
        print(f"Create tag failed {name}: {r.status_code} {r.text[:200]}", flush=True)


def ensure_tag_category(name, category):
    exists, data = tag_get(name)
    if not exists:
        tag_create(name, category)
        return

    current_cat = (data or {}).get("category")
    version = (data or {}).get("version")
    if version and current_cat != category:
        url = f"{OXI_BASE_URL}tag/{quote(name)}"
        payload = {"version": version, "category": category}
        r = requests.put(url, headers=token_headers(json=True), json=payload, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            print(f"Failed moving tag {name}: {r.status_code} {r.text[:200]}", flush=True)


def load_image_for_post(pid: int, content_url: str) -> Image.Image:
    if content_url.startswith("data/"):
        local_path = "/" + content_url.lstrip("/")
        with open(local_path, "rb") as f:
            img = Image.open(f)
            img.load()
        return img

    full_url = urljoin(OXI_BASE_URL, content_url)
    resp = requests.get(full_url, headers=token_headers(json=False), timeout=IMG_TIMEOUT)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content))
    img.load()
    return img


# -------------------- MAIN --------------------

def main():
    if not OXI_USER or not OXI_TOKEN:
        print("Error: OXI_USER or OXI_TOKEN not set.", flush=True)
        return

    ensure_model_files()
    tag_data = load_labels_no_pandas()

    print(f"Loading Model: {ONNX_PATH}", flush=True)
    sess = rt.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    print(f"Bot started. Polling every {POLL_SECONDS}s... FORCE_RETAG={FORCE_RETAG} CLEAR_OLD_TAGS={CLEAR_OLD_TAGS}", flush=True)

    seen_ids = set()
    offset = 0
    limit = 25

    while True:
        try:
            data = list_posts(limit=limit, offset=0)
            posts = data.get("results", [])

            for post in posts:
                pid = post.get("id")
                if not pid:
                    continue

                if not FORCE_RETAG and pid in seen_ids:
                    continue

                if post.get("type") != "image":
                    seen_ids.add(pid)
                    continue

                existing_tags = set()
                for t in post.get("tags", []):
                    if isinstance(t, dict) and t.get("names"):
                        existing_tags.add(t["names"][0])
                    elif isinstance(t, str):
                        existing_tags.add(t)

                if not FORCE_RETAG and MARKER_TAG in existing_tags:
                    seen_ids.add(pid)
                    continue

                content_url = post.get("contentUrl")
                if not content_url:
                    seen_ids.add(pid)
                    continue

                try:
                    img = load_image_for_post(pid, content_url)
                    (rating, rating_p), char_tags_scored, general_tags_scored = process_post_image(sess, img, tag_data)

                    # pick best character (if any) for “always first tag”
                    char_names = [t for (t, _) in char_tags_scored]
                    general_names = [t for (t, _) in general_tags_scored]

                    best_char = char_names[0] if char_names else None
                    rest_chars = char_names[1:] if len(char_names) > 1 else []

                    # Ensure categories
                    if rating:
                        ensure_tag_category(rating, CAT_RATING)
                    for t in char_names:
                        ensure_tag_category(t, CAT_CHARACTER)
                    for t in general_names:
                        ensure_tag_category(t, CAT_GENERAL)

                    # Order: character first, then rating, then rest
                    new_tags = []
                    if best_char:
                        new_tags.append(best_char)
                    if rating:
                        new_tags.append(rating)
                    new_tags.extend(rest_chars)
                    new_tags.extend(general_names)
                    new_tags.append(MARKER_TAG)

                    if CLEAR_OLD_TAGS:
                        merged = new_tags
                    else:
                        merged = list(existing_tags)
                        for t in new_tags:
                            if t not in existing_tags:
                                merged.append(t)

                    full_post = get_post(pid)
                    version = full_post.get("version")
                    if not version:
                        print(f"[{pid}] missing version, skipping", flush=True)
                        seen_ids.add(pid)
                        continue

                    print(
                        f"[{pid}] update: best_char={best_char} rating={rating} "
                        f"chars={len(char_names)} general={len(general_names)} "
                        f"(thr gen={GENERAL_THRESHOLD}, char={CHAR_THRESHOLD}, mcut gen={USE_MCUT_GENERAL}, char={USE_MCUT_CHARACTER})",
                        flush=True,
                    )

                    update_post(pid, version, merged)
                    seen_ids.add(pid)
                    time.sleep(0.4)

                except Exception as e:
                    print(f"[{pid}] Failed: {e}", flush=True)
                    seen_ids.add(pid)

        except Exception as e:
            # THIS is where your “Expecting value…” becomes readable now
            print(f"Loop error: {e}", flush=True)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
