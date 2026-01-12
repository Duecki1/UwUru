import os
import time
import base64
import csv
import numpy as np
import requests
import onnxruntime as rt
from io import BytesIO
from PIL import Image
from urllib.parse import urljoin, quote

# -------------------- ENV --------------------

OXI_BASE_URL = os.getenv("OXI_BASE_URL", "http://server:6666").rstrip("/") + "/"
OXI_USER = os.getenv("OXI_USER")
OXI_TOKEN = os.getenv("OXI_TOKEN")
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "5"))

# Thresholds
GENERAL_THRESHOLD = float(os.getenv("GENERAL_THRESHOLD", "0.35"))
CHAR_THRESHOLD = float(os.getenv("CHAR_THRESHOLD", "0.60"))

# MCut
USE_MCUT_GENERAL = os.getenv("USE_MCUT_GENERAL", "1") == "1"
USE_MCUT_CHARACTER = os.getenv("USE_MCUT_CHARACTER", "0") == "1"

MODEL_DIR = "/app/model"
ONNX_URL = "https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/model.onnx"
TAGS_URL = "https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/resolve/main/selected_tags.csv"
ONNX_PATH = os.path.join(MODEL_DIR, "model.onnx")
TAGS_PATH = os.path.join(MODEL_DIR, "selected_tags.csv")

MARKER_TAG = os.getenv("MARKER_TAG", "auto_tagged_wd3")

# -------------------- AUTH --------------------

def token_headers(json=True):
    raw = f"{OXI_USER}:{OXI_TOKEN}".encode("utf-8")
    b64 = base64.b64encode(raw).decode("ascii")
    h = {
        "Authorization": f"Token {b64}",
        "Accept": "application/json",
    }
    if json:
        h["Content-Type"] = "application/json"
    return h

# -------------------- LOGIC --------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mcut_threshold(probs):
    sorted_probs = probs[probs.argsort()[::-1]]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    t = difs.argmax()
    thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
    return thresh

def prepare_image(image: Image.Image, target_size: int):
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    canvas = Image.new("RGBA", image.size, (255, 255, 255))
    canvas.alpha_composite(image)
    image = canvas.convert("RGB")

    w, h = image.size
    max_dim = max(w, h)
    pad_left = (max_dim - w) // 2
    pad_top = (max_dim - h) // 2

    padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded_image.paste(image, (pad_left, pad_top))

    if max_dim != target_size:
        padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)

    image_array = np.asarray(padded_image, dtype=np.float32)
    image_array = image_array[:, :, ::-1]  # RGB -> BGR
    return np.expand_dims(image_array, axis=0)

# -------------------- MODEL LOADING --------------------

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
    tag_names, rating_indexes, general_indexes, character_indexes = [], [], [], []
    with open(TAGS_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for idx, row in enumerate(reader):
            if len(row) < 3:
                continue
            name = row[1]
            category = int(row[2])
            tag_names.append(name)

            if category == 9:
                rating_indexes.append(idx)
            elif category == 0:
                general_indexes.append(idx)
            elif category == 4:
                character_indexes.append(idx)

    return tag_names, rating_indexes, general_indexes, character_indexes

# -------------------- TAG CATEGORY FIX (OUTLINES) --------------------

def get_tag(tagname: str):
    r = requests.get(f"{OXI_BASE_URL}tag/{quote(tagname)}", headers=token_headers(), timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()

def set_tag_category(tagname: str, tag_version: str, category: str):
    r = requests.put(
        f"{OXI_BASE_URL}tag/{quote(tagname)}",
        headers=token_headers(),
        json={"version": tag_version, "category": category},
        timeout=30
    )
    r.raise_for_status()

def ensure_tag_category(tagname: str, category: str):
    tag = get_tag(tagname)
    if not tag:
        return
    if tag.get("category") == category:
        return
    set_tag_category(tagname, tag["version"], category)

# -------------------- INFERENCE --------------------

def process_post_image(session, img: Image.Image, tag_data):
    tag_names, rating_idxs, general_idxs, char_idxs = tag_data

    input_shape = session.get_inputs()[0].shape
    target_size = input_shape[1] if isinstance(input_shape[1], int) else 448

    image_input = prepare_image(img, target_size)
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name

    raw_preds = session.run([label_name], {input_name: image_input})[0][0]

    if np.min(raw_preds) < 0 or np.max(raw_preds) > 1.0:
        preds = sigmoid(raw_preds)
    else:
        preds = raw_preds.astype(float)

    ratings_names = [(tag_names[i], preds[i]) for i in rating_idxs]
    best_rating = max(ratings_names, key=lambda x: x[1])[0] if ratings_names else None

    general_res = [(tag_names[i], preds[i]) for i in general_idxs]
    if USE_MCUT_GENERAL:
        gen_probs = np.array([x[1] for x in general_res])
        gen_thresh = mcut_threshold(gen_probs)
        gen_thresh = max(GENERAL_THRESHOLD, gen_thresh)
    else:
        gen_thresh = GENERAL_THRESHOLD
    final_general = [x[0] for x in general_res if x[1] > gen_thresh]

    char_res = [(tag_names[i], preds[i]) for i in char_idxs]
    if USE_MCUT_CHARACTER:
        char_probs = np.array([x[1] for x in char_res])
        char_thresh = mcut_threshold(char_probs)
        char_thresh = max(0.15, char_thresh)
    else:
        char_thresh = CHAR_THRESHOLD
    final_chars = [x[0] for x in char_res if x[1] > char_thresh]

    # characters first, then general, then rating, then marker
    result_tags = []
    result_tags.extend(final_chars)
    result_tags.extend(final_general)
    if best_rating:
        result_tags.append(best_rating)
    result_tags.append(MARKER_TAG)

    return result_tags, final_chars, final_general, best_rating

# -------------------- API --------------------

def list_posts(limit=25, offset=0):
    query = quote("sort:id,desc")
    url = f"{OXI_BASE_URL}posts/?offset={offset}&limit={limit}&query={query}"
    r = requests.get(url, headers=token_headers(), timeout=30)
    r.raise_for_status()
    return r.json()

def get_post(post_id):
    r = requests.get(f"{OXI_BASE_URL}post/{post_id}", headers=token_headers(), timeout=30)
    r.raise_for_status()
    return r.json()

def update_post(post_id, version, tags):
    payload = {"version": version, "tags": list(tags)}
    r = requests.put(f"{OXI_BASE_URL}post/{post_id}", headers=token_headers(), json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def load_image_for_post(pid: int, content_url: str) -> Image.Image:
    if content_url.startswith("data/"):
        local_path = "/" + content_url.lstrip("/")
        with open(local_path, "rb") as f:
            img = Image.open(f)
            img.load()
        return img

    full_url = urljoin(OXI_BASE_URL, content_url)
    img_bytes = requests.get(full_url, headers=token_headers(json=False), timeout=60).content
    img = Image.open(BytesIO(img_bytes))
    img.load()
    return img

def extract_existing_tag_names(post_obj) -> set:
    existing = set()
    for t in post_obj.get("tags", []):
        if isinstance(t, dict) and t.get("names"):
            existing.add(t["names"][0])
        elif isinstance(t, str):
            existing.add(t)
    return existing

# -------------------- MAIN --------------------

def main():
    if not OXI_USER or not OXI_TOKEN:
        print("Error: OXI_USER or OXI_TOKEN not set.")
        return

    ensure_model_files()
    tag_data = load_labels_no_pandas()

    tag_names, rating_idxs, general_idxs, char_idxs = tag_data

    rating_set = set(tag_names[i] for i in rating_idxs)
    general_set = set(tag_names[i] for i in general_idxs)
    char_set = set(tag_names[i] for i in char_idxs)

    def desired_category(tagname: str):
        if tagname in char_set:
            return "character"
        if tagname in rating_set:
            return "rating"
        if tagname in general_set:
            return "general"
        return None

    print(f"Loading Model: {ONNX_PATH}", flush=True)
    sess = rt.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    current_offset = 0
    PAGE_SIZE = 50

    print("Bot started. Scanning ALL images in database...", flush=True)

    while True:
        try:
            print(f"-- Fetching page at offset {current_offset} --", flush=True)
            data = list_posts(limit=PAGE_SIZE, offset=current_offset)
            posts = data.get("results", [])

            if not posts:
                print("End of database reached. Restarting scan from top in 60s...", flush=True)
                current_offset = 0
                time.sleep(60)
                continue

            for post in posts:
                pid = post.get("id")
                if post.get("type") != "image":
                    continue

                content_url = post.get("contentUrl")
                if not content_url:
                    continue

                # ✅ skip already-tagged posts (marker present)
                existing_tags = extract_existing_tag_names(post)
                if MARKER_TAG in existing_tags:
                    continue

                try:
                    img = load_image_for_post(pid, content_url)

                    new_tags, chars, generals, rating = process_post_image(sess, img, tag_data)

                    # ✅ merge (don’t overwrite)
                    # keep existing order, append new ones that are missing
                    merged = list(existing_tags)
                    added = []
                    for t in new_tags:
                        if t not in existing_tags:
                            merged.append(t)
                            added.append(t)

                    # if nothing new to add, skip update
                    if not added:
                        continue

                    full_post = get_post(pid)
                    version = full_post.get("version")

                    print(f"[{pid}] adding {len(added)} tags (total after={len(merged)})", flush=True)
                    update_post(pid, version, merged)

                    # ✅ OUTLINE FIX only for newly added tags (less API spam)
                    for t in added:
                        cat = desired_category(t)
                        if cat:
                            try:
                                ensure_tag_category(t, cat)
                            except Exception as e:
                                print(f"[{pid}] could not set category for {t}: {e}", flush=True)

                    time.sleep(0.2)

                except Exception as e:
                    print(f"[{pid}] Error: {e}", flush=True)

            current_offset += PAGE_SIZE

        except Exception as e:
            print(f"Main Loop Error: {e}", flush=True)
            time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
