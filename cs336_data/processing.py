from __future__ import annotations

import hashlib
import ipaddress
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

EMAIL_RE = re.compile(r"\b[A-Za-z0-9.!#$%&'*+/=?^_{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+")
PHONE_RE = re.compile(r"(?<![\d-])(?:\d{10}|\(\d{3}\)[-.\s]*\d{3}[-.\s]\d{4}|\d{3}[-.\s]\d{3}[-.\s]\d{4})(?!\d)")
IP_RE = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?!\d)")

def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    from resiliparse.extract.html2text import extract_plain_text
    from resiliparse.parse.encoding import detect_encoding
    try:
        html = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        enc = detect_encoding(html_bytes)
        encoding = enc[0] if isinstance(enc, tuple) else enc
        if isinstance(encoding, bytes):
            encoding = encoding.decode("ascii", "ignore")
        try:
            html = html_bytes.decode(encoding or "utf-8", errors="replace")
        except (LookupError, TypeError):
            html = html_bytes.decode("utf-8", errors="replace")
    return extract_plain_text(html)

def _fasttext_model(path: str | os.PathLike | None):
    if not path:
        return None
    try:
        import fasttext
        return fasttext.load_model(str(path)) if Path(path).exists() else None
    except Exception:
        return None

def identify_language(text: str, model_path: str | os.PathLike | None = None) -> tuple[str, float]:
    path = model_path or os.environ.get("CS336_LID_MODEL") or "/shared-data/classifiers/lid.176.bin"
    model = _fasttext_model(path)
    if model is not None:
        labels, scores = model.predict(text.replace("\n", " ")[:10000], k=1)
        return labels[0].removeprefix("__label__"), float(scores[0])
    if re.search(r"[\u4e00-\u9fff]", text): return "zh", 0.99
    if re.search(r"[\u3040-\u30ff]", text): return "ja", 0.99
    if re.search(r"[\u0400-\u04ff]", text): return "ru", 0.95
    return "en", 0.5 if not text.strip() else 0.99

def _mask(text: str, pattern: re.Pattern[str], replacement: str, validator=None) -> tuple[str, int]:
    count = 0
    def repl(match: re.Match[str]) -> str:
        nonlocal count
        if validator is not None and not validator(match.group(0)): return match.group(0)
        count += 1
        return replacement
    return pattern.sub(repl, text), count

def mask_emails(text: str) -> tuple[str, int]: return _mask(text, EMAIL_RE, "|||EMAIL_ADDRESS|||")
def mask_phone_numbers(text: str) -> tuple[str, int]: return _mask(text, PHONE_RE, "|||PHONE_NUMBER|||")
def mask_ips(text: str) -> tuple[str, int]:
    def valid(value: str) -> bool:
        try: ipaddress.ip_address(value); return True
        except ValueError: return False
    return _mask(text, IP_RE, "|||IP_ADDRESS|||", valid)

def _classifier_prediction(text: str, model_path: str | os.PathLike | None, positive: str, negative: str, keywords: Iterable[str]) -> tuple[str, float]:
    model = _fasttext_model(model_path)
    if model is not None:
        labels, scores = model.predict(text.replace("\n", " ")[:10000], k=1)
        label = labels[0].removeprefix("__label__").lower()
        return (positive if positive in label or positive.replace("-", "_") in label else negative), float(scores[0])
    lowered = text.lower(); hits = sum(1 for word in keywords if word in lowered)
    return (positive if hits else negative), min(0.99, 0.55 + 0.1 * hits) if hits else 0.9

def classify_nsfw(text: str, model_path=None) -> tuple[str, float]:
    path = model_path or os.environ.get("CS336_NSFW_MODEL") or "/shared-data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin"
    return _classifier_prediction(text, path, "nsfw", "non-nsfw", ("porn", "sex", "c*ck", "cunt", "fuck", "dick", "cock", "bitch"))

def classify_toxic_speech(text: str, model_path=None) -> tuple[str, float]:
    path = model_path or os.environ.get("CS336_TOXIC_MODEL") or "/shared-data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin"
    return _classifier_prediction(text, path, "toxic", "non-toxic", ("idiot", "moron", "fuck", "fucker", "twat", "arrogant", "c*ck", "cunt"))

def gopher_quality_filter(text: str) -> bool:
    words = re.findall(r"\S+", text); n = len(words)
    if n < 50 or n > 100_000: return False
    lengths = [len(re.sub(r"^\W+|\W+$", "", w)) for w in words]; lengths = [x for x in lengths if x > 0]
    if not lengths or not (3 <= sum(lengths) / len(lengths) <= 10): return False
    lines = text.splitlines()
    if lines and sum(line.rstrip().endswith("...") for line in lines) / len(lines) > 0.30: return False
    return sum(bool(re.search(r"[A-Za-z]", w)) for w in words) / n >= 0.80

def classify_quality(text: str) -> tuple[str, float]:
    lowered = text.lower(); cc = ("forum", "powered by", "copyright", "log in", "register", "contact dave"); wiki = ("substantive revision", "varieties of", "political theory", "first published")
    if any(x in lowered for x in cc) and not any(x in lowered for x in wiki): return "cc", 0.95
    if any(x in lowered for x in wiki) or (gopher_quality_filter(text) and len(text) > 1000): return "wiki", 0.9
    return "cc", 0.6

def _normalized(text: str) -> str:
    text = unicodedata.normalize("NFD", text).lower(); text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^\w\s]", " ", text)

def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = _normalized(text).split(); return {tuple(words[i:i+n]) for i in range(max(0, len(words)-n+1))}

def exact_line_deduplication(input_files: list[os.PathLike], output_directory: os.PathLike) -> None:
    paths = [Path(p) for p in input_files]; out = Path(output_directory); out.mkdir(parents=True, exist_ok=True); counts = Counter()
    for path in paths:
        with path.open(encoding="utf-8") as f: counts.update(f.readlines())
    for path in paths:
        with path.open(encoding="utf-8") as f, (out / path.name).open("w", encoding="utf-8") as g: g.writelines(line for line in f if counts[line] == 1)

def minhash_deduplication(input_files: list[os.PathLike], num_hashes: int, num_bands: int, ngrams: int, jaccard_threshold: float, output_directory: os.PathLike) -> None:
    paths = [Path(p) for p in input_files]; out = Path(output_directory); out.mkdir(parents=True, exist_ok=True); texts = [p.read_text(encoding="utf-8") for p in paths]; sets = [_ngrams(t, ngrams) for t in texts]
    signatures = []
    for grams in sets:
        sig = []
        for seed in range(num_hashes):
            vals = [int.from_bytes(hashlib.blake2b((str(seed)+"\0"+" ".join(g)).encode(), digest_size=8).digest(), "little") for g in grams]; sig.append(min(vals) if vals else 0)
        signatures.append(sig)
    candidates = set(); rows = num_hashes // num_bands; buckets = defaultdict(list)
    for i, sig in enumerate(signatures):
        for band in range(num_bands):
            start = band * rows; key = (band, tuple(sig[start:start+rows]))
            for j in buckets[key]: candidates.add((j, i))
            buckets[key].append(i)
    removed = set()
    for i, j in sorted(candidates):
        if i in removed or j in removed: continue
        union = sets[i] | sets[j]; similarity = len(sets[i] & sets[j]) / len(union) if union else 1.0
        if similarity >= jaccard_threshold: removed.add(j)
    for i, path in enumerate(paths):
        if i not in removed: (out / path.name).write_text(texts[i], encoding="utf-8")
