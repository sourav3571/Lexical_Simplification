"""
download_datasets.py
Downloads all datasets required by SignDecoder's NLP pipeline
to their exact expected locations.

Run from: d:/SignDecoder/model for lexical simplification/
    python download_datasets.py
"""

import os
import sys
import csv
import json
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def status(msg):
    print("\n" + "="*60)
    print("  " + msg)
    print("="*60)

def ok(msg):    print("  [OK]   " + msg)
def skip(msg):  print("  [SKIP] " + msg)
def fail(msg):  print("  [FAIL] " + msg)
def info(msg):  print("         " + msg)

def fetch(url, dest):
    """Download url to dest, return (success, size_kb)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            f.write(r.read())
        return True, os.path.getsize(dest) // 1024
    except Exception as e:
        return False, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# 1. MAGPIE Idiom Corpus  -->  data/magpie/
# ─────────────────────────────────────────────────────────────────────────────
def download_magpie():
    status("MAGPIE Idiom Corpus  -->  data/magpie/")
    dest_dir  = os.path.join(BASE_DIR, "data", "magpie")
    dest_file = os.path.join(dest_dir, "MAGPIE_filtered_split_random.jsonl")
    os.makedirs(dest_dir, exist_ok=True)

    if os.path.exists(dest_file) and os.path.getsize(dest_file) > 10000:
        skip("Already present (" + str(os.path.getsize(dest_file) // 1024) + " KB)")
        return True

    url = "https://github.com/hslh/magpie-corpus/raw/refs/heads/master/MAGPIE_filtered_split_random.jsonl"
    info("Downloading: " + url)
    success, result = fetch(url, dest_file)
    if success:
        ok("Saved: data/magpie/MAGPIE_filtered_split_random.jsonl (" + str(result) + " KB)")
        return True
    else:
        fail("Download failed: " + str(result))
        info("Manual: https://github.com/hslh/magpie-corpus")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 2. VUA Amsterdam Metaphor Corpus  -->  data/vua/
#    Saves train/val/test as separate files
# ─────────────────────────────────────────────────────────────────────────────
def download_vua():
    status("VU Amsterdam Metaphor Corpus  -->  data/vua/")
    dest_dir = os.path.join(BASE_DIR, "data", "vua")
    os.makedirs(dest_dir, exist_ok=True)

    # Check if already present (main output file)
    out_file = os.path.join(dest_dir, "vua_processed.tsv")
    if os.path.exists(out_file) and os.path.getsize(out_file) > 10000:
        skip("Already present (" + str(os.path.getsize(out_file) // 1024) + " KB)")
        return True

    commit = "fc5e569fe321ba5a7404cf7416cbe23026aeaf79"
    base   = "https://github.com/gao-g/metaphor-in-context/raw/" + commit + "/data/VUA/"

    # VUA classification splits (sentence, word-level label)
    files = [
        (base + "VUA_formatted_train.csv", "vua_train.csv"),
        (base + "VUA_formatted_val.csv",   "vua_val.csv"),
        (base + "VUA_formatted_test.csv",  "vua_test.csv"),
    ]

    any_ok = False
    for url, fname in files:
        dest = os.path.join(dest_dir, fname)
        info("Downloading " + fname + "...")
        success, result = fetch(url, dest)
        if success:
            ok("  Saved data/vua/" + fname + " (" + str(result) + " KB)")
            any_ok = True
        else:
            fail("  " + fname + " failed: " + str(result))

    if not any_ok:
        info("Manual: https://github.com/gao-g/metaphor-in-context/tree/master/data/VUA")
        return False

    # Merge all into vua_processed.tsv  (format data_prep.py reads)
    info("Merging into vua_processed.tsv...")
    try:
        rows = []
        for _, fname in files:
            path = os.path.join(dest_dir, fname)
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
        with open(out_file, "w", encoding="utf-8", newline="") as f:
            f.write("id\tlabel\tpos\tword\tsentence\n")
            for i, row in enumerate(rows):
                sent  = row.get("sentence", "").replace("\t", " ")
                word  = row.get("word_index", row.get("verb", "")).replace("\t", " ")
                label = str(row.get("label", "0"))
                pos   = str(row.get("POS", "NN"))
                f.write(str(i) + "\t" + label + "\t" + pos + "\t" + word + "\t" + sent + "\n")
        ok("Merged into data/vua/vua_processed.tsv (" + str(os.path.getsize(out_file) // 1024) + " KB, " + str(len(rows)) + " rows)")
        return True
    except Exception as e:
        fail("Merge failed: " + str(e))
        return any_ok


# ─────────────────────────────────────────────────────────────────────────────
# 3. CWI 2018 Shared Task  -->  cwi2018_data/
# ─────────────────────────────────────────────────────────────────────────────
def download_cwi2018():
    status("CWI 2018 Shared Task  -->  cwi2018_data/")
    dest_dir = os.path.join(BASE_DIR, "cwi2018_data")
    os.makedirs(dest_dir, exist_ok=True)

    existing = [f for f in os.listdir(dest_dir) if f.endswith(".tsv")]
    if existing:
        total = sum(os.path.getsize(os.path.join(dest_dir, f)) for f in existing)
        skip("Already present (" + str(len(existing)) + " TSV files, " + str(total // 1024) + " KB)")
        return True

    base = "https://github.com/sheffieldnlp/cwisharedtask2018-teaching/raw/master/datasets/english/"
    files = [
        (base + "English_Train.tsv", "cwi2018_train.tsv"),
        (base + "English_Dev.tsv",   "cwi2018_dev.tsv"),
        (base + "English_Test.tsv",  "cwi2018_test.tsv"),
    ]

    info("Downloading from Sheffield NLP CWI 2018 repo...")
    any_ok = False
    for url, fname in files:
        dest = os.path.join(dest_dir, fname)
        info("Downloading " + fname + "...")
        success, result = fetch(url, dest)
        if success:
            ok("  Saved cwi2018_data/" + fname + " (" + str(result) + " KB)")
            any_ok = True
        else:
            fail("  " + fname + " failed: " + str(result))

    if not any_ok:
        info("Manual: https://sites.google.com/site/complexwordidentification2018/datasets")
    return any_ok


# ─────────────────────────────────────────────────────────────────────────────
# 4. Verify already-present datasets
# ─────────────────────────────────────────────────────────────────────────────
def verify_existing():
    status("Verifying already-present datasets")
    checks = [
        ("BenchLS.txt",               "BenchLS -- substitute ranking pairs (~929 sentences)"),
        ("lex_mturk.txt",             "LexMTurk -- crowdsourced substitutions (~500 sentences)"),
        ("ppdb_fallback.json",        "PPDB fallback -- paraphrase database subset"),
        ("augmented_cwi_data.json",   "Augmented CWI data"),
        ("data/idiom_train.json",     "Synthetic idiom train split"),
        ("data/metaphor_train.json",  "Synthetic metaphor train split"),
    ]
    for rel_path, label in checks:
        full = os.path.join(BASE_DIR, rel_path)
        if os.path.exists(full):
            kb = os.path.getsize(full) // 1024
            ok(label + "  [" + rel_path + "] (" + str(kb) + " KB)")
        else:
            fail("Missing: " + rel_path + "  [" + label + "]")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  SignDecoder -- Dataset Downloader")
    print("="*60)

    verify_existing()
    r1 = download_magpie()
    r2 = download_vua()
    r3 = download_cwi2018()

    print("\n" + "="*60)
    print("  FINAL SUMMARY")
    print("="*60)
    results = [
        ("MAGPIE Idiom Corpus   --> data/magpie/MAGPIE_filtered_split_random.jsonl", r1),
        ("VUAMC Metaphor Corpus --> data/vua/vua_processed.tsv",                    r2),
        ("CWI 2018 Shared Task  --> cwi2018_data/*.tsv",                             r3),
    ]
    for name, success in results:
        print("  " + ("[OK]  " if success else "[FAIL]") + " " + name)

    if all(r for _, r in results):
        print("\nAll datasets ready!")
    else:
        print("\nSome downloads failed. Check above for manual download links.")
    print("="*60 + "\n")
