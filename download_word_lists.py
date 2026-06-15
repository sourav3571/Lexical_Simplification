import urllib.request
import os

def download_file(url, filename):
    print(f"Downloading {url} to {filename}...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            with open(filename, 'wb') as out_file:
                out_file.write(response.read())
        print("Success!")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

# Dale-Chall easy words from textstat
dale_chall_url = "https://raw.githubusercontent.com/textstat/textstat/master/textstat/resources/en/easy_words.txt"
download_file(dale_chall_url, "dale_chall.txt")

# Oxford 3000 words
oxford_url = "https://raw.githubusercontent.com/sapbmw/The-Oxford-3000/master/The_Oxford_3000.txt"
download_file(oxford_url, "oxford3000.txt")

# Check file sizes and contents
if os.path.exists("dale_chall.txt"):
    with open("dale_chall.txt", "r", encoding="utf-8") as f:
        dc_words = [line.strip().lower() for line in f if line.strip()]
        print(f"Dale-Chall word count: {len(dc_words)}, Sample: {dc_words[:5]}")

if os.path.exists("oxford3000.txt"):
    with open("oxford3000.txt", "r", encoding="utf-8") as f:
        ox_words = [line.strip().lower() for line in f if line.strip()]
        print(f"Oxford 3000 word count: {len(ox_words)}, Sample: {ox_words[:5]}")