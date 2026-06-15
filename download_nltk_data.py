import os
import urllib.request
import zipfile

def download_and_extract(url, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    filename = url.split('/')[-1]
    zip_path = os.path.join(target_dir, filename)
    
    print(f"Downloading {url} to {zip_path}...")
    urllib.request.urlretrieve(url, zip_path)
    
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
        
    print(f"Cleaning up {zip_path}...")
    os.remove(zip_path)
    print("Done!")

if __name__ == "__main__":
    base_dir = os.path.expanduser("~/nltk_data")
    corpora_dir = os.path.join(base_dir, "corpora")
    tokenizers_dir = os.path.join(base_dir, "tokenizers")
    
    urls = {
        "wordnet": ("https://fastly.jsdelivr.net/gh/nltk/nltk_data@gh-pages/packages/corpora/wordnet.zip", corpora_dir),
        "brown": ("https://fastly.jsdelivr.net/gh/nltk/nltk_data@gh-pages/packages/corpora/brown.zip", corpora_dir),
        "omw-1.4": ("https://fastly.jsdelivr.net/gh/nltk/nltk_data@gh-pages/packages/corpora/omw-1.4.zip", corpora_dir),
        "punkt": ("https://fastly.jsdelivr.net/gh/nltk/nltk_data@gh-pages/packages/tokenizers/punkt.zip", tokenizers_dir)
    }
    
    for name, (url, target) in urls.items():
        try:
            print(f"Processing {name}...")
            download_and_extract(url, target)
        except Exception as e:
            print(f"Error processing {name}: {e}")
            
    print("All NLTK downloads complete!")
