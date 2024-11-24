import os
import subprocess

def download_spacy_model():
    try:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        print("Downloaded en_core_web_sm successfully!")
    except subprocess.CalledProcessError as e:
        print("Error downloading en_core_web_sm:", e)

if __name__ == "__main__":
    download_spacy_model()