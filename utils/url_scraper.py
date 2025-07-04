import requests
from bs4 import BeautifulSoup

def extract_from_url(url):
    try:
        html = requests.get(url, timeout=5).text
        soup = BeautifulSoup(html, "html.parser")
        texts = soup.find_all("p")
        return " ".join([t.get_text() for t in texts])
    except Exception as e:
        return f"\u274c Failed to extract: {e}"
