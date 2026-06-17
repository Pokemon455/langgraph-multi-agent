from langchain_core.tools import tool as tool_decorator
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor
import requests
import config

_SKIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "menu"}
_NAV_KW = [
    "Log in", "Sign up", "Cookie", "Subscribe",
    "Newsletter", "Privacy", "Terms", "Click here",
    "Skip to", "Jump to"
]


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text, self.skip = [], False

    def handle_starttag(self, tag, attrs):
        self.skip = tag in _SKIP_TAGS

    def handle_endtag(self, tag):
        self.skip = False

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.text.append(data.strip())


def _fetch_clean_content(url: str, max_chars: int = 1500):
    try:
        page = requests.get(url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        if any(w in page.text for w in ["Enable JavaScript", "Just a moment", "Cloudflare"]):
            return None
        p = _TextExtractor()
        p.feed(page.text)
        raw = " ".join(" ".join(p.text).split())
        sentences = [s.strip() for s in raw.split(".") if len(s.strip()) > 50]
        clean = [s for s in sentences if not any(k.lower() in s.lower() for k in _NAV_KW)]
        result = ". ".join(clean[:15])[:max_chars]
        return result if len(result) > 100 else None
    except Exception:
        return None


@tool_decorator
def search_web(query: str) -> str:
    """Search Google for current, real-time information."""
    try:
        resp = requests.post(
            "https://api.apify.com/v2/acts/apify~google-search-scraper/run-sync-get-dataset-items",
            params={"token": config.APIFY_TOKEN},
            json={
                "queries": query,
                "maxPagesPerQuery": 1,
                "resultsPerPage": 8,
                "languageCode": "en",
                "countryCode": "us"
            },
            timeout=60
        )
        organic = resp.json()[0].get("organicResults", [])
        if not organic:
            return "No search results found."

        filtered = [
            r for r in organic
            if not any(s in r.get("url", "") for s in ["youtube.com", "reddit.com", "twitter.com"])
        ][:5]

        contents = list(ThreadPoolExecutor(3).map(
            _fetch_clean_content,
            [r.get("url", "") for r in filtered[:3]]
        ))

        out = f"Search: '{query}'\n" + "=" * 50 + "\n\n"
        for i, r in enumerate(filtered):
            out += f"[{i+1}] {r.get('title', '')}\n"
            out += f"Snippet: {r.get('description', '').replace('Read more', '').strip()}\n"
            if i < 3 and contents[i]:
                out += f"Content: {contents[i]}\n"
            out += "\n" + "-" * 45 + "\n\n"
        return out
    except Exception as e:
        return f"Search error: {e}"
