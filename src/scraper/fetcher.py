import os
import time
import urllib.parse
import logging
import requests

logger = logging.getLogger(__name__)

class Fetcher:
    """
    Layer 1 & 2: Fetch and Cache Layer.
    Fetches HTML from URLs with polite delays and caches them to disk.
    """
    def __init__(self, cache_dir, delay_sec=2.0, user_agent="Mozilla/5.0", use_cache=True):
        self.cache_dir = cache_dir
        self.delay_sec = delay_sec
        self.headers = {"User-Agent": user_agent}
        self.use_cache = use_cache
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _get_cache_path(self, url):
        # Create a safe filename from the URL query string
        parsed = urllib.parse.urlparse(url)
        safe_name = parsed.path.split('/')[-1] + "_" + urllib.parse.quote_plus(parsed.query) + ".html"
        return os.path.join(self.cache_dir, safe_name)
        
    def fetch(self, url):
        """Returns (html_content, source_html_file)"""
        cache_path = self._get_cache_path(url)
        
        if self.use_cache and os.path.exists(cache_path):
            logger.info(f"CACHE HIT: {url}")
            with open(cache_path, 'r', encoding='utf-8') as f:
                return f.read(), cache_path
                
        logger.info(f"FETCHING: {url}")
        time.sleep(self.delay_sec)
        
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            html = response.text
            
            if self.use_cache:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(html)
                    
            return html, cache_path
            
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None, None
