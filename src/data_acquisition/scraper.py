import os
import re
import unicodedata
from time import sleep
from urllib.parse import urlparse, urlunparse 

import jsonlines
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


class SuttaScraper:
    """
    A class to scrape, parse, and save Sutta texts from dhammatalks.org.
    """

    def __init__(self, config: dict):
        """
        Initializes the scraper with configuration settings.

        Args:
            config (dict): The application's configuration dictionary,
                           expected to contain a 'dhammatalks' key.
        """
        dhammatalks_config = config['dhammatalks']
        self.master_url = dhammatalks_config['master_url']
        self.base_url = dhammatalks_config['base_url']
        self.books = dhammatalks_config['books_of_interest']
        self.avoid = dhammatalks_config['avoid_in_url']
        print("SuttaScraper initialized.")

    def get_sutta_links(self) -> list[dict]:
        """Scrapes the index page to find all relevant sutta URLs."""
        print(f"Fetching master list from {self.master_url}...")
        try:
            response = requests.get(self.master_url, timeout=30)
            response.raise_for_status()
            response.encoding = "UTF-8"
            soup = BeautifulSoup(response.text, "html.parser")

            links = soup.find_all("a")
            unique_urls = {} 

            for link in links:
                href = link.get('href')
                if href and any(book in href for book in self.books) and not any(av in href for av in self.avoid):
                    
                    # 1. Create initial full URL
                    full_url = f"{self.base_url}{href}"
                    # 2. Parse the URL into its components
                    parsed_url = urlparse(full_url)
                    # 3. Normalize the path: remove trailing slashes and make it lowercase
                    path = parsed_url.path.rstrip('/').lower()
                    # 4. Rebuild the URL without fragments (#) or queries (?)
                    normalized_url = urlunparse(
                        (parsed_url.scheme, parsed_url.netloc, path, 
                         '', '', '') # Empty params, query, and fragment
                    )

                    if normalized_url  not in unique_urls: # Deduplicate by the full URL
                        # 1. Get the filename part of the href (e.g., 'MN131.html')
                        filename = os.path.basename(href)
                        
                        # 2. Split the filename from its extension to get the clean ID
                        sutta_code, _ = os.path.splitext(filename) # -> ('MN131', '.html')

                        href_split = href.split("/")

                        unique_urls[normalized_url] = {
                            "sutta_id": sutta_code,
                            "book": href_split[2],
                            "sub_book": href_split[3] if len(href_split) > 4 else "None",
                            "url": full_url, # We save the original, clickable URL
                            "sutta_id_text": re.sub(r" +", " ", unicodedata.normalize("NFKD", link.get_text()))
                        }

            link_dicts = list(unique_urls.values())
            link_dicts.sort(key=lambda x: x['url'])
            print(f"Found and sorted {len(link_dicts)} sutta links to process.")
            return link_dicts
        except requests.RequestException as e:
            print(f"Failed to connect to master URL: {e}")
            return []

    def _parse_sutta_page(self, html_content: str) -> dict | None:
        """
        Parses the HTML of a single sutta page to extract structured data.
        This is an internal helper method.
        """
        soup = BeautifulSoup(html_content, "html.parser")
        sutta_div = soup.find("div", id="sutta")
        if not sutta_div:
            return None

        title_attr = sutta_div.find("h1")
        title_text = re.sub(r'\s{2,}', ' ', unicodedata.normalize("NFKD", title_attr.get_text()).strip()) if title_attr else None
        if title_attr:
            title_attr.decompose()

        # Decompose non-essential sections
        for tag in sutta_div.find_all(["p", "div"], class_=["seealso", "note"]):
            if tag:
                tag.decompose()

        full_text = unicodedata.normalize("NFKD", sutta_div.get_text()).strip()

        sutta_body = ""
        intro_text = ""
        if '* * *' in full_text:
            parts = full_text.split('* * *', 1)
            intro_text = parts[0].strip()
            sutta_body = parts[1].strip()
        else:
            sutta_body = full_text

        return {
            "title": title_text,
            "introduction": intro_text,
            "body": sutta_body,
        }

    def run(self, output_path: str):
        """
        Main method to run the full scraping and parsing pipeline.

        Args:
            output_path (str): The absolute path to the output .jsonl file.
        """
        sutta_links = self.get_sutta_links()

        if not sutta_links:
            print("No sutta links found. Exiting.")
            return

        # Ensure the directory exists before writing
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"Scraping suttas and saving to {output_path}...")

        suttas_saved_count = 1

        # Overwrite the file from scratch to ensure UID consistency
        with jsonlines.open(output_path, mode='w') as writer:
            for link_info in tqdm(sutta_links, desc="Scraping Suttas"):
                try:
                    response = requests.get(link_info['url'], timeout=30)
                    response.raise_for_status()
                    response.encoding = "UTF-8"

                    parsed_data = self._parse_sutta_page(response.text)

                    if parsed_data:
                        final_record = {**link_info, **parsed_data, }
                        writer.write(final_record)
                        suttas_saved_count += 1

                    sleep(0.1)

                except requests.RequestException as e:
                    print(f"Could not fetch {link_info['url']}: {e}")
        print(f"A total of {suttas_saved_count-1} suttas were scraped and saved.")


# --- This part is kept for potential direct execution or for clarity ---
# It shows how the class is intended to be used.
# The actual execution is handled by the script in `scripts/`.
def run_scraper(config: dict, output_path: str):
    """
    Initializes and runs the SuttaScraper.
    This function acts as a bridge between the script runner and the class.

    Args:
        config (dict): The application configuration dictionary.
        output_path (str): The absolute path to the output .jsonl file.
    """
    scraper = SuttaScraper(config)
    scraper.run(output_path)