import requests
import unicodedata
import re
import jsonlines
from bs4 import BeautifulSoup
from tqdm import tqdm
from time import sleep
import os

def get_sutta_links(master_url, base_url, books, avoid):
    """Scrapes the index page to find all relevant sutta URLs."""
    print(f"Fetching master list from {master_url}...")
    try:
        response = requests.get(master_url)
        response.raise_for_status()
        response.encoding = "UTF-8"
        soup = BeautifulSoup(response.text, "html.parser")
        
        links = soup.find_all("a")
        link_dicts = []

        for link in set(links): # Use set to get unique links initially
            href = link.get('href')
            if href and any(book in href for book in books) and not any(av in href for av in avoid):
                href_split = href.split("/")
                link_dicts.append({
                    "book": href_split[2],
                    "sub_book": href_split[3] if len(href_split) > 4 else "None",
                    "url": f"{base_url}{href}",
                    "sutta_id_text": re.sub(r" +", " ", unicodedata.normalize("NFKD", link.get_text()))
                })

        # Sort the list to ensure a deterministic order and stable IDs
        link_dicts.sort(key=lambda x: x['url'])
        
        print(f"Found and sorted {len(link_dicts)} sutta links to process.")
        return link_dicts
    except requests.RequestException as e:
        print(f"Failed to connect to master URL: {e}")
        return []

def parse_sutta_page(html_content):
    """Parses the HTML of a single sutta page to extract structured data."""
    soup = BeautifulSoup(html_content, "html.parser")
    sutta_div = soup.find("div", id="sutta")
    if not sutta_div:
        return None

    title_attr = sutta_div.find("h1")
    title_text = re.sub(r'\s{2,}', ' ', unicodedata.normalize("NFKD", title_attr.get_text()).strip()) if title_attr else None
    if title_attr: title_attr.decompose()

    # Decompose non-essential sections
    for tag in sutta_div.find_all(["p", "div"], class_=["seealso", "note"]):
        if tag: tag.decompose()

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

def run_scraper(config):
    """Main function to run the full scraping and parsing pipeline."""
    dhammatalks_config = config['dhammatalks']
    sutta_links = get_sutta_links(
        dhammatalks_config['master_url'],
        dhammatalks_config['base_url'],
        dhammatalks_config['books_of_interest'],
        dhammatalks_config['avoid_in_url']
    )

    if not sutta_links:
        print("No sutta links found. Exiting.")
        return

    # Use a relative path from the project root for the output file
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_path = os.path.join(project_root, config['output_paths']['raw_data'])
    
    # Ensure the directory exists before writing
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Scraping suttas and saving to {output_path}...")
    
    # Initialize a counter for the unique ID
    sutta_counter = 1
    
    # Overwrite the file from scratch to ensure UID consistency
    with jsonlines.open(output_path, mode='w') as writer:
        for link_info in tqdm(sutta_links, desc="Scraping Suttas"):
            try:
                response = requests.get(link_info['url'])
                response.raise_for_status()
                response.encoding = "UTF-8"
                
                parsed_data = parse_sutta_page(response.text)
                
                if parsed_data:
                    # Merge initial info with parsed data
                    final_record = {**link_info, **parsed_data}
                    
                    # Add ID
                    final_record['sutta_id'] = sutta_counter
                    
                    writer.write(final_record)
                    
                    # Increment the counter only after a successful write
                    sutta_counter += 1
                
                sleep(0.1)

            except requests.RequestException as e:
                print(f"Could not fetch {link_info['url']}: {e}")
    print(f"A total of {sutta_counter-1} suttas were scraped and saved.")