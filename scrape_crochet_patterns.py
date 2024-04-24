
# PGM Final Project
# Recommender System For Crochet Patterns
# Requirements: Python 3.9

import csv
import re
import requests



def parse_details(details_string):
    """
    Parse the concatenated details string into a dictionary of individual details.

    :param details_string: The concatenated string of pattern details.
    :return: A dictionary with parsed details.
    """
    parsed_details = {}
    # Define patterns for keys that appear in the string
    patterns = [
        'Skill Level', 'Yarn Brand', 'Yarn Name',
        'Yarn Weight', 'Hook Size', 'Stitches', 'Color', 'Category', 'Pattern Details'
    ]

    # Add a lookahead to each pattern to split the string, without consuming the delimiter
    split_pattern = '|'.join(f'(?={p})' for p in patterns)

    # Split the details string by the patterns, keeping the delimiter as part of the split segments
    segments = re.split(split_pattern, details_string)

    # Process each segment
    for segment in segments:
        if segment:
            # Split each segment into key and value by the first colon found
            parts = segment.split(':', 1)
            if len(parts) == 2:
                key, value = parts
                parsed_details[key.strip()] = value.strip()
            else:
                # Handle case where there's no colon or the field is empty
                # Assuming the key exists but has no value, assign a default value or None
                parsed_details[parts[0].strip()] = None  # Or use an empty string '' as default

    return parsed_details


def extract_pattern_details(pattern_url):
    """
    Extracts and parses details from a single pattern page.
    """
    details = {}
    response = requests.get(pattern_url, verify=False)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # Assuming the details are within a specific section - you'll need to adjust the selector
        # This example assumes all details are in a text block within a 'div' with a specific class or id
        details_section = soup.select_one('#block-2 > div > dl')
        category_section = soup.select_one('#block-2 > div > dl > dd:nth-child(2) > ul > li:nth-child(7) > a')

        if details_section:
            # Convert the entire details section into text
            details_text = details_section.get_text(separator=' ', strip=True)
            if category_section:
                category_section = "Category: " + category_section.get_text(separator=' ', strip=True)
                details_text, cat = details_text.split("Category")
                details_text += category_section
            # Now parse this text to extract individual details
            details = parse_details(details_text)
            details["Pattern Link"] = pattern_url
    else:
        print(f"Failed to retrieve pattern page. Status code: {response.status_code}")
    return details

def scrape_crochet_patterns(url):
    patterns = {}
    response = requests.get(url, verify=False)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('article')#, class_='kt-blocks-post-grid-item')
        for article in articles:
            # Look for an h2 or h3 within each article for the pattern title
            title_tag = article.find(['h2', 'h3'])
            if title_tag and title_tag.find('a', href=True):
                a = title_tag.find('a', href=True)
                print(f"Title: {a.text.strip()}, URL: {a['href']}")
                pattern_name = a.text.strip()
                pattern_url = a['href']
                details = extract_pattern_details(pattern_url)
                patterns[pattern_name] = details
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
    return patterns




def scrape_and_store_patterns_easycrochet():
    url = 'https://easycrochet.com/all-free-crochet-patterns/'
    patterns = scrape_crochet_patterns(url)

    url_clothing = 'https://easycrochet.com/category/crochet-patterns/crochet-clothing/'
    patterns_clothing = scrape_crochet_patterns(url_clothing)
    patterns.update(patterns_clothing)
    url_blankets = 'https://easycrochet.com/category/crochet-patterns/crochet-blankets/'
    patterns_blankets = scrape_crochet_patterns(url_blankets)
    patterns.update(patterns_blankets)
    url_hats = 'https://easycrochet.com/category/crochet-patterns/headwear/hats/'
    patterns_hats = scrape_crochet_patterns(url_clothing)
    patterns.update(patterns_hats)
    url_amigurumi = 'https://easycrochet.com/category/crochet-patterns/amigurumi-crochet-patterns/'
    patterns_amigurumi = scrape_crochet_patterns(url_amigurumi)
    patterns.update(patterns_amigurumi)
    url_holiday = 'https://easycrochet.com/category/crochet-patterns/crochet-holiday/'
    patterns_holiday = scrape_crochet_patterns(url_holiday)
    patterns.update(patterns_holiday)

    # Define your CSV file name
    csv_file = "venv/crochet_patterns.csv"

    # Open the file in write mode
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(
            ['Title', 'Skill Level', 'Yarn Brand', 'Yarn Name',
             'Yarn Weight', 'Hook Size', 'Stitches', 'Color', 'Category', 'Pattern Link'])

        # Write pattern details
        for pattern_name, details in patterns.items():
            writer.writerow([pattern_name] + [details.get(key, '') for key in
                                              ['Skill Level', 'Yarn Brand', 'Yarn Name', 'Yarn Weight', 'Hook Size',
                                               'Stitches', 'Color', 'Category', 'Pattern Link']])

if __name__ == '__main__':
    scrape_and_store_patterns_easycrochet()