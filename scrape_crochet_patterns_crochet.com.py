
# PGM Final Project
# Recommender System For Crochet Patterns
# Requirements: Python 3.9

import csv
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup


def parse_details(detail_map):
    """
    Parse the concatenated details string into a dictionary of individual details.

    :param details_string: The concatenated string of pattern details.
    :return: A dictionary with parsed details.
    """
    # Define patterns for keys that appear in the string
    patterns = [
        'Difficulty', 'Yarn Weight', 'Fiber Type',
        'Yardage', 'Pattern Type', 'Hook Sizes'
    ]
    parsed_details = {}
    for key, val in detail_map.items():
        if key in patterns:
            parsed_details[key] = val
        else:
            # Handle case where there's no colon or the field is empty
            # Assuming the key exists but has no value, assign a default value or None
            parsed_details[key] = None  # Or use an empty string '' as default

    return parsed_details


def extract_pattern_details(pattern_url):
    """
    Extracts and parses details from a single pattern page.
    """
    details = {}
    try:
        response = requests.get(pattern_url, verify=False, timeout=100)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Assuming the details are within a specific section - you'll need to adjust the selector
            # This example assumes all details are in a text block within a 'div' with a specific class or id
            details_container = soup.select(
                '#app > div.main-content > div.row.product-display.display-group-display > div.col-md-8.product-display-details > div > div.col-md-5 > div.details-display.col-md-12')
            # Check if the container is found
            if details_container:
                # Find all <b> tags within the container
                if details_container:
                    # Extracting text that includes <b> tags and their following text
                    detail_elements = details_container[0].find_all(lambda tag: tag.name == 'b' and tag.next_sibling)
                    if (len(detail_elements) == 0):
                        return {}
                    details_map = {}
                    for element in detail_elements:
                        label = element.text.strip()
                        value = element.next_sibling.strip() if element.next_sibling else 'No value available'
                        details_map[label.strip(':')] = value
                    print(details_map)
                    details = parse_details(details_map)
                    details['Pattern Link'] = pattern_url
                else:
                    print("No details container found")
            else:
                return {}
        else:
            print(f"Failed to retrieve pattern page. Status code: {response.status_code}")
    except:
        print("Failed to get responce")
        return {}
    return details

def scrape_crochet_patterns(url, url_base):
    patterns = {}
    response = requests.get(url, verify=False)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        list_container = soup.select('#category-items > div.category-items-list')
        if list_container:
            category_items = list_container[0].find_all('div', class_='category-item-section category-item-name')

            # Extract href attributes (links) and names from each item
            for item in category_items:
                link = url_base + item.find('a').get('href') if item.find('a') else None
                name = item.find('a').text.strip() if item.find('a') else 'No name available'
                print(f'Name: {name}, Link: {link}')
                details = extract_pattern_details(link)
                if(len(details) != 0):
                    patterns[name] = details
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
    return patterns



def scrape_and_store_patterns_easycrochet():
   url = 'https://www.crochet.com/patterns/view-all/c/500298?items=ALL'
   patterns = scrape_crochet_patterns(url, "https://www.crochet.com")

   # Define your CSV file name
   csv_file = "crochet_patterns2.csv"
   print(patterns)
   # Open the file in write mode
   with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
       writer = csv.writer(file)
       # Write the header
       writer.writerow(
           [
               'Title', 'Skill Level', 'Yarn Weight', 'Fiber Type',
        'Yardage', 'Pattern Type', 'Hook Size', 'Pattern Link'])

       # Write pattern details
       for pattern_name, details in patterns.items():
           writer.writerow([pattern_name] + [details.get(key, '') for key in
                                             ['Difficulty', 'Yarn Weight', 'Fiber Type',
                                            'Yardage', 'Pattern Type', 'Hook Sizes', 'Pattern Link']])

if __name__ == '__main__':
    csv_file = "crochet_patterns2.csv"
    df = pd.read_csv(csv_file)
    print(df.head(3))
    #scrape_and_store_patterns_easycrochet()
