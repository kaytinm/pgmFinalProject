# Crochet Pattern Recommender

## Introduction

This project implements a web application for recommending crochet patterns based on user preferences and input data. It utilizes machine learning models trained on data from popular crochet websites, such as easycrochet.com and crochet.com, to provide personalized recommendations to users.

## Functionality

### 1. Scraping Data

- Two Python scripts, `scrape_crochet_patterns.py` and `scrape_crochet_patterns_crochet.com.py`, are provided to scrape crochet patterns from easycrochet.com and crochet.com, respectively.
- The scraped data is stored in CSV files named `crochet_patterns.csv` and `crochet_patterns2.csv`.

### 2. Recommendation Models

- Three recommendation models are implemented:
  - `crochet_recommender_easycrochet.py`: Implements recommendation from easycrochet.com data using Bayesian Network.
  - `crochet_recommender_crochet_markov.py`: Implements recommendation from crochet.com data using Markov model.
  - `crochet_recommender_crochet_bayes.py`: Implements recommendation from crochet.com data using Bayesian Network.

### 3. Web Application (Flask)

- The main file, `ap.py`, serves as the entry point to the web application.
- Users can input their preferences and choose a recommendation engine (Bayesian Network, Markovian Network) through a web form.
- The selected preferences are processed, and recommendations are displayed to the user.

## Installation and Running

1. Clone the repository to your local machine.
2. Install the required dependencies listed in the `requirements.txt` file using pip: pip install -r requirements.txt
3. Run the web application using the following command: python ap.py


## Usage

1. Access the web application by navigating to `http://127.0.0.1:5000/` in your web browser.
2.  Select a recommendation engine.
3.  Input your attribute data.
4.  Select Recommend Patterns or Recommend Attribute.
      a.    If you select Recommend Attribute input attribute and select button.
6. Submit the form to receive personalized crochet pattern recommendations.
