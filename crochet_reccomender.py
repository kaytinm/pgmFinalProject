import csv
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import ExpectationMaximization as EM
from pgmpy.inference import VariableElimination
from pgmpy.inference import BeliefPropagation
from pgmpy.estimators import MaximumLikelihoodEstimator, ExpectationMaximization as EM
import pandas as pd
import numpy as np
import pandas as pd
import pickle
from pgmpy.inference import VariableElimination

def update_category(row):
    title_lower = row['Title'].lower()
    if any(word in title_lower for word in ['blanket', 'throw']):
        return 'Blanket'
    elif any(word in title_lower for word in ['beanie', 'hat', 'ear']):
        return 'Headwear'
    elif any(word in title_lower for word in ['scarf', 'cowl', 'neck']):
        return 'Scarf'
    elif any(word in title_lower for word in
             ['shawl', 'vest', 'cardigan', 'sweater', 'hoodie', 'wrap', 'shrug', 'poncho', 'top', 'skirt', 'dress',
              'afgan']):
        return 'Clothing'
    elif 'basket' in title_lower:
        return 'Basket'
    elif any(word in title_lower for word in ['coaster', 'placemat', 'cozy', 'ornament']):
        return 'Accessory'
    elif any(word in title_lower for word in
             ['amigurumi', 'penguin', 'octopus', 'jellyfish', 'owl', 'dog', 'lion', 'bear', 'monkey', 'luffy', 'bee',
              'panda', 'gnome', 'santa', 'frankenstein', 'pumpkin']):
        return 'Amigurumi'
    else:
        return 'Stitch/Granny Square'

def extract_number(text):
    import re  # Regular expression module
    # Search for numbers in the string
    match = re.search(r'\d+', text)
    if match:
        return int(match.group())  # Return the number if found
    return text  # Return the original text if no number is found
# Function to find numbers, calculate the mean if multiple

#TODO: Shouldn't actually be extracting mean size change later
def extract_mean_size(text):
    import re
    # Find all numbers (integers or decimals) before "mm"
    numbers = re.findall(r'\b\d+\.?\d*(?=\s*mm)', text)
    # Convert all found numbers to float and calculate mean if multiple values are found
    if numbers:
        numbers = list(map(float, numbers))  # Convert to float for accurate mean calculation
        if len(numbers) > 1:
            return sum(numbers) / len(numbers)  # Return the mean of the numbers
        return numbers[0]  # Return the number directly if only one
    return text  # Return the original text if no numbers are found

def check_multiple_colors(color):
    if str(color) == 'nan':
        return 'NA'
    elif',' in color:
        return 'Multi'
    return color

def preprocess_stitch_names(stitch_column):
    # Remove Back Loop and front loop to reduce model complexity.
    # The front loop and back loop stitching doesn't impact the difficutlty or much of the structure of the project.
    # It is more of a mild change on the base stitch type
    removals = ['Back Loop', 'Front Loop']
    for removal in removals:
        stitch_column = stitch_column.replace(removal, '', regex=True)
    # Remove stitch type before increase and decrease values to reduce model complexity.
    # The Type of stitch isn't what is notable in this case it is if you are doing an increase or decrease.
    stitch_column = stitch_column.str.split('(Two Together|Increase)').str[-1].str.strip()
    return stitch_column

def pattern_csv_to_df(filename):
    # Read CSV into DataFrame
    df = pd.read_csv(filename)

    # Process and clean the data
    df_unique = df.drop_duplicates(subset=['Title', 'Pattern Link'])
    df.dropna(subset=['Skill Level', 'Yarn Weight', 'Hook Size', 'Stitches'], inplace=True)

    # Clean whitespace
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Handle Different types of values for processing
    df['Yarn Weight'] = df['Yarn Weight'].apply(extract_number)
    df['Skill Level'] = df['Skill Level'].apply(extract_number)
    df['Hook Size'] = df['Hook Size'].apply(extract_mean_size)
    df['Category'] = df.apply(update_category, axis=1)
    df['Color'] = df['Color'].apply(check_multiple_colors)
    df['Stitches'] = preprocess_stitch_names(df['Stitches'])
    return df


from pgmpy.models import BayesianNetwork


def define_bayesian_network_structure():
    # Basic structure based on domain knowledge
    model_structure = [
        ('Yarn Weight', 'Hook Size'),
        ('Yarn Weight', 'Skill Level'),
        ('Hook Size', 'Skill Level'),
        ('Yarn Name', 'Yarn Weight'),
        ('Category', 'Skill Level')
    ]
    return model_structure


def preprocess_stitches_for_bayesian_network(data):
    # Identify unique stitches and add binary columns for each stitch
    unique_stitches = set(stitch.strip() for sublist in data['Stitches'].dropna().str.split(',') for stitch in sublist)
    unique_stitches.discard("")
    for stitch in unique_stitches:
        data[stitch] = data['Stitches'].str.contains(stitch).astype(int)
    # Update the model structure with stitch relationships
    model_structure = define_bayesian_network_structure()
    #model_structure += [(stitch, 'Category') for stitch in unique_stitches]
    model_structure += [(stitch, 'Skill Level') for stitch in unique_stitches]

    return data, model_structure

import networkx as nx
import matplotlib.pyplot as plt
def plot_bayesian_network(model_structure):
    G = nx.DiGraph(model_structure)
    pos = nx.spring_layout(G, seed=42)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='skyblue')

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=2, alpha=0.5, edge_color='gray')

    # labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif", font_weight='bold')

    plt.title("Bayesian Network", fontsize=24)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

import pickle

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def build_and_learn_bayesian_model(data, model_structure):
    # Initialize Bayesian Model
    model = BayesianNetwork(model_structure)
    print(model_structure)
    #plot_bayesian_network(model_structure) #TODO: Clean Up
    # Fit the model using an appropriate estimator
    #model.fit(data, estimator=MaximumLikelihoodEstimator) #Maybe try EM
    #save_model(model, 'Bayseian_Model_Crochet_Patterns2.pkl')
    model = load_model('Bayseian_Model_Crochet_Patterns12.pkl')
    return model

def user_input_for_attribute(attribute, possible_values=None):
    if possible_values:
        print(f"Enter {attribute} (possible values: {', '.join(map(str, possible_values))}) or 'skip' to not specify:")
    else:
        print(f"Enter {attribute} or 'skip' to not specify:")
    user_input = input().strip()
    if user_input.lower() == 'skip':
        return None
    return user_input

# Reccomendation for specific Attribute
def get_top_recommendations_for_attribute(result, attribute, top_n=1):
    # Create a DataFrame from the result
    states = result.state_names[attribute]
    probabilities = result.values
    df = pd.DataFrame({
        attribute: states,
        'Probability': probabilities
    })
    df.sort_values('Probability', ascending=False, inplace=True)

    # Get the top N results
    top_results = df.head(top_n)
    return top_results

def get_top_recommendations(result, attribute, top_n=1):
    # Convert the discrete factor to a DataFrame and sort by probability
    state_names = result.state_names[attribute]
    probabilities = result.values

    # Combine the states and probabilities into a DataFrame
    df = pd.DataFrame({
        attribute: state_names,
        'Probability': probabilities
    })

    # Sort the DataFrame by probabilities in descending order
    df.sort_values(by='Probability', ascending=False, inplace=True)
    # Return the top N results as lists
    return df.head(top_n)[attribute].tolist(), df.head(top_n)['Probability'].tolist()


def recommend_patterns(data, attributes):
    if not attributes:
        return pd.DataFrame()  # Return empty DataFrame if no attributes are provided
    # Start with the full dataset
    recommended_patterns = data
    # Filter for each attribute
    for attr, values in attributes.items():
        if values:  # Ensure that there are specified values to filter by
            recommended_patterns = recommended_patterns[recommended_patterns[attr].isin(values)]
            if recommended_patterns.empty:
                break  # Stop processing if no data meets the criteria
    return recommended_patterns.drop_duplicates()

# Get user inputs for available attributes
def get_user_input_for_attributes(recommendation_attributes, data):
    input_data = {}
    my_stitches_names = data.keys().to_list()[data.keys().to_list().index('Pattern Link') + 1:]
    for attr in recommendation_attributes:
        user_val = user_input_for_attribute(attr)
        if user_val is not None:
            # Convert to correct data types
            if attr in ['Skill Level', 'Yarn Weight']:
                user_val = int(user_val)
            elif attr == 'Hook Size':
                user_val = float(user_val)
            input_data[attr] = user_val
    # Special handling for hot-one-encoded stitches
    stitches = input("Enter Stitches (comma-separated if multiple, skip if unknown): ")
    if stitches.lower() != 'skip':
        stitches_input = [stitch.strip() for stitch in stitches.split(',')]
        for stitch in stitches_input:
            if stitch in my_stitches_names:
                input_data[stitch] = 1
            else:
                print("Sorry the stitch ", stitch, " is not availible.")
        more_stitches = input("Do you want a pattern with ONLY these stitches? (enter yes or no): ")
        if more_stitches.lower() == "no":
            recommendation_attributes.extend(my_stitches_names)
    else:
        recommendation_attributes.extend(my_stitches_names)
    return input_data, recommendation_attributes

def recommend_patterns_from_input(recommendation_attributes, input_data, inference_engine, data):
    probable_attributes = {}
    found_match = False
    top_n = 1
    max_prob = 0
    all_top_probs = {}
    threshold_probabilities = {}  # To store initial top probabilities for reference
    # TODO: Consider OR for searching based on stitches
    while not found_match and top_n <= 5:
        for attr in recommendation_attributes:
            if attr not in input_data:
                # Query the model
                result = inference_engine.query(variables=[attr], evidence=input_data)
                top_values, top_probs = get_top_recommendations(result, attr, top_n=top_n)
                all_top_probs[attr] = dict(zip(top_values, top_probs))
                probable_attributes[attr] = top_values
                # Initialize threshold for the first time
                if top_n == 1:
                    threshold_probabilities[attr] = .1  # set a threshold to 10%
            else:
                probable_attributes[attr] = [input_data[attr]]  # directly use the provided input

        recommended_patterns = recommend_patterns(data, probable_attributes)
        if not recommended_patterns.empty:
            found_match = True
            print(f"Found matches with top_{top_n} recommendations for {list(probable_attributes.keys())}.")
        else:
            top_n += 1
            # Reduce the set of probable attributes based on updated thresholds
            for attr, probs in threshold_probabilities.items():
                if attr not in input_data:  # Ensure not to override user input
                    filtered_values = [value for value, prob in all_top_probs[attr].items() if prob >= probs]
                    probable_attributes[attr] = filtered_values
            recommended_patterns = recommend_patterns(data, probable_attributes)
            if not recommended_patterns.empty:
                found_match = True
    if not found_match:
        print("No matches found even after expanding search.")
        return pd.DataFrame()
    else:
        return recommended_patterns

def reccommend_attribute_based_on_user_input(recommendation_attributes_orig, input_data, data, recommendation_attributes, inference_engine):
    stitches_names = data.keys().to_list()[data.keys().to_list().index('Pattern Link') + 1:]
    # Ask user which attribute they want a recommendation for
    print("Which attribute would you like a recommendation for?")
    recommendation_attribute = input().strip()
    while recommendation_attribute not in recommendation_attributes_orig or recommendation_attribute in input_data or len(
            set(input_data).intersection(set(stitches_names))) != 0:
        if recommendation_attribute not in recommendation_attributes_orig:
            print(f"Invalid attribute. Choose from: {', '.join(recommendation_attributes)}")
        if recommendation_attribute in input_data:
            print("You've already specified this attribute. Please choose another one.")
    if recommendation_attribute == "Stitches":
        recommendation_attribute = stitches_names
    # Perform inference for attribute recommendation
    if recommendation_attribute:
        if type(recommendation_attribute) == list:
            top_results = {}
            for attr in recommendation_attribute:
                result = inference_engine.query(variables=[attr], evidence=input_data)
                top_results[attr] = get_top_recommendations_for_attribute(result, attr)[attr].values[0]
            recommended_stitches = [key for key, value in top_results.items() if value == 1]
            print("Recommended Stitches are: ", recommended_stitches)
        else:
            result = inference_engine.query(variables=[recommendation_attribute], evidence=input_data)
            top_results = get_top_recommendations_for_attribute(result, recommendation_attribute)
            print(f"Recommended {recommendation_attribute}:")
            for index, row in top_results.iterrows():
                print(f"{row[recommendation_attribute]} with a probability of {row['Probability']:.4f}")

# GLOBAL USE FOR WEBSITE


filename = "crochet_patterns.csv"
data = pattern_csv_to_df(filename)
data, model_structure = preprocess_stitches_for_bayesian_network(data)

# Build and learn the Bayesian model
bayesian_model = build_and_learn_bayesian_model(data, model_structure)
inference_engine = VariableElimination(bayesian_model)

    # Define attributes for recommendation
recommendation_attributes = [
        'Skill Level', 'Yarn Name', 'Yarn Weight',
        'Hook Size', 'Category' #, 'Yarn Brand', 'Color', , 'Stitches'
]
recommendation_attributes_orig = [
        'Skill Level', 'Yarn Name', 'Yarn Weight',
        'Hook Size', 'Category', 'Stitches' #, 'Yarn Brand', 'Color', , 'Stitches'
]
recommendation_attribute = None


def process_input_data(form_data):
    input_data = {}
    stitches_names = data.keys().to_list()[data.keys().to_list().index('Pattern Link') + 1:]  # Adjust based on your data's column names for stitches

    # Process standard attributes
    for attr in ['Skill Level', 'Yarn Name', 'Yarn Weight', 'Hook Size', 'Category']:
        if attr in form_data and form_data[attr]:
            try:
                if attr in ['Skill Level', 'Yarn Weight']:
                    input_data[attr] = int(form_data[attr])
                elif attr == 'Hook Size':
                    input_data[attr] = float(form_data[attr])
                else:
                    input_data[attr] = form_data[attr]
            except ValueError:
                continue

    # Handling stitches
    stitches_input = form_data.get('Stitches', '').split(',')
    only_these_stitches = form_data.get('only_these_stitches', 'no') == 'yes'
    if stitches_input:
        unavailable_stitches = []
        for stitch in stitches_input:
            stitch = stitch.strip()
            if stitch in stitches_names:
                input_data[stitch] = 1
            else:
                unavailable_stitches.append(stitch)

        if unavailable_stitches:
            feedback = f"Sorry, the following stitches are not available: {', '.join(unavailable_stitches)}"
            # You might want to show this feedback to the user
        if not only_these_stitches:
            recommendation_attributes.extend(stitches_names)
            # If not only these stitches, extend to include all stitch columns
        #    for stitch in stitches_names:
        #        input_data.setdefault(stitch, 0)  # Set to 0 if not already set to 1

    return input_data

def recommend_patterns_from_bayes(input_data):
    probable_attributes = {}
    found_match = False
    top_n = 1
    max_prob = 0
    all_top_probs = {}
    threshold_probabilities = {}  # To store initial top probabilities for reference
    # TODO: Consider OR for searching based on stitches
    while not found_match and top_n <= 5:
        for attr in recommendation_attributes:
            if attr not in input_data:
                # Query the model
                result = inference_engine.query(variables=[attr], evidence=input_data)
                top_values, top_probs = get_top_recommendations(result, attr, top_n=top_n)
                all_top_probs[attr] = dict(zip(top_values, top_probs))
                probable_attributes[attr] = top_values
            else:
                probable_attributes[attr] = [input_data[attr]]  # directly use the provided input

        recommended_patterns = recommend_patterns(data, probable_attributes)
        if not recommended_patterns.empty:
            found_match = True
            print(f"Found matches with top_{top_n} recommendations for {list(probable_attributes.keys())}.")
            print("Recommded Patterns for Attributes: ", probable_attributes)
        else:
            top_n += 1
            # Reduce the set of probable attributes based on updated thresholds
            for attr in all_top_probs.keys():
                if attr not in input_data:  # Ensure not to override user input
                    filtered_values = [value for value, prob in all_top_probs[attr].items() if prob >= 0.1]
                    probable_attributes[attr] = filtered_values
            recommended_patterns = recommend_patterns(data, probable_attributes)
            if not recommended_patterns.empty:
                print("Recommded Patterns for Attributes: ", probable_attributes)
                found_match = True
    if not found_match:
        print("No matches found even after expanding search.")
        return pd.DataFrame()
    else:
        return recommended_patterns


def main():
    # Load and preprocess the data
    filename = "venv/crochet_patterns.csv"
    data = pattern_csv_to_df(filename)

    # Define Bayesian network structure
    model_structure = define_bayesian_network_structure()
    data, model_structure = preprocess_stitches_for_bayesian_network(data)

    # Build and learn the Bayesian model
    bayesian_model = build_and_learn_bayesian_model(data, model_structure)
    inference_engine = VariableElimination(bayesian_model)

    # Define attributes for recommendation
    recommendation_attributes = [
        'Skill Level', 'Yarn Name', 'Yarn Weight',
        'Hook Size', 'Category' #, 'Yarn Brand', 'Color', , 'Stitches'
    ]
    recommendation_attributes_orig = [
        'Skill Level', 'Yarn Name', 'Yarn Weight',
        'Hook Size', 'Category', 'Stitches' #, 'Yarn Brand', 'Color', , 'Stitches'
    ]
    recommendation_attribute = None
    input_data, recommendation_attributes = get_user_input_for_attributes(recommendation_attributes, data)
    recommended_patterns = recommend_patterns_from_input(recommendation_attributes, input_data, inference_engine, data)
    print("Recommended Patterns:")
    if not recommended_patterns.empty:
        print(recommended_patterns[['Title', 'Pattern Link']])
        # Make recommendations based on input
    reccommend_attribute_based_on_user_input(recommendation_attributes_orig, input_data, data, recommendation_attributes,inference_engine)

#if __name__ == "__main__":
#main()