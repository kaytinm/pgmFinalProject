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
    model.fit(data, estimator=MaximumLikelihoodEstimator) #Maybe MLE
    save_model(model, 'Bayseian_Model_Crochet_Patterns1.pkl')
    return model

from pgmpy.inference import VariableElimination

def perform_inference(bayesian_model, query_variable, evidence_dict):
    # Use Variable Elimination for inference in the Bayesian network.
    inference = VariableElimination(bayesian_model)
    # Query the model with the specified evidence to get the probability distribution of the query variable.
    result = inference.query(variables=[query_variable], evidence=evidence_dict)
    return result

# 6. Query Functions
def query_hook_size_given_yarn_weight(bayesian_model, yarn_weight):
    # Query the Bayesian model to find the most suitable hook size for a given yarn weight.
    result = perform_inference(bayesian_model, 'Hook Size', {'Yarn Weight': yarn_weight})
    return result

def query_yarn_weight_given_hook_size(bayesian_model, hook_size):
    # Query the Bayesian model to find the most suitable yarn weight for a given hook size.
    result = perform_inference(bayesian_model, 'Yarn Weight', {'Hook Size': hook_size})
    return result

def query_skill_level_given_project_details(bayesian_model, yarn_weight, hook_size, stitches_type):
    # Query the Bayesian model to estimate the skill level required for a project with specific details.
    evidence_dict = {'Yarn Weight': yarn_weight, 'Hook Size': hook_size, 'Stitches': stitches_type}
    result = perform_inference(bayesian_model, 'Skill Level', evidence_dict)
    return result

def query_recommended_stitches_given_yarn_weight_and_skill_level(bayesian_model, yarn_weight, skill_level):
    # Query the Bayesian model to recommend stitches suitable for the given yarn weight and skill level.
    evidence_dict = {'Yarn Weight': yarn_weight, 'Skill Level': skill_level}
    result = perform_inference(bayesian_model, 'Stitches', evidence_dict)
    return result

def query_project_category_given_stitches_and_skill_level(bayesian_model, stitches_type, skill_level):
    # Query the Bayesian model to suggest the category of projects suitable for the given stitches and skill level.
    evidence_dict = {'Stitches': stitches_type, 'Skill Level': skill_level}
    result = perform_inference(bayesian_model, 'Category', evidence_dict)
    return result

def query_yarn_name_given_yarn_weight(bayesian_model, yarn_weight):
    # Query the Bayesian model to suggest yarn names that are typically associated with a given yarn weight.
    result = perform_inference(bayesian_model, 'Yarn Name', {'Yarn Weight': yarn_weight})
    return result


from pgmpy.inference import VariableElimination, BeliefPropagation


def make_decision(model, user_preferences):
    # Use inference to recommend stitches based on user preferences
    inference = VariableElimination(model)
    query_result = inference.query(variables=['Stitches'], evidence=user_preferences)

    # Select the recommendation with the highest probability
    recommended_stitches = query_result['Stitches'].values.argmax()

    # Optionally, use belief propagation for more complex inference tasks
    # belief_propagation = BeliefPropagation(model)
    # result = belief_propagation.query(variables=['Stitches'], evidence=user_preferences)

    return recommended_stitches


def filter_patterns_based_on_recommendations(dataset, recommendations):
    matching_patterns = []

    for pattern in dataset:
        match_found = True

        for recommendation_key, recommendation_value in recommendations.items():
            if pattern[recommendation_key] != recommendation_value:
                match_found = False
                break

        if match_found:
            matching_patterns.append(pattern)

    return matching_patterns


def manual_inference(data, target_variable, evidence):
    # Filter data based on evidence
    filtered_data = data
    for key, value in evidence.items():
        filtered_data = filtered_data[filtered_data[key] == value]

    # Calculate frequency counts
    frequency_counts = filtered_data[target_variable].value_counts()
    total_count = len(filtered_data)

    # Convert frequency to probability
    probability_distribution = (frequency_counts / total_count).to_dict()

    return probability_distribution


def perform_bayesian_inference(bayesian_model, target_variable, evidence):
    inference = VariableElimination(bayesian_model)
    result = inference.query(variables=[target_variable], evidence=evidence)
    return result.values


def compare_inference_methods(data, test_data, bayesian_model, target_variable, evidence_variables):
    manual_correct_predictions = 0
    bayesian_correct_predictions = 0

    for index, instance in test_data.iterrows():
        evidence = {var: instance[var] for var in evidence_variables}
        actual_outcome = instance[target_variable]

        manual_probs = manual_inference(data, target_variable, evidence)
        manual_prediction = max(manual_probs, key=manual_probs.get)

        bayesian_probs = perform_bayesian_inference(bayesian_model, target_variable, evidence)
        bayesian_prediction = bayesian_probs.argmax()

        if manual_prediction == actual_outcome:
            manual_correct_predictions += 1
        if bayesian_prediction == actual_outcome:
            bayesian_correct_predictions += 1

    manual_accuracy = manual_correct_predictions / len(test_data)
    bayesian_accuracy = bayesian_correct_predictions / len(test_data)

    print("Manual Inference Accuracy:", manual_accuracy)
    print("Bayesian Network Inference Accuracy:", bayesian_accuracy)

    return manual_accuracy, bayesian_accuracy

def get_user_preferences():
    # Collect user preferences for yarn weight, hook size, skill level, etc.
    yarn_weight = input("Enter yarn weight preference: ")
    hook_size = input("Enter hook size preference: ")
    skill_level = input("Enter skill level preference: ")
    # Collect other preferences...

    return {
        'Yarn Weight': yarn_weight,
        'Hook Size': hook_size,
        'Skill Level': skill_level
        # Add other preferences...
    }

def suggest_patterns_based_on_preferences(bayesian_model, user_preferences):
    # Perform inference in the Bayesian Network to suggest patterns based on user preferences
    suggested_patterns = perform_inference(bayesian_model, 'Pattern Title', user_preferences)
    return suggested_patterns


def main():
    # Load and preprocess the data
    filename = "crochet_patterns.csv"
    data = pattern_csv_to_df(filename)

    # Define Bayesian network structure
    model_structure = define_bayesian_network_structure()
    data, model_structure = preprocess_stitches_for_bayesian_network(data)

    # Build and learn the Bayesian model
    bayesian_model = build_and_learn_bayesian_model(data, model_structure)

    # Perform various inferences
    # Example: Query hook size for a given yarn weight
    yarn_weight_example = 2  # Assume 2 corresponds to 'medium'
    hook_size_results = query_hook_size_given_yarn_weight(bayesian_model, yarn_weight_example)
    print("Probabilities for Hook Size given Yarn Weight 2 (medium):", hook_size_results)

    # Perform decision-making based on user preferences
    #user_preferences = {'Yarn Weight': 2, 'Skill Level': 1}  # Example preferences
    #recommended_stitches = make_decision(bayesian_model, user_preferences)
    #print("Recommended Stitches:", recommended_stitches)

    # Compare inference methods
    # Assume you have a separate test dataset or split from the original data
    #test_data = data.sample(frac=0.1)  # Using 10% of data as test data
    #evidence_variables = ['Yarn Weight', 'Skill Level']
    #target_variable = 'Hook Size'
    #compare_inference_methods(data, test_data, bayesian_model, target_variable, evidence_variables)
    while True:
        # Get user preferences
        user_preferences = get_user_preferences()

        # Suggest patterns based on user preferences
        suggested_patterns = suggest_patterns_based_on_preferences(bayesian_model, user_preferences)

        # Display suggested patterns to the user
        print("Suggested Patterns:")
        for pattern in suggested_patterns:
            print(pattern)

        # Ask if the user wants to continue or exit
        choice = input("Do you want to continue? (yes/no): ")
        if choice.lower() != 'yes':
            break
if __name__ == "__main__":
    main()