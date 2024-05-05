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
    #model.fit(data, estimator=MaximumLikelihoodEstimator) #Maybe try EM
    #save_model(model, 'Bayseian_Model_Crochet_Patterns1.pkl')
    model = load_model('../Bayseian_Model_Crochet_Patterns1.pkl')
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
import pandas as pd
from pgmpy.inference import VariableElimination

def get_user_input(all_stitches):
    print("Please enter your preferences (leave blank if no preference):")
    yarn_weight = input("Yarn Weight (e.g., 1, 2, 3...): ").strip() or None
    hook_size = input("Hook Size (in mm, e.g., 2.5, 3.0...): ").strip() or None

    # Collect stitches as multiple inputs
    stitch_inputs = input(f"Stitches (select from {', '.join(all_stitches)}): ").strip().split(',')
    category = input("Category (e.g. Clothing, Blanket, Headwear, Scarf, ...): ").strip() or None
    skill_level = input("Yarn Weight (1, 2, 3): ").strip() or None
    yarn_name = input("Yarn Name (e.g. Blanket Yarn): ").strip() or None

    stitches = {stitch.strip(): 1 for stitch in stitch_inputs if stitch.strip() in all_stitches}
    return yarn_weight, hook_size, stitches, category, skill_level, yarn_name

def infer_preferences_with_probabilities(model, **user_inputs):
    inference = VariableElimination(model)
    inferred_results = {}
    evidence = {key: value for key, value in user_inputs.items() if value is not None}

    # Define all possible query attributes in the model
    all_attributes = ['Yarn Weight', 'Hook Size', 'Skill Level', 'Stitches', 'Yarn Name', 'Category']

    for attribute in all_attributes:
        if attribute not in evidence:
            if evidence:
                query_result = inference.query(variables=[attribute], evidence=evidence, show_progress=False)
                # Store the full probability distribution
                inferred_results[attribute] = query_result
    return inferred_results

def find_matching_patterns(df, preferences, inferred_results):
    # Calculate scores for each pattern
    df = calculate_pattern_scores(df, inferred_results)

    # Filter the DataFrame based on the provided user preferences
    query_conditions = [f"`{key}` == {val!r}" if isinstance(val, str) else f"`{key}` == {val}" for key, val in preferences.items() if val is not None and key in df.columns]
    if query_conditions:
        filtered_df = df.query(" and ".join(query_conditions))
    else:
        filtered_df = df

    # Sort the patterns by scores in descending order
    sorted_patterns = filtered_df.sort_values(by='score', ascending=False)
    return sorted_patterns


def calculate_pattern_scores(df, inferred_results):
    # Initialize a score column in the DataFrame
    df['score'] = 0

    for attribute, distribution in inferred_results.items():
        # Extract the most probable value for simplicity in this example
        most_likely_value = distribution.state_names[attribute][np.argmax(distribution.values)]
        max_prob = max(distribution.values)

        # Increase scores based on the match probability
        df['score'] += df[attribute].apply(lambda x: max_prob if x == most_likely_value else 0)

    return df

import pandas as pd
import numpy as np
import pymc3 as pm
import arviz as az

def hierarchal_baysean_model(df):
    with pm.Model() as model:
        # Priors for overall means
        yarn_name_idx = pm.Normal('yarn_name_idx', mu=0, sigma=1, shape=len(df['Yarn Name'].unique()))
        yarn_weight_idx = pm.Normal('yarn_weight_idx', mu=0, sigma=1, shape=len(df['Yarn Weight'].unique()))
        hook_size_effect = pm.Normal('hook_size_effect', mu=0, sigma=1)
        category_idx = pm.Normal('category_idx', mu=0, sigma=1, shape=len(df['Category'].unique()))
        skill_level_effect = pm.Normal('skill_level_effect', mu=0, sigma=1)

        # Expected number of stitches
        stitches_mu = pm.Deterministic(
            'stitches_mu',
            yarn_name_idx[df['Yarn Name']] +
            yarn_weight_idx[df['Yarn Weight']] +
            hook_size_effect * df['Hook Size'] +
            category_idx[df['Category']] +
            skill_level_effect * df['Skill Level']
        )

        # Likelihood
        stitches = pm.Poisson('stitches', mu=np.exp(stitches_mu), observed=df['Stitches'])

        # Sampling
        trace = pm.sample(500, return_inferencedata=False)

    # Model diagnostics
    az.plot_trace(trace)
    plt.show()
    # Posterior summaries
    summary = az.summary(trace)
    print(summary)


def infer_preferences(model, yarn_weight=None, hook_size=None, stitches={}, category=None, skill_level=None, yarn_name=None):
    inference = VariableElimination(model)
    evidence = {}
    if yarn_weight:
        evidence['Yarn Weight'] = int(yarn_weight)
    if hook_size:
        evidence['Hook Size'] = float(hook_size)
    if skill_level:
        evidence['Skill Level'] = int(skill_level)
    evidence.update(stitches)
    if category:
        evidence.update(category)
    if yarn_name:
        evidence.update(yarn_name)
    # Define all possible query attributes
    all_attributes = ['Yarn Weight', 'Hook Size', 'Skill Level', 'Yarn Name', 'Category'] + list(stitches.keys())
    queries = {}

    # Determine which attributes need to be inferred
    for attribute in all_attributes:
        if attribute not in evidence:
            if evidence:  # Only perform inference if there's at least some evidence
                query_result = inference.query(variables=[attribute], evidence=evidence, show_progress=False)
                most_likely_state = query_result.state_names[attribute][np.argmax(query_result.values)]
                queries[attribute] = most_likely_state

    return queries


def find_matching_patterns(df, preferences):
    # Filter the DataFrame based on the provided preferences
    query = " and ".join(f"`{key}` == {val!r}" if isinstance(val, str) else f"`{key}` == {val}" for key, val in preferences.items() if val is not None)

    return df.query(query) if query else df

def apply_belief_propagation(model):
    # Initialize Belief Propagation
    belief_propagation = BeliefPropagation(model)

    return belief_propagation

def perform_inference_with_bp(bp_engine, query_variables, evidence=None):
    if evidence:
        result = bp_engine.query(variables=query_variables, evidence=evidence)
    else:
        result = bp_engine.query(variables=query_variables)

    return result

def main():
    # Load and preprocess the data
    filename = "crochet_patterns.csv"
    data = pattern_csv_to_df(filename)

    # Define Bayesian network structure
    model_structure = define_bayesian_network_structure()
    data, model_structure = preprocess_stitches_for_bayesian_network(data)

    # Build and learn the Bayesian model
    bayesian_model = build_and_learn_bayesian_model(data, model_structure)
    bp_engine = apply_belief_propagation(bayesian_model)

    #hierarchal_baysean_model(data)
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

    all_stitches = [col for col in data.columns if "Stitch" in col]  # Adjust based on actual column names

    yarn_weight, hook_size, stitches, category, skill_level, yarn_name = get_user_input(all_stitches)
    inferred_preferences = infer_preferences(bayesian_model, yarn_weight, hook_size, stitches, category, skill_level, yarn_name)
    #inferred_results = infer_preferences_with_probabilities(bayesian_model,yarn_weight, hook_size, stitches, category, skill_level, yarn_name)
    # Combine user input and inferred preferences
    combined_preferences = {**inferred_preferences, **{'Yarn Weight': yarn_weight, 'Hook Size': hook_size, 'Category':category, 'Skill Level':skill_level, 'Yarn Name':yarn_name}, **stitches}
    combined_preferences.update(inferred_preferences)

    clean_preferences = {k: v for k, v in combined_preferences.items() if v is not None}
    inferred_preferences = perform_inference_with_bp(bp_engine, clean_preferences.keys(), clean_preferences.values())
    matching_patterns = find_matching_patterns(data, inferred_preferences)

    # Find matching patterns and sort by scores
    #matching_patterns = find_matching_patterns(data, combined_preferences, inferred_results)
    print("Top Matching Patterns:")
    print(matching_patterns.head())

    # Find matching patterns
    matching_patterns = find_matching_patterns(data, clean_preferences)
    #print("Matching Patterns Found:")
    #print(matching_patterns)
if __name__ == "__main__":
    main()