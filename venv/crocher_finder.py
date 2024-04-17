import csv
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.estimators import ExpectationMaximization as EM
from pgmpy.inference import VariableElimination
from pgmpy.inference import BeliefPropagation
from pgmpy.estimators import MaximumLikelihoodEstimator, ExpectationMaximization as EM
import pandas as pd


def pattern_csv_to_df(filename):
    # Read CSV into DataFrame
    df = pd.read_csv(filename)

    # Clean whitespace from all string values in the DataFrame
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Handle numerical conversion for 'Yarn Weight' and 'Skill Level'
    # Assuming mappings are provided
    yarn_weight_mapping = {'light': 1, 'medium': 2, 'heavy': 3}
    skill_level_mapping = {'beginner': 1, 'intermediate': 2, 'expert': 3}

    df['Yarn Weight'] = df['Yarn Weight'].map(yarn_weight_mapping)
    df['Skill Level'] = df['Skill Level'].map(skill_level_mapping)

    # Handle missing values
    df.fillna(df.mean(), inplace=True)

    return df


from pgmpy.models import BayesianModel


def define_bayesian_network_structure():
    # Basic structure based on domain knowledge
    model_structure = [
        ('Yarn Weight', 'Hook Size'),
        ('Yarn Weight', 'Skill Level'),
        ('Hook Size', 'Skill Level'),
        ('Yarn Name', 'Yarn Weight'),
        ('Category', 'Skill Level'),
    ]
    return model_structure


def preprocess_stitches_for_bayesian_network(data):
    # Identify unique stitches and add binary columns for each stitch
    unique_stitches = set(stitch for sublist in data['Stitches'].dropna().str.split(',') for stitch in sublist)
    for stitch in unique_stitches:
        data[stitch] = data['Stitches'].str.contains(stitch).astype(int)

    # Update the model structure with stitch relationships
    model_structure = define_bayesian_network_structure()
    model_structure += [(stitch, 'Category') for stitch in unique_stitches]
    model_structure += [(stitch, 'Skill Level') for stitch in unique_stitches]

    return data, model_structure


def build_and_learn_bayesian_model(data, model_structure):
    # Initialize Bayesian Model
    model = BayesianModel(model_structure)

    # Fit the model using an appropriate estimator
    model.fit(data, estimator=EM)

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
    user_preferences = {'Yarn Weight': 2, 'Skill Level': 1}  # Example preferences
    recommended_stitches = make_decision(bayesian_model, user_preferences)
    print("Recommended Stitches:", recommended_stitches)

    # Compare inference methods
    # Assume you have a separate test dataset or split from the original data
    test_data = data.sample(frac=0.1)  # Using 10% of data as test data
    evidence_variables = ['Yarn Weight', 'Skill Level']
    target_variable = 'Hook Size'
    compare_inference_methods(data, test_data, bayesian_model, target_variable, evidence_variables)

if __name__ == "__main__":
    main()