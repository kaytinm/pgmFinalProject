from pgmpy.estimators import ExpectationMaximization as EM
from pgmpy.estimators import MaximumLikelihoodEstimator
import numpy as np
from pgmpy.inference import VariableElimination

from sklearn.model_selection import train_test_split
import re
import pandas as pd
from pgmpy.models import MarkovNetwork
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.factors.discrete import DiscreteFactor



# Data collection and preperation section
def update_category(row):
    title_lower = row['Title'].lower()
    if any(word in title_lower for word in ['blanket', 'throw', 'blankie']):
        return 'Blanket'
    elif any(word in title_lower for word in ['coaster', 'rug', 'wall', 'cozy',  'mat', 'cocoon', 'mit', 'clutch', 'duster', 'rauna',
                                              'cushion', 'cuff', 'pin', 'wallet', 'scrubbie', 'basket'
                                              'ornament', 'holder', 'dish', 'clutch', 'runner', 'keychain', 'slipper', 'coaster', 'bunting'
                                              'earring', 'pillow', 'bracelet', 'towel', 'bag', 'tote', 'pouch','gift', 'planter', 'floor', 'collar']):
        return 'Accessory'
    elif any(word in title_lower for word in
             ['shawl', 'vest', 'cardigan', 'cardi', 'crop', 'sweater', 'hoodie', 'wrap', 'shrug', 'poncho', 'top', 'cami',
              'skirt', 'dress', 'pullover', 'coat', 'swoncho', 'smock', 'sleeve', 'jumper', 'sock', 'hoodie', 'stocking', 'kimono'
              'afgan', 'turtleneck', 'tee', 'jacket', 'shorts', 'cover up', 'tunic', 'slouch',  'tunic', 'capelet']):
        return 'Clothing'
    elif any(word in title_lower for word in ['beanie', 'hat', 'ear', 'hood', 'cap', 'hair', 'toque', 'kerchief', 'balaclava', 'bandana', 'beret']):
        return 'Headwear'
    elif any(word in title_lower for word in ['scarf', 'cowl', 'neck']):
        return 'Scarf'
    elif any(word in title_lower for word in
             ['amigurumi', 'penguin', 'octopus', 'jellyfish', 'owl', 'dog', 'otter', 'pal', 'bear', 'cat', 'unicorn', 'moose', 'whale'
              'lion', 'bear', 'monkey', 'luffy', 'bee', 'puppy', 'lovey', 'sloth', 'animal', 'bumble', 'creature', 'gingerbread'
              'panda', 'gnome', 'santa', 'bunny', 'frankenstein', 'pumpkin', 'triceratops', 'bird', 'furries', 'flamingo']):
        return 'Amigurumi'
    else:
        return 'Other/Unknown'


def get_yardage_range(data):
    copy_data = data.copy()
    def average_yardage(yardage_str):
        """Calculate the difference between the max and min of a yardage range."""
        if '-' in yardage_str:
            min_yard, max_yard = map(int, yardage_str.split('-'))
            return round(((max_yard + min_yard)/2), -2)
        return 0  # Return 0 for single values since there's no range

    # Apply the function to the 'Yardage' column
    # Calculate average yardage for each entry
    copy_data['Average Yardage'] = copy_data['Yardage'].apply(average_yardage)

    # Define bins for categorization, up to 29,000 yards in increments of 1,000
    bins = list(range(0, 30000, 1000))
    bin_labels = [f"{bins[i]}-{bins[i + 1] - 1}" for i in range(len(bins) - 1)]

    # Categorize the average yardage into bins
    copy_data['Yardage Range'] = pd.cut(copy_data['Average Yardage'], bins=bins, labels=bin_labels, right=False)
    data['Yardage Range'] = copy_data['Yardage Range'].astype(str)

    return data


def average_weight(yarn_weight_string):
    yarn_weights = {
        'Lace': 1,
        'Fingering': 2,
        'Sport': 3,
        'DK': 4,
        'Worsted': 5,
        'Aran/Heavy Worsted': 6,
        'Bulky': 7,
        'Super Bulky': 8
    }
    # Split the string by commas and strip any surrounding whitespace
    weights = yarn_weight_string.split(',')
    weights = [w.strip() for w in weights]

    # Convert yarn weights to numeric values
    numeric_weights = [yarn_weights[weight] for weight in weights if weight in yarn_weights]

    # Return the average of these values
    if numeric_weights:
        return sum(numeric_weights) / len(numeric_weights)
    else:
        return -1  # In case of a typo or unrecognised yarn weight


def extract_number(text):
    # Search for numbers in the string
    match = re.search(r'\d+', text)
    if match:
        return int(match.group())  # Return the number if found
    return np.nan  # Return the original text if no number is found
# Function to find numbers, calculate the mean if multiple

#TODO: Shouldn't actually be extracting mean size change later
def extract_mean_size(text):
    # Find all numbers (integers or decimals) before "mm"
    if text == '9” High x 25” Circumference': # Special Case
        return 9
    numbers = re.findall(
        r'\b\d+\.?\d*(?=\s*mm)|(?<=/)\d+\.?\d*|\b\d+\.?\d*(?=/)|(?<=")\s*\d+\.?\d*|\b\d+\.?\d*(?=\s*")', text)

    # Convert all found numbers to float and calculate mean if multiple values are found
    if numbers:
        numbers = list(map(float, numbers))  # Convert to float for accurate mean calculation
        if len(numbers) > 1:
            return round(sum(numbers) / len(numbers), 2)  # Return the mean of the numbers roud to 2nd decimal place for consistancy
        return numbers[0]  # Return the number directly if only one
    else:
        return -1  # Return the original text if no numbers are found


def pattern_csv_to_df(filename):
    # Read CSV into DataFrame
    df = pd.read_csv(filename)

    # Process and clean the data
    df_unique = df.drop_duplicates(subset=['Title', 'Pattern Link'])
    df.dropna(subset=['Skill Level', 'Yarn Weight', 'Hook Size', 'Fiber Type'], inplace=True)

    # Clean whitespace
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # TODO: Change Yarn Name and Yarn Brand to Fiber Type
    # Find Fiber type from Yarn Name and Yarn Brand, the fiber type has a greater influence on the pattern that will be chosen
    # Yarn Weight and the yarn being used is dirrectly correlated however fiber type has more of an influence on category
    # More natural yarns like whool, cotton, or bamboo tend to be prefered for clothing as it creates a more high quality garment
    # Whereas using more expensive natural yarns don't make as much sense for a stuffed animal or a household accessory
    # Handle Different types of values for processing
    df['Average Hook Size'] = df['Hook Size'].apply(extract_mean_size)
    df['Category'] = df.apply(update_category, axis=1)
    df = get_yardage_range(df)
    # Color doesn't have an impact in this case and doesn't really effect the pattern since a different color yarn can be picked
    df['Average Yarn Weight'] = df['Yarn Weight'].apply(average_weight)
    #unique_vals = df.apply(lambda x: x.unique())
    return df



# Baysian Model Section
#TODO: change to markov model
def define_network_structure():
    # Basic structure based on domain knowledge
    model_structure = [
        # generally yarn weight impacts the hook size as the yarn weight increases the hook you should use tends to increase
        ('Average Hook Size', 'Average Yarn Weight'),
        # Working with different sized yarns can impact the skill level for a project using a small yarn can tend to be difficult
        ('Skill Level', 'Average Yarn Weight'),
        # Hook sizes tend to change with the category you generally use a smaller hook size for a armigrumi since it i
        ('Category','Average Hook Size'),
        # Some fiber types are less likely to have certian yarn weights
        # for example it is harder to find a silk or bamboo yarn in a large weight
        # Hobii.com has no bamboo yarn over weight 4
        ('Fiber Type', 'Average Yarn Weight'),
        # Some fiber types are more popular than others for certian projects
        # For example cotton
        ('Category', 'Fiber Type'),
        # The category
        ('Category', 'Yardage Range')
    ]
    return model_structure


def plot_network(model_structure):
    G = nx.DiGraph(model_structure)
    pos = nx.spring_layout(G, seed=42)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='skyblue')

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=2, alpha=0.5, edge_color='gray')

    # labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif", font_weight='bold')

    plt.title("Network", fontsize=24)
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

def encode_data(data):
    encoded_data = data.copy()
    mappings = {}
    for column in data.columns:
        # Check if the column is of a categorical type or object
        if data[column].dtype == 'object' or pd.api.types.is_categorical_dtype(data[column]):
            encoded_data[column], mapping = pd.factorize(data[column])
            mappings[column] = {label: index for index, label in enumerate(mapping)}
        else:
            # Copy the data as is if not categorical
            encoded_data[column] = data[column]
    return encoded_data, mappings


def calculate_empirical_probabilities(data, variables):
    # Generate a DataFrame with all combinations of variable states
    var_levels = [data[var].unique() for var in variables]
    all_combinations = pd.MultiIndex.from_product(var_levels, names=variables).to_frame(index=False)
    encoded_data = data.copy()
    map, encoded_data = encode_data(encoded_data)
    # Calculate the joint frequency of the variables
    frequency_table = data.groupby(variables).size().reset_index(name='counts')
    frequency_table = all_combinations.merge(frequency_table, on=variables, how='left').fillna(0)

    # Convert frequency to probability
    frequency_table['probability'] = frequency_table['counts'] / len(data)
    return frequency_table


def create_factors_for_markov_network(data, model_structure):
    model = MarkovNetwork(model_structure)

    for edge in model.edges():
        # Calculate empirical probabilities for each edge
        variables = list(edge)
        prob_table = calculate_empirical_probabilities(data, variables)

        # Create a factor from the probability table
        values = prob_table['probability'].values
        cardinality = [len(data[var].unique()) for var in variables]
        factor = DiscreteFactor(variables, cardinality, values)
        # Add the factor to the model
        model.add_factors(factor)

    return model

def build_and_learn_markov_model(data, model_structure, load=False, doplot=False):
    # Initialize Markov Model
    encoded_data = data.copy()
    encoded_data, mappings = encode_data(data)
    if load:
        with open('Markov_Model_Crochet_Patterns2HotEM.pkl', 'rb') as f:
            model = pickle.load(f)
        return model, mappings
    else:
        model = create_factors_for_markov_network(data, model_structure)

        if doplot:
            plot_network(model_structure)
        with open('Markov_Model_Crochet_PatternsHot2EM.pkl', 'wb') as f:
            pickle.dump(model, f)
        return model, mappings

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
    states = result.state_names[result.variables[0]]
    #states = result.state_names[attribute]
    probabilities = result.values
    df = pd.DataFrame({
        attribute: states,
        'Probability': probabilities
    })
    df.sort_values('Probability', ascending=False, inplace=True)

    # Get the top N results
    top_results = df.head(top_n)
    return top_results

def decode_attributes(encoded_attributes, mappings):
    decoded_attributes = {}
    for key, value in encoded_attributes.items():
        if key in mappings and value[0] in mappings[key]:
            # Decode the value using the mapping
            decoded_attributes[key] = mappings[key][value[0]]
        else:
            # If the mapping or value is not found, return the original value or handle the missing case
            decoded_attributes[key] = "Unknown"  # Or None, or keep the encoded value as is
    return decoded_attributes


def recommend_patterns(data, attributes):
    if not attributes:
        return pd.DataFrame()  # Return empty DataFrame if no attributes are provided
    # Start with the full dataset
    recommended_patterns = data
    # Filter for each attribute
    for attr, values in attributes.items():
        if type(values) != list:
            values = [values]
        if values:  # Ensure that there are specified values to filter by
            recommended_patterns = recommended_patterns[recommended_patterns[attr].isin(values)]
            if recommended_patterns.empty:
                break  # Stop processing if no data meets the criteria
    return recommended_patterns.drop_duplicates()

# Get user inputs for available attributes
def get_user_input_for_attributes(recommendation_attributes, data):
    input_data = {}
    for attr in recommendation_attributes:
        user_val = user_input_for_attribute(attr)
        if user_val is not None:
            # Convert to correct data types
            if attr in ['Average Hook Size', 'Average Yarn Weight']:
                user_val = float(user_val)
            input_data[attr] = user_val
    return input_data, recommendation_attributes


# GLOBAL USE FOR WEBSITE
filename = "crochet_patterns2.csv"
data = pattern_csv_to_df(filename)

recommendation_attributes = [
    'Skill Level', 'Average Yarn Weight',
    'Fiber Type', 'Yardage Range', 'Category', 'Average Hook Size'

]
recommendation_attributes_orig = recommendation_attributes.copy()
recommendation_attributes_out = [
'Title','Skill Level', 'Yarn Weight', 'Average Yarn Weight'
    'Fiber Type', 'Yardage', 'Yardage Range', 'Category', 'Hook Size','Average Hook Size', 'Pattern Link'
]
attributes = recommendation_attributes.copy()

# Build and learn the Bayesian model
model_structure = define_network_structure()
recommendation_data = data[recommendation_attributes]
markov_model, mappings = build_and_learn_markov_model(recommendation_data, model_structure)
inference_engine = VariableElimination(markov_model)

# Define attributes for recommendation
recommendation_attributes_orig = recommendation_attributes
recommendation_attribute = None


def process_input_data(form_data):
    input_data = {}

    # Process standard attributes
    for attr in recommendation_attributes:
        if attr in form_data and form_data[attr]:
            try:
                if attr == ['Average Hook Size', 'Average Yarn Weight', 'Yardage Range']:
                    input_data[attr] = float(form_data[attr])
                else:
                    input_data[attr] = form_data[attr]
            except ValueError:
                continue
    return input_data


def get_top_recommendations(result, top_n=1):
    values = result.values
    flat_indices = np.argsort(values.flatten())[-top_n:][::-1]
    top_probs = [0]
    top_values_indices = np.unravel_index(flat_indices, values.shape)
    no_to_name = result.no_to_name
    # Convert indices to meaningful names using no_to_name
    top_values = {}
    for dim_index, indices in enumerate(top_values_indices):
        variable_name = result.variables[dim_index]
        name_mapping = no_to_name[variable_name]
        names = [name_mapping[index] for index in indices]
        top_values[variable_name] = names

    return top_values, top_probs


def encode_user_input(attributes, user_inputs, mappings):
    """
    Encodes user input using stored mappings to integer codes.
    """
    encoded_input = {}
    for attribute in attributes:
        if attribute in user_inputs:
            if attribute in mappings and user_inputs[attribute] in mappings[attribute].keys():
                encoded_input[attribute] = mappings[attribute][user_inputs[attribute]]
            elif attribute not in mappings:
                encoded_input[attribute] = user_inputs[attribute]
            else :
                # Handle unseen categories or missing mappings
                encoded_input[attribute] = -1
        else:
            encoded_input[attribute] = None
    return encoded_input



# Web app
def recommend_patterns_from_bayes(input_data):
    print(input_data)
    encoded_input = encode_user_input(recommendation_attributes_orig, input_data, mappings)
    input_data = {k: v for k, v in encoded_input.items() if v is not None}
    probable_attributes = {}
    found_match = False
    top_n = 1
    threshold_probabilities = {}  # To store initial top probabilities for reference
    # TODO: Consider OR for searching based on stitches
    non_input_attributes = []
    for attr in recommendation_attributes:
        if attr not in input_data:
            non_input_attributes.append(attr)
    # Query the model
    print("getting result")
    result_map = inference_engine.map_query(variables=non_input_attributes, evidence=input_data)
    print(result_map)
    result = inference_engine.query(variables=non_input_attributes, evidence=input_data)
    print("result got")

    while not found_match and top_n <= 5:
        #result_map = inference_engine.map_query(variables=non_input_attributes, evidence=input_data)
        top_values, top_probs = get_top_recommendations(result, top_n=top_n)
        new_values = top_values.copy()
        # Handle Encoding
        for attr in recommendation_attributes:
            if attr in non_input_attributes:
                if attr in mappings.keys():
                    for i in range(len(top_values[attr])):
                        for key, value in mappings[attr].items():
                            if value == top_values[attr][i]:
                                new_values[attr][i] = key
                                break
                            if top_values[attr][i] == -1:
                                print("unspecified value")
                                break
                    # Initialize threshold for the first time
                    if top_n == 1:
                        threshold_probabilities[attr] = .1  # set a threshold to 10%
            else:
                new_values[attr] = [input_data[attr]]
                if attr in mappings.keys(): # directly use the provided input
                    for key, value in mappings[attr].items():
                        if value == input_data[attr]:
                            new_values[attr] = [key]
                            break

        recommended_patterns = recommend_patterns(data, new_values)
        if not recommended_patterns.empty:
            print("FOUND")
            print(f"Found matches with top_{top_n} recommendations for {list(new_values.keys())}.")
            print(recommended_patterns[recommendation_attributes_out])
            return recommended_patterns[recommendation_attributes_out]
        else:
            top_n += 1
    if not found_match:
        print("No matches found even after expanding search.")
        return pd.DataFrame()

# Web app
def get_recommendation_for_attribute(recommendation_attribute, input_data):
    encoded_input = encode_user_input(recommendation_attributes_orig, input_data, mappings)
    input_data = {k: v for k, v in encoded_input.items() if v is not None}
    while recommendation_attribute not in recommendation_attributes_orig or recommendation_attribute in input_data:
        if recommendation_attribute not in recommendation_attributes_orig:
            print(f"Invalid attribute. Choose from: {', '.join(recommendation_attributes)}")
        if recommendation_attribute in input_data:
            print("You've already specified this attribute. Please choose another one.")
    # Perform inference for attribute recommendation
    if recommendation_attribute:
        if (type(recommendation_attribute)!= list):
            recommendation_attribute = [recommendation_attribute]
        result_map = inference_engine.map_query(variables=recommendation_attribute, evidence=input_data,
                                                    elimination_order="MinWeight")

        if recommendation_attribute[0] in mappings.keys():
            for key, value in mappings[recommendation_attribute[0]].items():
                if result_map[recommendation_attribute[0]] == -1:
                    result_map[recommendation_attribute[0]] = 'Unspecified'
                    return result_map
                if value == result_map[recommendation_attribute[0]]:
                    result_map[recommendation_attribute[0]] = key
                    return result_map
        else:
            return result_map


def evaluate_model(df, recommendation_attributes):
    # Build and learn the Bayesian model
    model_structure = define_network_structure()
    markov_model, mappings = build_and_learn_markov_model(data[recommendation_attributes], model_structure)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    markov_model, mappings = build_and_learn_markov_model(data[recommendation_attributes], model_structure)


    # Generate predictions for a specified column
    def predict(model, data, target):
        inference = VariableElimination(model)
        predictions = []
        for _, row in data.iterrows():
            evidence = row.to_dict()
            evidence.pop(target)  # Remove the target variable from evidence
            predicted = inference.map_query(variables=[target], evidence=evidence)
            predictions.append(predicted[target])
        return predictions

    # Train the model
    markov_model = build_and_learn_markov_model(train_df[recommendation_attributes], model_structure)


    # Predict the 'Category' for test data
    test_df['predicted_category'] = predict(markov_model, test_df, 'Category')

    # Evaluate accuracy
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(test_df['Category'], test_df['predicted_category'])
    print(f'Accuracy of the Bayesian network for Category prediction: {accuracy:.2f}')


def main():
    # Load and preprocess the data
    recommendation_attributes = [
        'Skill Level', 'Average Yarn Weight',
        'Fiber Type', 'Yardage Range', 'Category', 'Average Hook Size'

    ]
    filename = "crochet_patterns2.csv"

    data = pattern_csv_to_df(filename)
    attributes = recommendation_attributes.copy()
    # Define Bayesian network structure
    model_structure = define_network_structure()
    #plot_bayesian_network(model_structure)
    # Build and learn the Bayesian model
    markov_model, mappings = build_and_learn_markov_model(recommendation_data, model_structure)
    inference_engine = VariableElimination(markov_model)
    # Define attributes for recommendation

    recommendation_attribute = None
    input_data, recommendation_attributes = get_user_input_for_attributes(recommendation_attributes, data)

    #encoded_input = encode_user_input(recommendation_attributes, input_data, mappings)
   # cleaned_data = {k: v for k, v in encoded_input.items() if v is not None}
    recommended_patterns = recommend_patterns_from_bayes(input_data)
   # recommended_patterns = recommend_patterns_from_input(recommendation_attributes, cleaned_data, inference_engine,
    #                                                     data)
    print("Recommended Patterns:")
    if not recommended_patterns.empty:
        print(recommended_patterns[['Title', 'Pattern Link']])
        # Make recommendations based on input
    rec = get_recommendation_for_attribute('Average Hook Size', input_data)
    print(rec)
   # reccommend_attribute_based_on_user_input(recommendation_attributes_orig, input_data, data,
    #                                         recommendation_attributes, inference_engine)


if __name__ == "__main__":
    #print("hi")
    main()