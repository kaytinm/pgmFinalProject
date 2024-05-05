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
from sklearn_crfsuite import CRF
from sklearn.model_selection import train_test_split
import joblib   # Depending on your sklearn version, joblib might be included in sklearn

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


import pandas as pd


def preprocess_stitches(df):
    keywords = [
        "Decrease",
        "Two Together",
        "Three Together",
        "Front Loop",
        "Back Loop",
        "Increase"
    ]

    # Explode the 'Stitches' column into separate rows for each stitch
    df_exploded = df['Stitches'].str.split(',').explode().str.strip()

    # Function to clean individual stitch names
    def clean_stitch_name(stitch):
        for keyword in keywords:
            if keyword in stitch:
                # Remove keyword and split by spaces to handle cases like 'Single Crochet Two Together'
                parts = stitch.replace(keyword, '')
                # Remove empty strings resulting from split and replace original stitch name
                return keyword+','+ parts
        return stitch

    # Apply cleaning function to each stitch name
    cleaned_stitches = df_exploded.dropna().apply(clean_stitch_name).str.split(',').explode().str.strip()
    df['Cleaned_Stitches'] = cleaned_stitches.groupby(cleaned_stitches.index).apply(lambda x: ', '.join(x.unique()))
    stitch_columns = df['Cleaned_Stitches'].str.get_dummies(sep=', ')

    # Concatenate the original dataframe with the new stitch columns
    df = pd.concat([df, stitch_columns], axis=1)

    return df



def get_unique_stitches(df):
    keywords = [
        "Decrease",
        "Two Together",
        "Three Together",
        "Front Loop",
        "Back Loop",
        "Increase"
    ]

    unique_stitches = data['Stitches'].str.split(',').explode().str.strip().drop_duplicates().sort_values().dropna()
    for keyword in keywords:
        unique_stitches = unique_stitches.replace(keyword, '', regex=True)
    unique_stitches = [stitch.strip() for stitch in unique_stitches]
    unique_stitches = sorted(set(unique_stitches))
    return unique_stitches

def hot_one_encode_stitches(data):
    # Split the 'Stitches' column into individual stitches
    stitch_columns = data['Stitches'].str.get_dummies(sep=',')
    # Merge these new columns back into the original DataFrame
    data = pd.concat([data, stitch_columns], axis=1)
    return data

def extract_number(text):
    import re  # Regular expression module
    # Search for numbers in the string
    match = re.search(r'\d+', text)
    if match:
        return int(match.group())  # Return the number if found
    return np.nan  # Return the original text if no number is found
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
        return np.nan
    elif',' in color:
        return 'Multi'
    return color


import pandas as pd


def determine_fiber_type(row):
    yarn_name = row['Yarn Name']
    yarn_name = str(yarn_name).lower()
    yarn_names_list = []
    if ',' in yarn_name:
        yarn_names_list = yarn_name.split(',')
    fiber_map = {
        'hygge': 'Acrylic',
        'softee baby': 'Acrylic',
        'pound of love': 'Acrylic',
        'super saver': 'Acrylic',
        'blanket extra': 'Polyester',
        'heartland': 'Acrylic',
        'vanna\'s choice': 'Acrylic',
        'brava worsted': 'Acrylic',
        'feels like heaven': 'Nylon',
        'homespun': 'Cotton/Cotton Blend',
        'charisma': 'Acrylic',
        'feels like butta': 'Polyester',
        'mandala': 'Acrylic',
        'blanket yarn': 'Polyester',
        'tina tape yarn': 'Lyocell',
        'palette': 'Wool/Wool Blend',
        'summerlite 4 ply': 'Cotton/Cotton Blend',
        'summer nights': 'Acrylic',
        'stone washed': 'Cotton/Cotton Blend',
        'must-have': 'Cotton/Cotton Blend',
        'simply dk': 'Acrylic',
        're-spekt yarn': 'Cotton/Cotton Blend',
        'cotlin': 'Cotton/Cotton Blend',
        'bonus dk': 'Acrylic',
        'mellowspun dk': 'Acrylic',
        'swish dk': 'Wool/Wool Blend',
        'coboo yarn': 'Cotton/Cotton Blend',
        'kindred': 'Wool/Wool Blend',
        'beehive baby sport': 'Acrylic',
        'flikka': 'Cotton/Cotton Blend',
        'jeans': 'Cotton/Cotton Blend',
        'caron cakes': 'Wool/Wool Blend',
        'baby soft': 'Acrylic',
        'simply aran': 'Acrylic',
        'dishie': 'Cotton/Cotton Blend',
        'lily sugar\'n cream': 'Cotton/Cotton Blend',
        're-spun': 'Polyester',
        'unforgettable': 'Acrylic',
        'simply soft': 'Acrylic',
        'heatherly worsted': 'Acrylic',
        'for the home cording': 'Cotton/Cotton Blend',
        'stitch soak scrub': 'Nylon',
        'red heart soft': 'Acrylic',
        'skein tones': 'Acrylic',
        'basic stitch yarn': 'Acrylic',
        'comfy': 'Cotton/Cotton Blend',
        'vel-luxe yarn': 'Polyester',
        'comfy worsted': 'Cotton',
        'i love this yarn': 'Acrylic',
        'bundle up': 'Polyester',
        'nuboo': 'Lyocell',
        'scrubby': 'Polyester',
        'caron one pound': 'Acrylic',
        're-up': 'Cotton/Cotton Blend',
        'kid classic': 'Cotton/Cotton Blend',
        'premier home': 'Wool',
        'soft baby steps': 'Acrylic',
        'crayola cake': 'Acrylic',
        'terryspun': 'Polyester',
        'royal melange': 'Acrylic',
        'shawl in a ball': 'Cotton',
        'big twist premium': 'Acrylic',
        'impeccable': 'Acrylic',
        'satin': 'Acrylic',
        'gumdrop': 'Polyester',
        'feels like bliss': 'Nylon',
        'simply chunky': 'Acrylic',
        'hue + me': 'Acrylic',
        'maker home dec': 'Cotton/Cotton Blend',
        'bellissima chunky yarn': 'Acrylic',
        'hometown usa': 'Acrylic',
        'rewind': 'Polyester',
        'soft essentials': 'Acrylic',
        'baby blanket': 'Polyester',
        'simply super chunky': 'Polyester',
        'big twist natural blend': 'Wool/Wool Blend',
        'go for faux': 'Polyester',
        'isaac mizrahi craft': 'Acrylic',
        'country loom': 'Acrylic',
        'mega bulky': 'Acrylic'
    }
    if len(yarn_names_list) != 0:
        for name in yarn_names_list:
            if 'wool' in name or 'sheep' in name or 'alpaca' in name:
                return 'Wool/Wool Blend'
            elif 'cotton' in name:
                return 'Cotton/Cotton Blend'
                # Checking specific yarn names with case-insensitive match
            elif name in fiber_map.keys():
                return fiber_map[name]
    # Checking for general terms
    if 'wool' in yarn_name or 'sheep' in yarn_name or 'alpaca' in yarn_name:
        return 'Wool/Wool Blend'
    elif 'cotton' in yarn_name:
        return 'Cotton/Cotton Blend'
    # Checking specific yarn names with case-insensitive match
    elif yarn_name in fiber_map.keys():
        return fiber_map[yarn_name]

    else:
        # about 38% of the fiber types are unknown
        # when 'Fiber Type' is 'Unknown', it does not significantly sway the recommendation one way or another
        return np.nan



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
    # TODO: Change Yarn Name and Yarn Brand to Fiber Type
    # Find Fiber type from Yarn Name and Yarn Brand, the fiber type has a greater influence on the pattern that will be chosen
    # Yarn Weight and the yarn being used is dirrectly correlated however fiber type has more of an influence on category
    # More natural yarns like whool, cotton, or bamboo tend to be prefered for clothing as it creates a more high quality garment
    # Whereas using more expensive natural yarns don't make as much sense for a stuffed animal or a household accessory
    # Handle Different types of values for processing
    #df['Fiber Type'] = df.apply(determine_fiber_type, axis=1)
    df['Yarn Weight'] = df['Yarn Weight'].apply(extract_number)
    df['Skill Level'] = df['Skill Level'].apply(extract_number)
    df['Hook Size'] = df['Hook Size'].apply(extract_mean_size)
    df['Category'] = df.apply(update_category, axis=1)
    # Color doesn't have an impact in this case and doesn't really effect the pattern since a different color yarn can be picked
    df['Color'] = df['Color'].apply(check_multiple_colors).fillna('nan', inplace=True)
    #df['Stitches'] = preprocess_stitch_names(df['Stitches'])
    #df = preprocess_stitches(df)

    return df


from pgmpy.models import BayesianNetwork


def define_bayesian_network_structure():
    # Basic structure based on domain knowledge
    model_structure = [
        # generally yarn weight impacts the hook size as the yarn weight increases the hook you should use tends to increase
        ('Hook Size', 'Yarn Weight'),
        # Working with different sized yarns can impact the skill level for a project using a small yarn can tend to be difficult
        ('Skill Level', 'Yarn Weight'),
        # Hook sizes tend to change with the category you generally use a smaller hook size for a armigrumi since it i
        ('Category','Hook Size'),
        #('Category', 'Fiber Type'),
        # Different stitches are condidered harder than others
        # if there is increasing or deacreasing in the pattern this can effect the difficulty of the pattern
        # Stitches can be broke down into diffuculty level beginner, easy, intermediate and experienced
        # Switching between stitch types can also increase difficulty level in a crochet pattern so the number of stitches should also be accounted for
        # We should let the user specify if they only want a  specific stitch?
        ('Skill Level', 'Stitches'),
        # Stitch type also impacts the look of crochet piece as some create more of a tight-knit look where others are looser
        # Stitches therefore also impact the category as you would use a tighter stitch for a stuffy so stuffing doesn't fall out
        ('Category', 'Stitches')
    ]
    #unique_stitches = get_unique_stitches(data)
    #model_structure += [('Skill Level', stitch) for stitch in unique_stitches]
    return model_structure

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

def encode_data(data):
    encoded_data = data.copy()
    mappings = {}
    for column in data.columns:
        print(column)
        # Check if the column is of a categorical type or object
        if data[column].dtype == 'object' or pd.api.types.is_categorical_dtype(data[column]):
            encoded_data[column], mapping = pd.factorize(data[column])
            mappings[column] = {label: index for index, label in enumerate(mapping)}
        else:
            # Copy the data as is if not categorical
            encoded_data[column] = data[column]
    return encoded_data, mappings

def build_and_learn_bayesian_model(data, model_structure):
    # Initialize Bayesian Model
    encoded_data = data.copy()
    encoded_data, mappings = encode_data(data)
    model = BayesianNetwork(model_structure)
    print(model_structure)
    #plot_bayesian_network(model_structure) #TODO: Clean Up
    # Fit the model using an appropriate estimator
    # Replace any 'Unknown' with np.nan to treat them as missing values
    #model.fit(encoded_data, estimator=EM) #calculates the probabilities directly from the data frequencies
    #save_model(model, 'Bayseian_Model_Crochet_PatternsNoHot2EM.pkl')
    model = load_model('Bayseian_Model_Crochet_PatternsNoHot2EM.pkl')
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
    print(recommendation_attributes)
    #my_stitches_names = data.keys().to_list()[data.keys().to_list().index('Pattern Link') + 1:]
    for attr in recommendation_attributes:
        user_val = user_input_for_attribute(attr)
        if user_val is not None:
            # Convert to correct data types
            if attr in ['Skill Level', 'Yarn Weight']:
                user_val = int(user_val)
            elif attr == 'Hook Size':
                user_val = float(user_val)
            input_data[attr] = user_val
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
                new_values = top_values.copy()
                if attr in mappings.keys():
                    for i in range(len(top_values)):
                        for key, value in mappings[attr].items():
                            if value == top_values[i]:
                                if len(new_values) >= i:
                                    new_values.append(key)
                                else:
                                    new_values[i] = key
                                break
                            if top_values[i] == -1:
                                new_values.pop(i)
                                break
                probable_attributes[attr] = new_values
                # Initialize threshold for the first time
                if top_n == 1:
                    threshold_probabilities[attr] = .1  # set a threshold to 10%
            else:
                probable_attributes[attr] = [input_data[attr]]
                if attr in mappings.keys(): # directly use the provided input
                    for key, value in mappings[attr].items():
                        if value == input_data[attr]:
                            probable_attributes[attr] = [key]
                            break

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
                    new_values = []
                    if attr in mappings.keys():
                        for i in range(len(filtered_values)):
                            for key, value in mappings[attr].items():
                                if value == top_values[i]:
                                    new_values.append(key)
                                    break
                                if top_values[i] == -1:
                                    break
                        filtered_values = new_values

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
    # Ask user which attribute they want a recommendation for
    print("Which attribute would you like a recommendation for?")
    recommendation_attribute = input().strip()
    while recommendation_attribute not in recommendation_attributes_orig or recommendation_attribute in input_data:
        if recommendation_attribute not in recommendation_attributes_orig:
            print(f"Invalid attribute. Choose from: {', '.join(recommendation_attributes)}")
            recommendation_attribute = input().strip()
        if recommendation_attribute in input_data:
            print("You've already specified this attribute. Please choose another one.")
            recommendation_attribute = input().strip()
    # Perform inference for attribute recommendation
    if recommendation_attribute:
        #encode input_data
        for attr, in_val in input_data.items():
            if attr in mappings.keys():
                for key, val in mappings[attr].items():
                    if in_val == key:
                        input_data[attr] = val
        result = inference_engine.query(variables=[recommendation_attribute], evidence=input_data)
        top_results = get_top_recommendations_for_attribute(result, recommendation_attribute)
        print(f"Recommended {recommendation_attribute}:")
        if top_results[recommendation_attribute].to_list()[0] == -1: #[0] == -1:
            print("Unknown value with a probability of : ")# + top_results['Probability'])
            states = result.state_names[recommendation_attribute]
            probabilities = result.values
            df = pd.DataFrame({
                recommendation_attribute: states,
                'Probability': probabilities
            })
            df.sort_values('Probability', ascending=False, inplace=True)

            # Get the top N results
            top_results = top_results.iloc[1]
        # Decode Output
        if recommendation_attribute in mappings.keys():
            for key, value in mappings[recommendation_attribute].items():
                if value == top_results[recommendation_attribute].to_list()[0] :
                    print(f"{key} with a probability of {top_results['Probability'].to_list()[0]:.4f}")
                    break
        else:
            print(f"{top_results[recommendation_attribute]} with a probability of {top_results['Probability']:.4f}")


# GLOBAL USE FOR WEBSITE
filename = "crochet_patterns.csv"
data = pattern_csv_to_df(filename)
recommendation_attributes = [
        'Skill Level', 'Yarn Weight',
        'Hook Size', 'Category', 'Stitches'
    ]
attributes = recommendation_attributes.copy()
# Build and learn the Bayesian model
model_structure = define_bayesian_network_structure()
bayesian_model, mappings = build_and_learn_bayesian_model(data[attributes], model_structure)
print(mappings.keys())
inference_engine = VariableElimination(bayesian_model)

# Define attributes for recommendation
recommendation_attributes_orig = [
        'Skill Level', 'Yarn Weight',
        'Hook Size', 'Category', 'Stitches' #, 'Yarn Brand', 'Color'
]
recommendation_attribute = None


def process_input_data(form_data):
    input_data = {}
    stitches_names = data.keys().to_list()[data.keys().to_list().index('Pattern Link') + 1:]  # Adjust based on your data's column names for stitches

    # Process standard attributes
    for attr in ['Skill Level', 'Yarn Weight', 'Hook Size', 'Category', 'Stitches']:
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
    return input_data
def encode_user_input(attributes, user_inputs, mappings):
    """
    Encodes user input using stored mappings to integer codes.
    """
    print("user input: ", user_inputs)
    encoded_input = {}
    for attribute in attributes:
        if attribute in user_inputs:
            print(attribute)
            if attribute in mappings and user_inputs[attribute] in mappings[attribute].keys():
                # Map categorical string input to its corresponding integer code
                encoded_input[attribute] = mappings[attribute][user_inputs[attribute]]
            elif attribute not in mappings:
                # Assume it's already in the correct format (e.g., numeric input)
                encoded_input[attribute] = user_inputs[attribute]
            else:
                # Handle unseen categories or missing mappings
                encoded_input[attribute] = -1  # Or any other placeholder for unknown categories
        else:
            # Handle missing attributes if necessary
            encoded_input[attribute] = None
    print("ret encoded input")
    print(encoded_input)
    return encoded_input

# Web app
def recommend_patterns_from_bayes(input_data):
    encoded_input = encode_user_input(recommendation_attributes, input_data, mappings)
    print("encoded input: ")
    print(encoded_input)
    input_data = {k: v for k, v in encoded_input.items() if v is not None}
    probable_attributes = {}
    found_match = False
    top_n = 1
    max_prob = 0
    all_top_probs = {}
    print(input_data)
    threshold_probabilities = {}  # To store initial top probabilities for reference
    while not found_match and top_n <= 5:
        for attr in recommendation_attributes:
            if attr not in input_data:
                # Query the model
                result = inference_engine.query(variables=[attr], evidence=input_data)
                top_values, top_probs = get_top_recommendations(result, attr, top_n=top_n)
                all_top_probs[attr] = dict(zip(top_values, top_probs))
                new_values = top_values.copy()
                if attr in mappings.keys():
                    for i in range(len(top_values)):
                        for key, value in mappings[attr].items():
                            if value == top_values[i]:
                                if len(new_values) >= i:
                                    new_values.append(key)
                                else:
                                    new_values[i] = key
                                break
                            if top_values[i] == -1:
                                print("unspecified value")
                                break
                probable_attributes[attr] = new_values
                # Initialize threshold for the first time
                if top_n == 1:
                    threshold_probabilities[attr] = .1  # set a threshold to 10%
            else:
                probable_attributes[attr] = [input_data[attr]]
                if attr in mappings.keys(): # directly use the provided input
                    for key, value in mappings[attr].items():
                        if value == input_data[attr]:
                            probable_attributes[attr] = [key]
                            break

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
                    new_values = []
                    if attr in mappings.keys():
                        for i in range(len(filtered_values)):
                            for key, value in mappings[attr].items():
                                if value == top_values[i]:
                                    new_values.append(key)
                                    break
                                if top_values[i] == -1:
                                    break
                        filtered_values = new_values

                    probable_attributes[attr] = filtered_values
            recommended_patterns = recommend_patterns(data, probable_attributes)
            if not recommended_patterns.empty:
                found_match = True
    if not found_match:
        print("No matches found even after expanding search.")
        return pd.DataFrame()
    else:
        return recommended_patterns

# Web app
def get_recommendation_for_attribute(recommendation_attribute, input_data):
    encoded_input = encode_user_input(recommendation_attributes, input_data, mappings)
    input_data = {k: v for k, v in encoded_input.items() if v is not None}
    while recommendation_attribute not in recommendation_attributes_orig or recommendation_attribute in input_data:
        if recommendation_attribute not in recommendation_attributes_orig:
            print(f"Invalid attribute. Choose from: {', '.join(recommendation_attributes)}")
        if recommendation_attribute in input_data:
            print("You've already specified this attribute. Please choose another one.")
    # Perform inference for attribute recommendation
    if recommendation_attribute:
        result = inference_engine.query(variables=[recommendation_attribute], evidence=input_data)
        top_results = get_top_recommendations_for_attribute(result, recommendation_attribute)
        print(f"Recommended {recommendation_attribute}:")
        print(mappings.keys())
        print(recommendation_attribute)
        for index, row in top_results.iterrows():
            if recommendation_attribute in mappings.keys():
                for key, value in mappings[recommendation_attribute].items():
                    print(top_results[recommendation_attribute].tolist()[0])
                    if top_results[recommendation_attribute].tolist()[0] == -1:
                        print(f"Unspecified value: {key} with a probability of {row['Probability']:.4f}")
                        row[recommendation_attribute] = 'Unspecified'
                        return row
                    if value == row[recommendation_attribute]:
                        row[recommendation_attribute] = key
                        return row

            else:
                return row



def main():
    # Load and preprocess the data
    recommendation_attributes = [
        'Skill Level', 'Fiber Type', 'Yarn Weight',
        'Hook Size', 'Category', 'Stitches'
    ]
    filename = "venv/crochet_patterns.csv"

    data = pattern_csv_to_df(filename)
    attributes = recommendation_attributes
    unique_stitches = get_unique_stitches(data)
    #attributes.append(unique_stitches)
    # Define Bayesian network structure
    model_structure = define_bayesian_network_structure()
    #plot_bayesian_network(model_structure)
    # Build and learn the Bayesian model
    bayesian_model, mappings = build_and_learn_bayesian_model(data[recommendation_attributes], model_structure)
    inference_engine = VariableElimination(bayesian_model)
    # Define attributes for recommendation

    recommendation_attributes_orig = [
        'Skill Level', 'Fiber Type', 'Yarn Weight',
        'Hook Size', 'Category', 'Stitches'  # , 'Yarn Brand', 'Color', , 'Stitches'
    ]
    recommendation_attribute = None
    input_data, recommendation_attributes = get_user_input_for_attributes(recommendation_attributes, data)

    encoded_input = encode_user_input(recommendation_attributes, input_data, mappings)
    cleaned_data = {k: v for k, v in encoded_input.items() if v is not None}
    recommend_patterns_from_bayes(input_data)
    recommended_patterns = recommend_patterns_from_input(recommendation_attributes, cleaned_data, inference_engine,
                                                         data)
    print("Recommended Patterns:")
    if not recommended_patterns.empty:
        print(recommended_patterns[['Title', 'Pattern Link']])
        # Make recommendations based on input
    reccommend_attribute_based_on_user_input(recommendation_attributes_orig, input_data, data,
                                             recommendation_attributes, inference_engine)


if __name__ == "__main__":
    #main()
    print("hi")