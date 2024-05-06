from pgmpy.estimators import ExpectationMaximization as EM
from pgmpy.estimators import MaximumLikelihoodEstimator
import numpy as np
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
import re
import pandas as pd
from pgmpy.models import BayesianNetwork
import networkx as nx
import matplotlib.pyplot as plt


# Data collection and preperation section
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

def preprocess_stitches(df):
    keywords = [
        "Decrease",
        "Two Together",
        "Three Together",
        "Front Loop",
        "Back Loop",
        "Increase"
    ]
    decrease_keywords = [
        "Two Together",
        "Three Together"]

    include_keywords = [
        "Decrease",
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
                if keyword in include_keywords:
                    return keyword + ',' + parts
                elif keyword in decrease_keywords:
                    return keyword + ',' + parts
                else:
                    return parts
                #return keyword+','+ parts
        return stitch

    # Apply cleaning function to each stitch name
    cleaned_stitches = df_exploded.dropna().apply(clean_stitch_name).str.split(',').explode().str.strip()
    df['Cleaned_Stitches'] = cleaned_stitches.groupby(cleaned_stitches.index).apply(lambda x: ','.join(x.unique()))
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

    unique_stitches = df['Cleaned_Stitches'].str.split(',').explode().str.strip().drop_duplicates().sort_values().dropna()
    keywords_in_stitches = []
    for keyword in keywords:
        unique_stitches = unique_stitches.replace(keyword, '', regex=True)
        if keyword not in unique_stitches:
            keywords_in_stitches.append(keyword)
    unique_stitches_list = [stitch.strip() for stitch in unique_stitches]
    unique_stitches_list.extend(keywords)
    unique_stitches_set = sorted(set(unique_stitches_list))
    if "" in unique_stitches_set:
        unique_stitches_set.remove("")
    return unique_stitches_set

def hot_one_encode_stitches(data):
    # Split the 'Stitches' column into individual stitches
    stitch_columns = data['Stitches'].str.get_dummies(sep=',')
    # Merge these new columns back into the original DataFrame
    data = pd.concat([data, stitch_columns], axis=1)
    return data

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

def preprocess_stitches_for_bayesian_network(data):
    preprocess_stitches_df = data.copy()
    preprocess_stitches_df = preprocess_stitches(preprocess_stitches_df)
    unique_stitches = get_unique_stitches(preprocess_stitches_df)
    for index, dfrow in preprocess_stitches_df.iterrows():
        stitches_data = dfrow['Cleaned_Stitches']
        stitches = stitches_data.split(',')
        for unique_stitch in unique_stitches:
            preprocess_stitches_df.loc[index, unique_stitch] = True if unique_stitch in stitches else False
    return preprocess_stitches_df, unique_stitches


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
    #df['Color'] = df['Color'].apply(check_multiple_colors).fillna('nan', inplace=True)
    df, unique_stitches = preprocess_stitches_for_bayesian_network(df)

    return df, unique_stitches



# Baysian Model Section
def define_bayesian_network_structure():
    # Basic structure based on domain knowledge

    model_structure = [
        # generally yarn weight impacts the hook size as the yarn weight increases the hook you should use tends to increase
        ('Hook Size', 'Yarn Weight'),
        # Working with different sized yarns can impact the skill level for a project using a small yarn can tend to be difficult
        ('Skill Level', 'Yarn Weight'),
        # Hook sizes tend to change with the category you generally use a smaller hook size for a armigrumi since it i
        ('Category','Hook Size'),
        # Different stitches are condidered harder than others
        # if there is increasing or deacreasing in the pattern this can effect the difficulty of the pattern
        # Stitches can be broke down into diffuculty level beginner, easy, intermediate and experienced
        # Switching between stitch types can also increase difficulty level in a crochet pattern so the number of stitches should also be accounted for
        # We should let the user specify if they only want a  specific stitch?
        # Stitch type also impacts the look of crochet piece as some create more of a tight-knit look where others are looser
        # Stitches therefore also impact the category as you would use a tighter stitch for a stuffy so stuffing doesn't fall out
    ]
    model_structure += [('Skill Level', stitch) for stitch in unique_stitches]
    model_structure += [('Category', stitch) for stitch in unique_stitches]
    return model_structure



def plot_bayesian_network(edges, unique_stitches):
    G = nx.DiGraph()

    # Add all edges from the edges list, replacing unique stitch nodes with 'stitches'
    for source, target in edges:
        if source in unique_stitches:
            source = 'stitches'
        if target in unique_stitches:
            target = 'stitches'
        G.add_edge(source, target)

    # Remove self-loops on 'stitches' node if any
    G.remove_edges_from(nx.selfloop_edges(G))

    pos = nx.spring_layout(G, seed=42)  # positions for all nodes
    pos['stitches'] = np.array([0.3, 0.3])
    # Node colors and sizes
    node_colors = ['yellow' if node == 'stitches' else 'pink' for node in G.nodes()]
    node_sizes = [8000 if node == 'stitches' else 1000 for node in G.nodes()]  # Increased size for 'stitches'

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=2, alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif", font_weight='bold')

    # Add visual subnodes within 'stitches' node
    radius = 0.05
    sub_node_radius = 0.01
    angle_step = 360 / len(unique_stitches)
    sub_font_size = 4
    for i, stitch in enumerate(unique_stitches):
        angle = np.deg2rad(i * angle_step)
        sub_pos = (pos['stitches'][0] + radius * np.cos(angle), pos['stitches'][1] + radius * np.sin(angle))
        plt.gca().add_patch(plt.Circle(sub_pos, sub_node_radius, color='orange', ec='black', zorder=10))
        plt.text(sub_pos[0], sub_pos[1], stitch, fontsize=sub_font_size, ha='center', va='center', zorder=11)

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
        # Check if the column is of a categorical type or object
        if data[column].dtype == 'object' or pd.api.types.is_categorical_dtype(data[column]):
            encoded_data[column], mapping = pd.factorize(data[column])
            mappings[column] = {label: index for index, label in enumerate(mapping)}
        else:
            # Copy the data as is if not categorical
            encoded_data[column] = data[column]
    return encoded_data, mappings

def build_and_learn_bayesian_model(data, model_structure, load=False, doplot=False):
    # Initialize Bayesian Model
    encoded_data = data.copy()
    encoded_data, mappings = encode_data(data)
    if load:
        model = load_model('Bayseian_Model_Crochet_PatternsHotEM.pkl')
        if doplot:
            plot_bayesian_network(model_structure, unique_stitches)
        return model, mappings
    else:
        model = BayesianNetwork(model_structure)
        save_model(model, 'Bayseian_Model_Crochet_PatternsHotEM.pkl')
        model.fit(encoded_data, estimator=EM)
        if doplot:
            plot_bayesian_network(model_structure, unique_stitches)
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


# GLOBAL USE FOR WEBSITE
filename = "crochet_patterns.csv"
data, unique_stitches = pattern_csv_to_df(filename)

recommendation_attributes = [
        'Skill Level', 'Yarn Weight',
        'Hook Size', 'Category'
    ]
recommendation_attributes_orig = [
    'Skill Level', 'Yarn Weight',
    'Hook Size', 'Category', 'Stitches'  # , 'Yarn Brand', 'Color', , 'Stitches'
]
recommendation_attributes_out = [
    'Title','Skill Level', 'Yarn Weight',
    'Hook Size', 'Category', 'Stitches', 'Pattern Link'  # , 'Yarn Brand', 'Color', , 'Stitches'
]
attributes = recommendation_attributes.copy()

recommendation_attributes.extend(unique_stitches)
# Build and learn the Bayesian model
model_structure = define_bayesian_network_structure()
recommendation_data = data[recommendation_attributes]
bayesian_model, mappings = build_and_learn_bayesian_model(recommendation_data, model_structure)
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
            if 'Stitches' == attribute:
                # Convert the input string to a list of stitch names, trimming whitespace
                # If user_inputs['Stitches'] is a string like "stitch1, stitch2, stitch3"
                stitch_list = []
                if (type(user_inputs['Stitches']) == list):
                    stitch_list = user_inputs['Stitches']
                else:
                    stitch_list = [stitch.strip() for stitch in user_inputs['Stitches'].split(',')]
                stitch_input = {stitch: 1 for stitch in stitch_list}
                for key, value in stitch_input.items():
                    encoded_input[key] = value
            if attribute in mappings and user_inputs[attribute] in mappings[attribute].keys():
                # Map categorical string input to its corresponding integer code
                if attribute != "Stitches":
                    encoded_input[attribute] = mappings[attribute][user_inputs[attribute]]
            elif attribute not in mappings:
                # Assume it's already in the correct format (e.g., numeric input)
                if attribute != "Stitches":
                    encoded_input[attribute] = user_inputs[attribute]
            else:
                # Handle unseen categories or missing mappings
                encoded_input[attribute] = -1  # Or any other placeholder for unknown categories
        else:
            # Handle missing attributes if necessary
            if attribute != "Stitches":
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
    recommended_patterns = recommend_patterns(data, input_data)
    if recommended_patterns.empty:
        print("No pattern match for given input data")
        return pd.DataFrame()
    elif recommended_patterns.shape[0] == 1:
        print("Already only one pattern for given input data")
        return recommended_patterns
    # TODO: Consider OR for searching based on stitches
    while not found_match and top_n <= 5:
        non_input_attributes = []
        for attr in recommendation_attributes:
            if attr not in input_data:
                non_input_attributes.append(attr)
        # Query the model
        print("getting result")
        result = inference_engine.query(variables=non_input_attributes, evidence=input_data)
        print("result got")
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
    encoded_input = encode_user_input(recommendation_attributes, input_data, mappings)
    input_data = {k: v for k, v in encoded_input.items() if v is not None}
    while recommendation_attribute not in recommendation_attributes_orig or recommendation_attribute in input_data:
        if recommendation_attribute not in recommendation_attributes_orig:
            print(f"Invalid attribute. Choose from: {', '.join(recommendation_attributes)}")
        if recommendation_attribute in input_data:
            print("You've already specified this attribute. Please choose another one.")
    # Perform inference for attribute recommendation
    if recommendation_attribute:
        if recommendation_attribute == "Stitches":
            print("unique_stitches")
            print(type(list(unique_stitches)))
            result_map = inference_engine.map_query(variables=list(unique_stitches), evidence=input_data, elimination_order="MinWeight")
            print(result_map)
            print([(key, val) for key, val in result_map.items()])
            stitches_str = ""
            for key, val in result_map.items():
                if key in unique_stitches:
                    if len(stitches_str) == 0 and val == 1:
                        stitches_str = key
                    elif val == 1:
                        stitches_str += " ," + key
            return {"Stitches": stitches_str}
        else:
            result_map = inference_engine.map_query(variables=recommendation_attribute, evidence=input_data,
                                                    elimination_order="MinWeight")

            if recommendation_attribute in mappings.keys():
                for key, value in mappings[recommendation_attribute].items():
                    if result_map[recommendation_attribute] == -1:
                        result_map[recommendation_attribute] = 'Unspecified'
                        return result_map
                    if value == result_map[recommendation_attribute]:
                        result_map[recommendation_attribute] = key
                        return result_map
            else:
                return result_map


def evaluate_model(df, recommendation_attributes):
    # Build and learn the Bayesian model
    model_structure = define_bayesian_network_structure()
    bayesian_model, mappings = build_and_learn_bayesian_model(data[recommendation_attributes], model_structure)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    bayesian_model, mappings = build_and_learn_bayesian_model(data[recommendation_attributes], model_structure)


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
    bayesian_model = build_and_learn_bayesian_model(train_df[recommendation_attributes], model_structure)


    # Predict the 'Category' for test data
    test_df['predicted_category'] = predict(bayesian_model, test_df, 'Category')

    # Evaluate accuracy
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(test_df['Category'], test_df['predicted_category'])
    print(f'Accuracy of the Bayesian network for Category prediction: {accuracy:.2f}')


def main():
    # Load and preprocess the data
    recommendation_attributes = [
        'Skill Level', 'Yarn Weight',
        'Hook Size', 'Category', 'Stitches'
    ]
    filename = "venv/crochet_patterns.csv"

    data, unique_stitches = pattern_csv_to_df(filename)
    attributes = recommendation_attributes.copy()
    attributes.append(unique_stitches)
    # Define Bayesian network structure
    model_structure = define_bayesian_network_structure()
    #plot_bayesian_network(model_structure)
    # Build and learn the Bayesian model
    bayesian_model, mappings = build_and_learn_bayesian_model(recommendation_data, model_structure)
    inference_engine = VariableElimination(bayesian_model)
    # Define attributes for recommendation

    recommendation_attributes_orig = [
        'Skill Level', 'Yarn Weight',
        'Hook Size', 'Category', 'Stitches'  # , 'Yarn Brand', 'Color', , 'Stitches'
    ]
    recommendation_attribute = None
    input_data, recommendation_attributes = get_user_input_for_attributes(recommendation_attributes, data)

    encoded_input = encode_user_input(recommendation_attributes, input_data, mappings)
    cleaned_data = {k: v for k, v in encoded_input.items() if v is not None}
    recommended_patterns = recommend_patterns_from_bayes(input_data)
   # recommended_patterns = recommend_patterns_from_input(recommendation_attributes, cleaned_data, inference_engine,
    #                                                     data)
    print("Recommended Patterns:")
    if not recommended_patterns.empty:
        print(recommended_patterns[['Title', 'Pattern Link']])
        # Make recommendations based on input
    rec = get_recommendation_for_attribute('Stitches', input_data)
    print(rec)
   # reccommend_attribute_based_on_user_input(recommendation_attributes_orig, input_data, data,
    #                                         recommendation_attributes, inference_engine)


if __name__ == "__main__":
    #print("hi")
    main()