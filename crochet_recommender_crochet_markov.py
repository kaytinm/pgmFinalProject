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
    elif any(word in title_lower for word in ['coaster', 'rug', 'wall', 'cozy',  'mat', 'cocoon', 'clutch', 'duster', 'rauna', 'curtain', 'cushion',
                                              'cushion', 'cuff', 'pin', 'wallet', 'scrubbie', 'basket', 'backpack', 'basket', 'hanger',
                                              'ornament', 'holder', 'dish', 'cloth', 'sack', 'boget', 'scrub',
                                              'shoulder strap', 'bunting', 'banner', 'pot', 'scrub',
                                              'clutch', 'runner', 'keychain', 'coaster', 'bunting', 'trivit', 'ring', 'motif',
                                              'earring', 'pillow', 'trivet','bracelet', 'towel', 'bag', 'tote', 'pouch','gift', 'planter', 'floor', 'collar']):
        return 'Accessory'
    elif any(word in title_lower for word in
             ['cardi', 'shawl', 'vest', 'tank', 'cardigan', 'onesie', 'booties', 'sock' ,'mit', 'raglan', 'cardi',
              'cardi', 'crop', 'sweater', 'hoodie', 'wrap', 'shrug', 'poncho', 'top', 'cami', 'stole', 'set',
              'skirt', 'dress', 'pullover', 'coat', 'swoncho', 'ruana', 'romper', 'gloves', 'blouse', 'kimono', 'shardi',
              'smock', 'sleeve', 'jumper', 'sock', 'hoodie', 'stocking', 'kimono', 'pochette', 'bra', 'cover',
              'afgan', 'turtleneck', ' tee', 'jacket', 'shorts', 'cover up', 'tunic', 'slouch',  'tunic', 'capelet'])and 'octopus' not in title_lower:
        return 'Clothing'
    elif any(word in title_lower for word in ['beanie', 'hat', 'ear', 'head', 'bonnet', 'hood',
                                              'hood', 'cap', 'hair', 'toque', 'kerchief', 'balaclava', 'bandana', 'beret']):
        return 'Clothing'
    elif any(word in title_lower for word in ['scarf', 'cowl', 'neck', 'afghan', 'hood']):
        return 'Clothing'
    elif any(word in title_lower for word in
             ['amigurumi', 'carrots', 'idol', 'cactus', 'wreath', 'deer', 'gingerbread', 'toy', 'snowman',
              'penguin', 'octopus', 'jellyfish', 'owl', 'chick', 'panda', 'hippo', 'ice cream',
              'dog', 'otter', 'pal', 'bear', 'cat', 'unicorn', 'moose', 'whale', 'buck', 'whale',
              'lion', 'bear', 'monkey', 'luffy', 'bee', 'puppy', 'lovey', 'sloth', 'animal', 'bumble', 'creature', 'gingerbread',
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
        else:
            return round(float(yardage_str), -2)

    copy_data['Average Yardage'] = copy_data['Yardage'].apply(average_yardage)

    bins = [0, 4999, 9999, 14999, 19999, 20000]
    bin_labels = ['0-4999', '5000-9999', '10000-14999', '15000-19999', '20000+']

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
    weights = yarn_weight_string.split(',')
    weights = [w.strip() for w in weights]

    numeric_weights = [yarn_weights[weight] for weight in weights if weight in yarn_weights]

    if numeric_weights:
        #return max(set(numeric_weights))
        #return str(weights)
        return round(sum(numeric_weights) / len(numeric_weights),1)
    else:
        return -1  # In case of a typo or unrecognised yarn weight


def extract_number(text):
    # Search for numbers in the string
    match = re.search(r'\d+', text)
    if match:
        return int(match.group())
    return np.nan

def extract_mean_size(text):
    # Find all numbers (integers or decimals) before "mm"
    if text == '9” High x 25” Circumference': # Special Case
        return 9
    numbers = re.findall(
        r'\b\d+\.?\d*(?=\s*mm)|(?<=/)\d+\.?\d*|\b\d+\.?\d*(?=/)|(?<=")\s*\d+\.?\d*|\b\d+\.?\d*(?=\s*")', text)

    if numbers:
        numbers = list(map(float, numbers))
        if len(numbers) > 1:
            #return str(numbers)
            return round((sum(numbers) / len(numbers))*4)/4 # Return the mean of the numbers round hook sizes are by .25s
        return (numbers[0])#round(numbers[0]*4)/4 # Return the number directly if only one
    else:
        return -1


def pattern_csv_to_df(filename):
    df = pd.read_csv(filename)

    df = df.drop_duplicates(subset=['Title', 'Pattern Link'])
    df.dropna(subset=['Skill Level', 'Yarn Weight', 'Hook Size', 'Fiber Type'], inplace=True)

    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # TODO: Change Yarn Name and Yarn Brand to Fiber Type

    df['Average Hook Size'] = df['Hook Size'].apply(extract_mean_size)
    df['Category'] = df.apply(update_category, axis=1)
    df = get_yardage_range(df)
    df['Average Yarn Weight'] = df['Yarn Weight'].apply(average_weight)
    unique_vals = df.apply(lambda x: x.unique())
    return df



# Baysian Model Section
#TODO: change to markov model
def define_network_structure():
    model_structure = [
        ('Average Yarn Weight', 'Skill Level'),
        ('Average Yarn Weight', 'Average Hook Size'),
        ('Average Hook Size', 'Category'),
        ('Fiber Type', 'Average Yarn Weight'),
        ('Fiber Type', 'Category'),
        ('Yardage Range', 'Category')
    ]
    return model_structure


def plot_network(model_structure):
    G = nx.DiGraph(model_structure)
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='skyblue')

    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=2, alpha=0.5, edge_color='gray', arrows=False)

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
        if data[column].dtype == 'object' or pd.api.types.is_categorical_dtype(data[column]):
            encoded_data[column], mapping = pd.factorize(data[column])
            mappings[column] = {label: index for index, label in enumerate(mapping)}
        else:
            encoded_data[column] = data[column]
    return encoded_data, mappings


def calculate_joint_probabilities(data, variables):
    clean_data = data.dropna(subset=variables)
    var_levels = [clean_data[var].astype('category').cat.categories for var in variables]
    all_combinations = pd.MultiIndex.from_product(var_levels, names=variables).to_frame(index=False)
    frequency_table = clean_data.groupby(variables).size().reset_index(name='counts')
    frequency_table = all_combinations.merge(frequency_table, on=variables, how='left').fillna(0)
    total_counts = frequency_table['counts'].sum()
    frequency_table['probability'] = frequency_table['counts'] / total_counts
    return frequency_table

def create_factors_for_markov_network(data, model_structure):
    model = MarkovNetwork(model_structure)

    for edge in model.edges():
        variables = list(edge)
        prob_table = calculate_joint_probabilities(data, variables)
        state_names = {var: list(set(prob_table[var].values)) for var in variables}
        # Create a factor from the probability table
        values = prob_table['probability'].values
        cardinality = [len(data[var].unique()) for var in variables]
        factor = DiscreteFactor(variables, cardinality, values, state_names=state_names)
        # Add the factor to the model
        model.add_factors(factor)

    return model

def build_and_learn_markov_model(data, model_structure, load=False, doplot=False):
    # Initialize Markov Model
    _, mappings = encode_data(data)
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
    states = result.state_names[result.variables[0]]
    probabilities = result.values
    df = pd.DataFrame({
        attribute: states,
        'Probability': probabilities
    })
    df.sort_values('Probability', ascending=False, inplace=True)

    top_results = df.head(top_n)
    return top_results

def decode_attributes(encoded_attributes, mappings):
    decoded_attributes = {}
    for key, value in encoded_attributes.items():
        if key in mappings and value[0] in mappings[key]:
            decoded_attributes[key] = mappings[key][value[0]]
        else:
            decoded_attributes[key] = "Unknown"
    return decoded_attributes


def filter_dataframe(df, conditions):
    query_parts = []

    for key, values in conditions.items():
        if isinstance(values, list):
            conditions_list = [f"`{key}` == {repr(value)}" for value in values]
            query_parts.append(f"({' | '.join(conditions_list)})")
        else:
            query_parts.append(f"`{key}` == {repr(values)}")

    query_string = ' & '.join(query_parts)

    return df.query(query_string)


def recommend_patterns(data, attributes):
    if not attributes:
        return pd.DataFrame()  # Return empty DataFrame if no attributes are provided
    recommended_patterns = data
    for attr, values in attributes.items():
        if type(values) != list:
            values = [values]
        if values:
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

model_structure = define_network_structure()
recommendation_data = data[recommendation_attributes]
markov_model, mappings = build_and_learn_markov_model(recommendation_data, model_structure)
inference_engine = VariableElimination(markov_model)

recommendation_attributes_orig = recommendation_attributes
recommendation_attribute = None


def process_input_data_markov(form_data):
    input_data = {}

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
def recommend_patterns_from_markov(input_data):
    print(input_data)
    #encoded_input = encode_user_input(recommendation_attributes_orig, input_data, mappings)
    input_data = {k: v for k, v in input_data.items() if v is not None}
    probable_attributes = {}
    found_match = False
    top_n = 1
    threshold_probabilities = {}
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

    while not found_match and top_n <= 20:
        #result_map = inference_engine.map_query(variables=non_input_attributes, evidence=input_data)
        top_values, top_probs = get_top_recommendations(result, top_n=top_n)
        new_values = top_values.copy()
        for a in input_data:
            new_values[a] = [input_data[a]]
        recommended_patterns = recommend_patterns(data, new_values)
        if not recommended_patterns.empty:
            print("FOUND")
            print(f"Found matches with top_{top_n} recommendations for {list(new_values.keys())}.")
            print(recommended_patterns)
            return recommended_patterns
        else:
            top_n += 1
    if not found_match:
        print("No matches found even after expanding search.")
        return pd.DataFrame()

# Web app
def get_recommendation_for_attribute_markov(recommendation_attribute, input_data):
    #encoded_input = encode_user_input(recommendation_attributes_orig, input_data, mappings)

    input_data = {k: v for k, v in input_data.items() if v is not None}
    if recommendation_attribute not in recommendation_attributes_orig or recommendation_attribute in input_data:
        if recommendation_attribute not in recommendation_attributes_orig:
            print(f"Invalid attribute. Choose from: {', '.join(recommendation_attributes)}")
        if recommendation_attribute in input_data:
            print("You've already specified this attribute. Please choose another one.")
        return {}
    # Perform inference for attribute recommendation
    if recommendation_attribute:
        if (type(recommendation_attribute)!= list):
            recommendation_attribute = [recommendation_attribute]
        result_map = inference_engine.map_query(variables=recommendation_attribute, evidence=input_data,
                                                    elimination_order="MinWeight")

        return result_map

def find_nearest(array, value):
   array = np.asarray(array)
   if all(isinstance(x, (int, float, complex)) for x in array):
       idx = (np.abs(array - value)).argmin()
   else:
       return None
   return array[idx]



from sklearn import metrics
def evaluate_model(df, recommendation_attributes, target_val):
    model_structure = define_network_structure()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Generate predictions for a specified column
    def predict(inference, data, target):
        predictions = []
        for _, row in data.iterrows():
            evidence = row.to_dict()
            evidence.pop(target)  # Remove the target variable from evidence
            new_evidence = evidence.copy()
            for k, v in evidence.items():
                if v not in train_df[k].values:
                    nearest = find_nearest(train_df[k].values, v)
                    if nearest == None:
                        del(new_evidence[k])
                    else:
                        new_evidence[k] = nearest
            predicted = inference.map_query(variables=[target], evidence=new_evidence, show_progress=False)
            predictions.append(predicted[target])
        return predictions

    # Train the model
    model, _ = build_and_learn_markov_model(train_df[recommendation_attributes], model_structure)
    inference = VariableElimination(model)

    test_df['predicted_category'] = predict(inference, test_df[recommendation_attributes], target=target_val)
    true_values = test_df[target_val]
    predicted_values = test_df['predicted_category']
    if "Average" in target_val:  # encode
        combined_list = list(true_values.copy())
        combined_list.extend(predicted_values)
        unique_values = sorted(set(combined_list))

        encoding_dict = {value: idx for idx, value in enumerate(unique_values, start=1)}

        encoded_list1 = [encoding_dict[value] for value in true_values.values]
        true_values = encoded_list1
        encoded_list2 = [encoding_dict[value] for value in predicted_values]
        predicted_values = encoded_list2
    accuracy = metrics.accuracy_score(true_values, predicted_values)
    print(f"Accuracy units: {accuracy:.2f}")

    precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(true_values, predicted_values, average='macro')
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)
    if (type(true_values) != list):
        if (str(true_values.values[0]).isnumeric()):
            r2 = metrics.r2_score(true_values, predicted_values)
            print("R-squared:", r2)
    elif (str(true_values[0]).isnumeric()):
        r2 = metrics.r2_score(true_values, predicted_values)
        print("R-squared:", r2)


def do_eval(df, recommendation_attributes):
    for attr in recommendation_attributes:
        print("Eval for " , attr)
        evaluate_model(df, recommendation_attributes, attr)

def main():
    # Load and preprocess the data
    recommendation_attributes = [
        'Skill Level', 'Average Yarn Weight',
        'Fiber Type', 'Yardage Range', 'Category', 'Average Hook Size'

    ]
    filename = "crochet_patterns2.csv"

    data = pattern_csv_to_df(filename)
    do_eval(data, recommendation_attributes)
    attributes = recommendation_attributes.copy()
    model_structure = define_network_structure()
    #plot_network(model_structure)
    markov_model, mappings = build_and_learn_markov_model(recommendation_data, model_structure)
    inference_engine = VariableElimination(markov_model)

    recommendation_attribute = None
    input_data, recommendation_attributes = get_user_input_for_attributes(recommendation_attributes, data)

    #encoded_input = encode_user_input(recommendation_attributes, input_data, mappings)
   # cleaned_data = {k: v for k, v in encoded_input.items() if v is not None}
    recommended_patterns = recommend_patterns_from_markov(input_data)
   # recommended_patterns = recommend_patterns_from_input(recommendation_attributes, cleaned_data, inference_engine,
    #                                                     data)
    print("Recommended Patterns:")
    if not recommended_patterns.empty:
        print(recommended_patterns[['Title', 'Pattern Link']])
        # Make recommendations based on input
    #rec = get_recommendation_for_attribute_markov('Average Hook Size', input_data)
   # print(rec)
   # reccommend_attribute_based_on_user_input(recommendation_attributes_orig, input_data, data,
    #                                         recommendation_attributes, inference_engine)


if __name__ == "__main__":
    #print("hi")
    main()