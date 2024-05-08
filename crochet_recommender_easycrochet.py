import sklearn.metrics
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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import random
import pickle

# Data collection and preperation section
def update_category(row):
    title_lower = row['Title'].lower()
    if any(word in title_lower for word in ['blanket', 'throw']):
        return 'Blanket'
    elif any(word in title_lower for word in ['beanie', 'hat', 'ear', 'head', 'bonnet'] ) and 'bunny' not in title_lower:
        return 'Clothing'
    elif any(word in title_lower for word in ['scarf', 'cowl', 'neck']):
        return 'Clothing'
    elif any(word in title_lower for word in
             ['shawl', 'vest', 'cardigan', ' tee', 'sweater', 'sock','hoodie', 'warmer','jumper', 'capelet'
              'collar', 'wrap', 'shrug', 'poncho', 'top', 'skirt', 'dress', 'kimono', 'cover up'
              'afghan', 'stocking', 'mask', 'mit', 'boot', 'glove'])and 'octopus' not in title_lower:
        return 'Clothing'
    elif any(word in title_lower for word in ['coaster', 'bow', 'placemat', 'cozy', 'ornament', 'crown', 'keychain', 'holder', 'towel', 'garland', 'bed', 'kitchen'
                                              'basket', 'bag', 'stocking', 'cushion', 'applique','scrunchie', 'mask', 'cloth', 'pillow', 'purse', 'soap', 'cover'
                                              'babygrow', 'mit', 'doily','pouffe', 'set', 'boot', 'scrub', 'glove', ' pad', 'pouch', ' mat', 'wall']):
        return 'Accessory'
    elif any(word in title_lower for word in
             ['amigurumi', 'penguin', 'octopus', 'jellyfish', 'owl', 'dog', 'lion', 'bear', 'monkey', 'luffy', 'bee', 'lovey', 'pineapple'
              'panda', 'gnome', 'santa', 'frankenstein', 'giaraffe','frog', 'squid', 'worm', 'bunny', 'chick', 'worm','candy cane', 'motif'
              'pumpkin', 'bunnies', 'monster', 'reindeer', 'tiger','slimes', 'christmas star', 'sunflower', 'cloud', 'egg', 'pizza']):
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
        if 'Tunisian' in stitch:
            return 'Tunisian'
        for keyword in keywords:
            if keyword in stitch:
                parts = stitch.replace(keyword, '')
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
    decrease_keywords = [
        "Two Together",
        "Three Together"]

    include_keywords = [
        "Decrease",
        "Increase"
    ]
    unique_stitches = df['Cleaned_Stitches'].str.split(',').explode().str.strip().drop_duplicates().sort_values().dropna()
    keywords_in_stitches = []
    for keyword in keywords:
        if keyword in include_keywords:
            unique_stitches = unique_stitches.replace(keyword, '', regex=True)
        elif keyword in decrease_keywords:
            unique_stitches = unique_stitches.replace(keyword, '', regex=True)

    unique_stitches_list = [stitch.strip() for stitch in unique_stitches]
    unique_stitches_list.extend(include_keywords)
    unique_stitches_set = sorted(set(unique_stitches_list))
    if "" in unique_stitches_set:
        unique_stitches_set.remove("")
    return unique_stitches_set

def hot_one_encode_stitches(data):
    stitch_columns = data['Stitches'].str.get_dummies(sep=',')
    data = pd.concat([data, stitch_columns], axis=1)
    return data

def extract_number(text):
    # Search for numbers in the string
    match = re.search(r'\d+', text)
    if match:
        return int(match.group())
    return np.nan

def extract_mean_size(text):
    # Find all numbers (integers or decimals) before "mm"
    numbers = re.findall(r'\b\d+\.?\d*(?=\s*mm)', text)
    if numbers:
        numbers = list(map(float, numbers))
        if len(numbers) > 1:
            return sum(numbers) / len(numbers)
        return numbers[0]
    return text

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
            preprocess_stitches_df.loc[index, unique_stitch] = 1 if unique_stitch in stitches else 0
    return preprocess_stitches_df, unique_stitches


def pattern_csv_to_df(filename):
    df = pd.read_csv(filename)

    # Process and clean the data
    df_unique = df.drop_duplicates(subset=['Title', 'Pattern Link'])
    df.dropna(subset=['Skill Level', 'Yarn Weight', 'Hook Size', 'Stitches'], inplace=True)

    # Clean whitespace
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df['Yarn Weight'] = df['Yarn Weight'].apply(extract_number)
    df['Skill Level'] = df['Skill Level'].apply(extract_number)
    df['Hook Size'] = df['Hook Size'].apply(extract_mean_size)
    df['Category'] = df.apply(update_category, axis=1)
    df, unique_stitches = preprocess_stitches_for_bayesian_network(df)

    return df, unique_stitches



# Baysian Model Section
def define_bayesian_network_structure():
    # Basic structure based on domain knowledge

    model_structure = [
        ('Yarn Weight', 'Skill Level'),
        ('Yarn Weight', 'Hook Size'),
        ('Hook Size', 'Category')
    ]
    model_structure += [(stitch, 'Skill Level') for stitch in unique_stitches]
    return model_structure
def plot_bayesian_network(edges, stitches):
    G = nx.DiGraph()

    for source, target in edges:
        if source in unique_stitches:
            source = 'Stitches'
        if target in unique_stitches:
            target = 'Stitches'
        G.add_edge(source, target)

    G.remove_edges_from(nx.selfloop_edges(G))

    pos = {
        "Yarn Weight": (0.5, 1),  # Top-most level
        "Hook Size": (0.3, 0.5),  # Middle , left
        "Category": (0.3, 0),  # Bottom , left
        "Skill Level": (0.7, 0),  # Bottom , right
        "Stitches": (1, 0.3) # right, middle
    }

    node_colors = ['yellow' if node == 'Stitches' else 'pink' for node in G.nodes()]
    node_sizes = [8000 if node == 'Stitches' else 1000 for node in G.nodes()]  # Increased size for 'stitches'
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='-|>', arrows=True, arrowsize=20, width=2, alpha=0.5,
                           edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif", font_weight='bold')
    # Add visual subnodes within 'stitches' node
    radius = 0.05
    sub_node_radius = 0.01
    angle_step = 360 / len(unique_stitches)
    sub_font_size = 4
    for i, stitch in enumerate(unique_stitches):
        angle = np.deg2rad(i * angle_step)
        sub_pos = (pos['Stitches'][0] + radius * np.cos(angle), pos['Stitches'][1] + radius * np.sin(angle))
        plt.gca().add_patch(plt.Circle(sub_pos, sub_node_radius, color='orange', ec='black', zorder=10))
        plt.text(sub_pos[0], sub_pos[1], stitch, fontsize=sub_font_size, ha='center', va='center', zorder=11)

    plt.title("Bayesian Network for Crochet Patterns")
    plt.show()



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
            # Decode the value using the mapping
            decoded_attributes[key] = mappings[key][value[0]]
        else:
            # If the mapping or value is not found, return the original value or handle the missing case
            decoded_attributes[key] = "Unknown"
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
        if values:
            recommended_patterns = recommended_patterns[recommended_patterns[attr].isin(values)]
            if recommended_patterns.empty:
                break
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
filename = "venv/crochet_patterns.csv"
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
bayesian_model, mappings = build_and_learn_bayesian_model(recommendation_data, model_structure, doplot=True)
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
                stitch_list = []
                if (type(user_inputs['Stitches']) == list):
                    stitch_list = user_inputs['Stitches']
                else:
                    stitch_list = [stitch.strip() for stitch in user_inputs['Stitches'].split(',')]
                stitch_input = {stitch: 1 for stitch in stitch_list}
                for key, value in stitch_input.items():
                    encoded_input[key] = value
            else:
                if attribute in mappings and user_inputs[attribute] in mappings[attribute].keys():
                    encoded_input[attribute] = mappings[attribute][user_inputs[attribute]]
                elif attribute not in mappings:
                    encoded_input[attribute] = user_inputs[attribute]
                else :
                    encoded_input[attribute] = -1
        else:
            if attribute != "Stitches":
                encoded_input[attribute] = None
    return encoded_input



# Web app
def recommend_patterns_from_bayes(input_data, inference_engine=inference_engine):
    encoded_input = encode_user_input(recommendation_attributes_orig, input_data, mappings)
    input_data = {k: v for k, v in encoded_input.items() if v is not None}
    probable_attributes = {}
    found_match = False
    top_n = 1
    threshold_probabilities = {}
    recommended_patterns = recommend_patterns(data, input_data)
    non_input_attributes = []
    for attr in recommendation_attributes:
        if attr not in input_data:
            non_input_attributes.append(attr)
    # Query the model
    print(non_input_attributes)
    print(input_data)
    result = inference_engine.query(variables=non_input_attributes, evidence=input_data)
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
                        threshold_probabilities[attr] = .1
            else:
                new_values[attr] = [input_data[attr]]
                if attr in mappings.keys():
                    for key, value in mappings[attr].items():
                        if value == input_data[attr]:
                            new_values[attr] = [key]
                            break

        recommended_patterns = recommend_patterns(data, new_values)
        if not recommended_patterns.empty:
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
    if recommendation_attribute not in recommendation_attributes_orig or recommendation_attribute in input_data:
        if recommendation_attribute not in recommendation_attributes_orig:
            print(f"Invalid attribute. Choose from: {', '.join(recommendation_attributes)}")
        if recommendation_attribute in input_data:
            print("You've already specified this attribute. Please choose another one.")
        return {}
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
            if (type(recommendation_attribute)!= list):
                recommendation_attribute = [recommendation_attribute]
            result_map = inference_engine.map_query(variables=recommendation_attribute, evidence=input_data,
                                                    elimination_order="MinWeight")

            if recommendation_attribute[0] in mappings.keys():
                for key, value in mappings[recommendation_attribute].items():
                    if result_map[recommendation_attribute] == -1:
                        result_map[recommendation_attribute] = 'Unspecified'
                        return result_map
                    if value == result_map[recommendation_attribute]:
                        result_map[recommendation_attribute] = key
                        return result_map
            else:
                return result_map

def random_list_within_list(values_list):
    length = random.randint(1, len(values_list)-1)
    return [random.choice(values_list) for _ in range(length)]


def custom_accuracy(true_values, predicted_values):
    if len(true_values) != len(predicted_values):
        raise ValueError("true_values and predicted_values must have the same length.")

    correct = 0
    total = len(true_values)

    for true, pred in zip(true_values, predicted_values):
        if true == pred:
            correct += 1

    accuracy = correct / total
    return accuracy
# EVALUATION

def evaluate_model(df, recommendation_attributes, category_eval="Hook Size"):
    # Build and learn the Bayesian model
    model_structure = define_bayesian_network_structure()
    bayesian_model, mappings = build_and_learn_bayesian_model(data[recommendation_attributes], model_structure)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    rec_attrs = recommendation_attributes.copy()
    encode_train_df, map = encode_data(train_df[rec_attrs])
    encoded_test_df, map = encode_data(test_df[rec_attrs])
    if 'Stitches' in rec_attrs:
        rec_attrs.remove('Stitches')
    a = recommendation_attributes.copy()
    if 'Stitches' in a:
        a.remove('Stitches')

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def predict(model, data, target, known_values):
        inference = VariableElimination(model)
        predictions = []
        if "Stitches" == target:
            predictions = {}

        for _, row in data.iterrows():
            evidence = row.to_dict()
            evidence_vals = {k: v for k, v in evidence.items() if k in a}
            vals = encode_user_input(a, evidence_vals, mappings)
            cleaned_data = {k: v for k, v in vals.items() if v is not None}
            old_target = target
            if "Stitches" == target:
                for stitch in unique_stitches:
                    cleaned_data.pop(stitch)
                    evidence.pop(stitch)
            else:
                cleaned_data.pop(target)
                evidence.pop(target)
            for k, v in cleaned_data.items():
                if v not in encode_train_df[k].values:
                    nearest = find_nearest(encode_train_df[k].values, v)
                    cleaned_data[k] = nearest
            if target == "Stitches":
                predicted = inference.map_query(variables=unique_stitches, evidence=cleaned_data, elimination_order="MinWeight",
                                                show_progress=False)
            else:
                predicted = inference.map_query(variables=[target], evidence=cleaned_data, elimination_order="MinWeight", show_progress=False)
            if target == "Stitches":
                for stitch in unique_stitches:
                    if stitch in predictions.keys():
                        predictions[stitch].append((predicted[stitch]))
                    else:
                        predictions[stitch] = [predicted[stitch]]

            else:
                predictions.append(predicted[old_target])
        return predictions

    # Train the model
    known_values = {attr: np.unique(train_df[attr]) for attr in rec_attrs}

    bayesian_model, mappings = build_and_learn_bayesian_model(data[rec_attrs], model_structure)


    if category_eval == "Stitches":
        true_values = {}
        stitch_map = predict(bayesian_model, test_df, category_eval, known_values)
        predicted_values = stitch_map
        for stitch in unique_stitches:
            true_values[stitch] = encoded_test_df[stitch].to_list()
        true_vals = true_values
        for k,v in true_vals.items():
            true_vals[k] = np.array(true_vals[k])
        for k,v in predicted_values.items():
            predicted_values[k] = np.array(predicted_values[k])
        all_true = []
        all_pred = []
        for stitch in unique_stitches:
            all_true.extend(true_values[stitch])
            all_pred.extend(predicted_values[stitch])
        overall_accuracy = accuracy_score(all_true, all_pred)
        print(f"Overall accuracy: {overall_accuracy:.2f}")

        metrics = {}
        for stitch, preds in predicted_values.items():
            true = true_values[stitch]
            precision, recall, f1_score, _ = precision_recall_fscore_support(true, preds, average='binary')
            metrics[stitch] = {
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1_score
            }
        prec = []
        rec = []
        f1 = []
        for stitch in unique_stitches:
            prec.append(metrics[stitch]['Precision'])
            rec.append(metrics[stitch]['Recall'])
            f1.append(metrics[stitch]['F1 Score'])

        # Print precision, recall, F1 for each stitch type
        print("F1: ", np.array(f1).mean())
        print('Precision: ',np.array(prec).mean())
        print('Recall: ', np.array(rec).mean())

        return

    else:
        test_df['predicted_category'] = predict(bayesian_model, test_df, category_eval, known_values)

    tolerance_level = 0.05
    true_values = encoded_test_df[category_eval]
    predicted_values = test_df['predicted_category']
    if category_eval == 'Hook Size': # encode for categorical for evaluation
        combined_list = list(true_values.copy())
        combined_list.extend(predicted_values)
        unique_values = sorted(set(combined_list))

        # Create a dictionary to map each unique value to a unique code
        encoding_dict = {value: idx for idx, value in enumerate(unique_values, start=1)}

        # Encode both lists
        encoded_list1 = [encoding_dict[value] for value in true_values.values]
        true_values = encoded_list1
        encoded_list2 = [encoding_dict[value] for value in predicted_values]
        predicted_values = encoded_list2

    accuracy = accuracy_score(true_values, predicted_values)
    print(f"Accuracy of predictions within ±{tolerance_level} units: {accuracy:.2f}")


    precision, recall, f1_score, _ = precision_recall_fscore_support(true_values, predicted_values, average='macro')
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)

    # Evaluate accuracy
    mse = mean_squared_error(true_values, predicted_values)
    print(f"mse: {mse:.2f}")

    rmse = np.sqrt(mse)
    print(f"rmse: {rmse:.2f}")

    mae = mean_absolute_error(true_values, predicted_values)
    print(f"mae: {mae:.2f}")

    r2 = sklearn.metrics.r2_score(true_values, predicted_values)
    print("R-squared:", r2)


def evaluate_model_attributes(attributes):
    print("Evaluate Hook Size")
    evaluate_model(data, attributes, category_eval="Hook Size")
    print("Evaluate Skill Level")
    evaluate_model(data, attributes, category_eval="Skill Level")
    print("Evaluate Yarn Weight")
    evaluate_model(data, attributes, category_eval="Yarn Weight")
    print("Evaluate Category")
    evaluate_model(data, attributes, category_eval="Category")
    print("Evaluate Stitches")
    evaluate_model(data, attributes, category_eval="Stitches")

def main():
    # Load and preprocess the data
    filename = "venv/crochet_patterns.csv"

    recommendation_attributes = [
        'Skill Level', 'Yarn Weight',
        'Hook Size', 'Category', 'Stitches'
    ]
    data, unique_stitches = pattern_csv_to_df(filename)
    attributes = recommendation_attributes.copy()
    attributes.extend(unique_stitches)
    evaluate_model_attributes(attributes)

    # Define Bayesian network structure
    model_structure = define_bayesian_network_structure()
    #plot_bayesian_network(model_structure)
    # Build and learn the Bayesian model
    bayesian_model, mappings = build_and_learn_bayesian_model(recommendation_data, model_structure)
    inference_engine = VariableElimination(bayesian_model)

    recommendation_attributes_orig = [
        'Skill Level', 'Yarn Weight',
        'Hook Size', 'Category', 'Stitches'  # , 'Yarn Brand', 'Color', , 'Stitches'
    ]
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
    rec = get_recommendation_for_attribute('Hook Size', input_data)
    print(rec)
   # reccommend_attribute_based_on_user_input(recommendation_attributes_orig, input_data, data,
    #                                         recommendation_attributes, inference_engine)


if __name__ == "__main__":
    #print("hi")
    main()