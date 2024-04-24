# Kaytin Matrangola
# PGM Final Project
# Recommender System For Crochet Patterns
# Requirements: Python 3.9

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
from tkmacosx import *
import tkinter as tk
from tkinter import ttk

class CrochetHelperApp:
    def __init__(self, master):

        self.master = master
        master.title("Crochet Helper")
        master.geometry("400x200")

        # Load data and build model
        self.data = self.pattern_csv_to_df("venv/crochet_patterns.csv")  # Adjust path as needed
        self.data['Category'] = self.data.apply(self.update_category, axis=1)
        self.model = self.build_bayesian_model(self.data)

        # Yarn Weight Input
        tk.Label(master, text="Yarn Weight:", font=("Helvetica", 12)).grid(row=0, sticky=tk.W)
        self.yarn_weight_var = tk.StringVar()
        self.yarn_weight_entry = ttk.Combobox(master, textvariable=self.yarn_weight_var)
        self.yarn_weight_entry['values'] = ('Light', 'Medium', 'Bulky')
        self.yarn_weight_entry.grid(row=0, column=1)


        # Stitches Input
        tk.Label(master, text="Stitches:", font=("Helvetica", 12)).grid(row=1, sticky=tk.W)
        self.stitches_var = tk.StringVar()
        self.stitches_entry = ttk.Entry(master, textvariable=self.stitches_var)
        self.stitches_entry.grid(row=1, column=1)

        # Button to Perform Inference
        self.infer_button = ttk.Button(master, text="Get Recommendations", command=self.perform_inference)
        self.infer_button.grid(row=2, column=0, columnspan=2)

        # Output Labels
        self.hook_size_label = tk.Label(master, text="Recommended Hook Size:", font=("Helvetica", 12))
        self.hook_size_label.grid(row=3, column=0, columnspan=2)

        self.skill_level_label = tk.Label(master, text="Skill Level for Your Stitches:", font=("Helvetica", 12))
        self.skill_level_label.grid(row=4, column=0, columnspan=2)
        # Define the function to update the category based on the title
    def update_category(self, row):
        title_lower = row['Title'].lower()
        if any(word in title_lower for word in ['blanket', 'throw']):
            return 'Blanket'
        elif any(word in title_lower for word in ['beanie', 'hat', 'ear']):
            return 'Headwear'
        elif any(word in title_lower for word in ['scarf', 'cowl', 'neck']):
            return 'Scarf'
        elif any(word in title_lower for word in ['shawl', 'vest', 'cardigan', 'sweater', 'hoodie', 'wrap', 'shrug', 'poncho', 'top', 'skirt', 'dress', 'afgan']):
            return 'Clothing'
        elif 'basket' in title_lower:
            return 'Basket'
        elif any(word in title_lower for word in ['coaster', 'placemat', 'cozy', 'ornament']):
            return 'Accessory'
        elif any(word in title_lower for word in ['amigurumi', 'penguin', 'octopus', 'jellyfish', 'owl', 'dog', 'lion', 'bear', 'monkey', 'luffy', 'bee', 'panda', 'gnome', 'santa', 'frankenstein', 'pumpkin']):
            return 'Amigurumi'
        else:
            return 'Stitch/Granny Square'

    def pattern_csv_to_df(self, filename):
        #TODO : where we can find numbers in the string make it int ie weight and skill level
        # Use pandas to read the CSV file
        df = pd.read_csv(filename)
        # Remove whitespace from values in all columns
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        #df['StitchesList'] = df['Stitches'].str.split(',')
        # Assuming 'data' is your DataFrame and 'Stitches' contains lists of stitches
        #df['Stitches_Combined'] = df['StitchesList'].apply(lambda stitches: '_'.join(sorted(set(stitches))))
        df = df.fillna(" ")
        return df

    def build_bayesian_model(self, data):
        # Define the structure of the Bayesian Network according to the specified relationships
        model_structure = [
            ('Yarn Weight', 'Hook Size'),  # Yarn Weight influences Hook Size
            ('Yarn Weight', 'Skill Level'),  # Yarn Weight influences Skill Level
            ('Hook Size', 'Skill Level'),  # Hook Size influences Skill Level
            #('Skill Level', 'Stitches'),  # Skill Level influences Stitches
            ('Yarn Name', 'Yarn Weight'),  # Yarn Type influences Yarn Weight
            #('Stitches', 'Category')  # Stitches influence Category
        ]
        # Initialize the model with the defined structure
        model = BayesianModel(model_structure)
        # Learn the parameters from the data using Maximum Likelihood Estimation
        # Use EM for parameter learning with missing data
        #data.drop('Stitches', axis=1, inplace=True)

        model.fit(data, estimator=EM)
        return model

    def perform_inference(self, bayesian_model, query_variable, evidence_dict):
        """
        General function to perform inference on a Bayesian model.
        """
        inference = VariableElimination(bayesian_model)
        result = inference.query(variables=[query_variable], evidence=evidence_dict)
        return result

    def query_hook_size_given_yarn_weight(self, bayesian_model, yarn_weight):
        result = CrochetHelperApp.perform_inference(bayesian_model, 'Hook Size', {'Yarn Weight': yarn_weight})
        return result

    def query_skill_level_given_stitches(self, bayesian_model, stitches_type):
        #result = perform_inference(bayesian_model, 'Skill Level', {'Stitches': stitches_type})
        #return result
        print("")

    def make_decision(self, model, user_preferences):
        """
        Makes a decision based on the Bayesian model and user preferences.
        """
        inference = VariableElimination(model)

        # Assume 'user_preferences' is a dict like {'Yarn Weight': 'Light', 'Skill Level': 'Beginner'}
        # Let's say the user wants a recommendation for 'Stitches'
        query_result = inference.query(variables=['Stitches'], evidence=user_preferences)

        # Making a decision based on the highest probability
        recommended_stitches = max(query_result.values, key=query_result.values.get)
        belief_propagation = BeliefPropagation(model)
        result = belief_propagation.query(variables=['Stitches'], evidence=user_preferences)
        return recommended_stitches


    def manual_inference(self, data, target, evidence):
        """
        Perform manual inference based on frequency counts from the dataset.
        """
        # Filter the data based on the evidence
        filtered_data = data
        for var, val in evidence.items():
            filtered_data = filtered_data[filtered_data[var] == val]

        # Compute the probability distribution of the target variable
        prob_dist = filtered_data[target].value_counts(normalize=True).to_dict()

        return prob_dist


if __name__ == '__main__':
    #scrape_and_store_patterns_easycrochet()
    #TODO: think about as inference problem (module 6)
    pattern_df = CrochetHelperApp.pattern_csv_to_df("crochet_patterns.csv")
    bayesian_model = CrochetHelperApp.build_bayesian_model(pattern_df)
    yarn_weight_example = 'Weight 4 - Medium'
    stitches_example = 'Single Crochet'

    hook_size_result = CrochetHelperApp.query_hook_size_given_yarn_weight(bayesian_model, yarn_weight_example)
    skill_level_result = CrochetHelperApp.query_skill_level_given_stitches(bayesian_model, stitches_example)
#
    print("Hook Size Given Yarn Weight:")
    print(hook_size_result)
    print("\nSkill Level Given Stitches:")
    print(skill_level_result)
    #root = tk.Tk()
    #print("bla")
    #app = CrochetHelperApp(root)
    #print("hehe")
    #root.mainloop()
    # IDEA Semi-supervised Learning: Allow users to provide feedback on model predictions (e.g., successful projects) to improve the model over time.