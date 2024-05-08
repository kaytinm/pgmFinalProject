from flask import Flask, render_template, request, redirect, url_for, session
from flask_session import Session  # Assuming Flask-Session is installed and configured
from crochet_recommender_easycrochet import *
from crochet_recommender_crochet_markov import *
from crochet_recommender_crochet_bayes import *
app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    engine_type = request.form['recommendationEngine']
    form_data = {}
    for key in request.form.keys():
        if key not in ['action', 'recommendationEngine']:
            form_data[key] = request.form.getlist(key) if len(request.form.getlist(key)) > 1 else request.form[key]
    session['input_data'] = form_data
    if engine_type == 'BayesianNetwork':
        return redirect(url_for('bayes_recommener_index'))
    elif engine_type == 'MarkovianNetwork':
        return redirect(url_for('markov_recommender_index'))
    elif engine_type == 'BayesianNetwork2':
        return redirect(url_for('bayes_recommender2_index'))
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def bayes_recommender3():
    action = request.form['action']
    form_data = {}
    for key in request.form.keys():
        if key != 'action':
            # Retrieve list for checkboxes or single value otherwise
            form_data[key] = request.form.getlist(key) if len(request.form.getlist(key)) > 1 else request.form[key]
    print(form_data)
    input_data = process_input_data_markov(form_data)
    session['input_data'] = input_data
    if action == 'Recommend Pattern':
        # Process the data and recommend a pattern
        show_loading_bar = True
        results = recommend_patterns_from_markov(input_data)
        print(results)
        return render_template('results.html', results=results.to_html(classes='dataframe', escape=False), input_data=input_data)
    elif action == 'Recommend Attribute':
        # Redirect to attribute recommendation page with input data
        print(input_data)
        return redirect(url_for('recommend_attribute3', input_data=input_data))

def markov_recommender():
    action = request.form['action']
    form_data = {}
    for key in request.form.keys():
        if key != 'action':
            # Retrieve list for checkboxes or single value otherwise
            form_data[key] = request.form.getlist(key) if len(request.form.getlist(key)) > 1 else request.form[key]
    print(form_data)
    input_data = process_input_data3(form_data)
    session['input_data'] = input_data
    if action == 'Recommend Pattern':
        # Process the data and recommend a pattern
        show_loading_bar = True
        results = recommend_patterns_from_model(input_data)
        print(results)
        return render_template('results.html', results=results.to_html(classes='dataframe', escape=False), input_data=input_data)
    elif action == 'Recommend Attribute':
        # Redirect to attribute recommendation page with input data
        print(input_data)
        return redirect(url_for('recommend_attribute_markov', input_data=input_data))


@app.route('/recommend_attribute', methods=['GET', 'POST'])
def recommend_attribute3():
    # Retrieve input_data from session to persist it across requests
    input_data = session.get('input_data', {})
    print(input_data)
    if request.method == 'POST':
        # User submits the attribute name to get a recommendation
        attribute_name = request.form['attribute_name']
        print(attribute_name)
        # Call the recommendation function using the attribute name and stored input data
        recommendation = get_recommendation_for_attribute3(attribute_name, input_data)
        print(recommendation)
        # Render a template to show the recommendation results
        return render_template('attribute_result.html', recommendation=recommendation[attribute_name], attribute_name=attribute_name)
    # On GET request, just show the form to enter the attribute name
    return render_template('select_attribute3.html')


@app.route('/submit', methods=['POST'])
def bayesian_recommender():
    action = request.form['action']
    form_data = {}
    for key in request.form.keys():
        if key != 'action':
            # Retrieve list for checkboxes or single value otherwise
            form_data[key] = request.form.getlist(key) if len(request.form.getlist(key)) > 1 else request.form[key]
    print(form_data)
    input_data = process_input_data(form_data)
    session['input_data'] = input_data
    if action == 'Recommend Pattern':
        # Process the data and recommend a pattern
        show_loading_bar = True
        results = recommend_patterns_from_bayes(input_data)
        print(results)
        return render_template('results.html', results=results.to_html(classes='dataframe', escape=False), input_data=input_data)
    elif action == 'Recommend Attribute':
        # Redirect to attribute recommendation page with input data
        print(input_data)
        return redirect(url_for('recommend_attribute', input_data=input_data))

@app.route('/recommend_attribute', methods=['GET', 'POST'])
def recommend_attribute_markov():
    # Retrieve input_data from session to persist it across requests
    input_data = session.get('input_data', {})
    print(input_data)
    if request.method == 'POST':
        # User submits the attribute name to get a recommendation
        attribute_name = request.form['attribute_name']
        print(attribute_name)
        # Call the recommendation function using the attribute name and stored input data
        recommendation = get_recommendation_for_attribute_markov(attribute_name, input_data)
        print(recommendation)
        # Render a template to show the recommendation results
        return render_template('attribute_result.html', recommendation=recommendation[attribute_name], attribute_name=attribute_name)
    # On GET request, just show the form to enter the attribute name
    return render_template('select_attribute_markov.html')


@app.route('/recommend_attribute', methods=['GET', 'POST'])
def recommend_attribute():
    # Retrieve input_data from session to persist it across requests
    input_data = session.get('input_data', {})
    print(input_data)
    if request.method == 'POST':
        # User submits the attribute name to get a recommendation
        attribute_name = request.form['attribute_name']
        print(attribute_name)
        # Call the recommendation function using the attribute name and stored input data
        recommendation = get_recommendation_for_attribute(attribute_name, input_data)
        print(recommendation)
        # Render a template to show the recommendation results
        return render_template('attribute_result.html', recommendation=recommendation[attribute_name], attribute_name=attribute_name)
    # On GET request, just show the form to enter the attribute name
    return render_template('select_attribute.html')


@app.route('/results')
def results():
    results = session.get('results', 'No results found')
    return render_template('results.html', results=results)

if __name__ == "__main__":
    app.run(debug=True)
