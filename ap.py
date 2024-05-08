from flask import Flask, render_template, request, redirect, url_for, session
from flask_session import Session  # Assuming Flask-Session is installed and configured
from crochet_recommender_hot import *
from crochet_recommender_2 import *
from crochet_recommender_3 import *

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    engine_type = request.form.get('recommendationEngine')
    session['engine_type'] = engine_type

    print(engine_type)

    form_data = {}
    for key in request.form.keys():
        if key not in ['action', 'recommendationEngine']:
            form_data[key] = request.form.getlist(key) if len(request.form.getlist(key)) > 1 else request.form[key]
    session['input_data'] = form_data
    if engine_type == 'bayesianNetwork':
        return redirect(url_for('bayesian_recommender'))
    elif engine_type == 'markovianNetwork':
        return redirect(url_for('markov_recommender'))
    elif engine_type == 'bayesianNetwork2':
        return redirect(url_for('bayes_recommender3'))
    return render_template('index.html')



@app.route('/bayes_recommender3', methods=['GET', 'POST'])
def bayes_recommender3():
    if request.method == 'GET':
        model, recommendation_attributes, inference_engine, recommendation_attributes, data = start3()
        session['inference_engine'] = inference_engine
        session['data'] = data
        session['recommendation_attributes'] = recommendation_attributes
        return render_template('recommender2_input.html')

    action = request.form.get('action')
    inference_engine = session['inference_engine']
    data = session['data']
    recommendation_attributes = session['recommendation_attributes']
    form_data = {}
    for key in request.form.keys():
        if key != 'action':
            # Retrieve list for checkboxes or single value otherwise
            form_data[key] = request.form.getlist(key) if len(request.form.getlist(key)) > 1 else request.form[key]
    print(form_data)

    input_data = process_input_data3(form_data, recommendation_attributes)
    session['input_data'] = input_data

    if action == 'Recommend Pattern':
        # Process the data and recommend a pattern
        show_loading_bar = True
        results = recommend_patterns_from_model(input_data, recommendation_attributes, inference_engine, data)
        print(results)
        return render_template('results.html', results=results.to_html(classes='dataframe', escape=False),
                               input_data=input_data)
    elif action == 'Recommend Attribute':
        # Redirect to attribute recommendation page with input data
        return redirect(url_for('recommend_attribute3', input_data=input_data))
    return render_template('recommender2_input.html')


@app.route('/markov_recommender', methods=['GET', 'POST'])
def markov_recommender():
    action = request.form.get('action')

    markov_model, mappings, recommendation_attributes, inference_engine = start_markov()

    form_data = {}
    for key in request.form.keys():
        if key != 'action':
            # Retrieve list for checkboxes or single value otherwise
            form_data[key] = request.form.getlist(key) if len(request.form.getlist(key)) > 1 else request.form[key]
    print(form_data)
    input_data = process_input_data3(form_data, recommendation_attributes)
    session['input_data'] = input_data
    session['recommendation_attributes'] = recommendation_attributes
    session['inference_engine'] = inference_engine
    if action == 'Recommend Pattern':
        # Process the data and recommend a pattern
        show_loading_bar = True
        results = recommend_patterns_from_markov(input_data, recommendation_attributes, inference_engine, data)
        print(results)
        return render_template('results.html', results=results.to_html(classes='dataframe', escape=False),
                               input_data=input_data)
    elif action == 'Recommend Attribute':
        # Redirect to attribute recommendation page with input data
        print(input_data)
        return redirect(url_for('recommend_attribute_markov', input_data=input_data))
    return render_template('recommender2_input.html')


@app.route('/recommend_attribute3', methods=['GET', 'POST'])
def recommend_attribute3():
    # Retrieve input_data from session to persist it across requests
    input_data = session.get('input_data', {})
    recommendation_attributes = session.get('recommendation_attributes', {})
    inference_engine = session.get('inference_engine')
    print(input_data)
    if request.method == 'POST':
        # User submits the attribute name to get a recommendation
        attribute_name = request.form['attribute_name']
        print(attribute_name)
        # Call the recommendation function using the attribute name and stored input data
        recommendation = get_recommendation_for_attribute3(attribute_name, input_data, recommendation_attributes,
                                                           inference_engine)
        print(recommendation)
        # Render a template to show the recommendation results
        return render_template('attribute_result.html', recommendation=recommendation[attribute_name],
                               attribute_name=attribute_name)
    # On GET request, just show the form to enter the attribute name
    return render_template('select_attribute2.html')


@app.route('/bayesian_recommender', methods=['POST', 'GET'])
def bayesian_recommender():
    if request.method == 'GET':
        # Initialize the page for user input
        return render_template('bayes_recommender_input.html')

    action = request.form.get('action', 'default_action')
    print("Action received:", action)

    # Initialize the recommender system with necessary data
    recommendation_attributes, recommendation_attributes_out, recommendation_attributes_orig, inference_engine, mappings, data, unique_stitches = start1()
    session['mappings'] = mappings
    session['inference_engine'] = inference_engine
    session['recommendation_attributes'] = recommendation_attributes
    session['unique_stitches'] = unique_stitches
    session['recommendation_attributes_orig'] = recommendation_attributes_orig

    # Collect form data, excluding 'action'
    form_data = {key: request.form.getlist(key) if len(request.form.getlist(key)) > 1 else request.form[key] for key in request.form if key != 'action'}
    print("Form data:", form_data)

    # Process input data
    input_data = process_input_data(form_data, data)
    session['input_data'] = input_data

    if action == 'Recommend Pattern':
        # Process the data and recommend a pattern
        results = recommend_patterns_from_bayes(input_data, inference_engine, recommendation_attributes_orig, mappings, data, recommendation_attributes, recommendation_attributes_out)
        print("Recommendation results:", results)
        return render_template('results.html', results=results.to_html(classes='dataframe', escape=False), input_data=input_data)

    elif action == 'Recommend Attribute':
        # Redirect to attribute recommendation page with stored input data
        return redirect(url_for('recommend_attribute'))

    # Default response if no recognized action found
    return render_template('bayes_recommender_input.html')


@app.route('/recommend_attribute_markov', methods=['GET', 'POST'])
def recommend_attribute_markov():
    # Retrieve input_data from session to persist it across requests
    input_data = session.get('input_data', {})
    recommendation_attributes = session.get('recommendation_attributes', {})
    inference_engine = session.get('inference_engine')
    print(input_data)
    if request.method == 'POST':
        # User submits the attribute name to get a recommendation
        attribute_name = request.form['attribute_name']
        print(attribute_name)
        # Call the recommendation function using the attribute name and stored input data
        recommendation = get_recommendation_for_attribute_markov(attribute_name, input_data, recommendation_attributes,
                                                                 inference_engine)
        print(recommendation)
        # Render a template to show the recommendation results
        return render_template('attribute_result.html', recommendation=recommendation[attribute_name],
                               attribute_name=attribute_name)
    # On GET request, just show the form to enter the attribute name
    return render_template('select_attribute_markov.html')


@app.route('/recommend_attribute', methods=['GET', 'POST'])
def recommend_attribute():
    # Retrieve input_data from session to persist it across requests
    recommendation_attributes_orig = session.get('recommendation_attributes_orig', {})
    mappings = session.get('mappings', {})
    input_data = session.get('input_data', {})
    recommendation_attributes = session.get('recommendation_attributes', {})
    inference_engine = session.get('inference_engine')
    unique_stitches = session.get('unique_stitches', [])
    print(input_data)
    if request.method == 'POST':
        # User submits the attribute name to get a recommendation
        attribute_name = request.form['attribute_name']
        print(attribute_name)
        # Call the recommendation function using the attribute name and stored input data
        recommendation = get_recommendation_for_attribute(attribute_name, input_data,
                                                          recommendation_attributes_orig, mappings,
                                                          recommendation_attributes, unique_stitches, inference_engine)

        # Render a template to show the recommendation results
        return render_template('attribute_result.html', recommendation=recommendation,
                               attribute_name=attribute_name)
    # On GET request, just show the form to enter the attribute name
    return render_template('select_attribute.html')


@app.route('/results')
def results():
    results = session.get('results', 'No results found')
    return render_template('results.html', results=results)


if __name__ == "__main__":
    app.run(debug=True)
