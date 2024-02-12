from flask import Flask, request, jsonify, render_template,flash, redirect, url_for,send_from_directory, send_file,session
from spellchecker import SpellChecker
from pymongo import MongoClient
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re
import os
import mimetypes
from urllib.parse import unquote
import gridfs
import io
from flask_pymongo import PyMongo
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
import logging
from passlib.hash import sha256_crypt

# Configure logging to show warnings and errors
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)



app = Flask(__name__)
fb = MongoClient().test
fs = gridfs.GridFS(fb)

app.jinja_env.filters['mimetype_is_text'] = lambda filename: mimetypes.guess_type(filename)[0].startswith('text/')

spell = SpellChecker()
client = MongoClient('mongodb://localhost:27017/')
db = client['questions_db']
collection = db['questions_collection']
c2 =db['user_credentials']
offensive_words = ["18+","romance","dating","fuck","i love you"]

# Define keywords for each category
categories_keywords = {
    'python': {'python', 'django', 'flask', 'numpy', 'pandas'},
    'java': {'java', 'spring', 'hibernate', 'android','oops'},
    'language query': {'language', 'query', 'programming'},
    'database': {'database', 'sql', 'mysql', 'postgresql','transactions','acid','transaction','storage','distributed systems'},
    'database query': {'database', 'query', 'sql'},
    'bigdata': {'bigdata', 'hadoop', 'spark', 'hive'},
    'machine learning': {'machine learning', 'ml', 'scikit-learn', 'tensorflow','supervised','unsupervised','algorithm','cross validation','dimensions','dimensional reduction'},
    'data science': {'data science', 'data analytics', 'data visualization','probability'},
    'generative ai': {'generative ai', 'gan', 'neural network'},
    'artificial intelligence': {'artificial intelligence', 'ai','generative ai','robotics'},
    'deep learning': {'deep learning', 'dl', 'convolutional neural network'},
    'natural language processing' : {'nlp','semantic','parsing','sentence','tokenizer','tokenization','stemming','ner','entity recognition' }
}


'''def preprocess_text(text):
    
    # Tokenize the text and correct spelling errors
    tokens = text.split()
    corrected_tokens = [spell.correction(token) for token in tokens]
    return ' '.join(corrected_tokens)'''

def preprocess_text(text):

    # Separate words from special characters with regular expressions
    tokens = re.findall(r"\w+|[^\w\s]", text)  # Matches words and non-word characters

    corrected_tokens = []
    for token in tokens:
        if re.match(r"\w+", token):  # Check if token is a word
            try:
                corrected_token = spell.correction(token)
                corrected_tokens.append(corrected_token)
            except:  # Handle potential errors during correction
                print(f"Error correcting token: {token}")
                corrected_tokens.append(token)
        else:  # Special character or symbol: keep as-is
            corrected_tokens.append(token)

    return ' '.join(corrected_tokens)


def categorize_question(question_text):
    for category, keywords in categories_keywords.items():
        if any(keyword in question_text.lower() for keyword in keywords):
            return category
    return 'Uncategorized'


def contains_offensive_words(text):
    # Check if any offensive word is present in the text
    for word in offensive_words:
        if word.lower() in text.lower():
            return True
    return False

@app.route('/')
def index():
    return render_template('landingpage.html')

@app.route('/about')
def about():
    return render_template('ABOUT.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/mainpage')
def mainpage():
    return render_template('index.html')

# Route for the sign-up page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if username already exists
        if c2.find_one({'username': username}):
            flash('Username already exists', 'error')
            return redirect(url_for('signup'))

        # Hash the password
        hashed_password = sha256_crypt.hash(password)

        # Insert new user into the database
        c2.insert_one({'username': username, 'password': hashed_password})
        flash('You have successfully signed up', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

# Route for the login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if username exists
        user_data = c2.find_one({'username': username})
        if user_data:
            # Verify password
            if sha256_crypt.verify(password, user_data['password']):
                session['logged_in'] = True
                session['username'] = username
                flash('You are now logged in', 'success')
                return redirect(url_for('mainpage'))
            else:
                flash('Invalid password', 'error')
                return redirect(url_for('login'))
        else:
            flash('Username not found', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')

# Route for logout
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))

@app.route('/store_question', methods=['POST'])
def store_question():
    try:
        # Get question text
        question_text = request.form.get('question_text')
        
        # Check for offensive words
        if contains_offensive_words(question_text):
            flash("We're not allowed to allow you to spoil our community.", "error")
            return redirect(url_for('mainpage'))

        # Get files
        files = request.files.getlist('files')

        
        category = categorize_question(question_text)
        
        # Store files in database
        encoded_files = []
        for file in files:
            # Convert file to base64 encoding
            encoded_file = base64.b64encode(file.read()).decode('utf-8')
            encoded_files.append(encoded_file)

        # Store question and files in MongoDB
        question_data = {
            'question_text': question_text,
             'category': category,
            'files': encoded_files
        }
        collection.insert_one(question_data)

        flash('Question stored successfully', 'success')
        return redirect(url_for('mainpage'))

    except Exception as e:
        return jsonify({'error': str(e)}), 500



'''@app.route('/view_categories')
def view_categories():
    # Get distinct categories from the database
    categories = collection.distinct('category')

    # Create a dictionary to store questions for each category
    categorized_questions = {}
    for category in categories:
        questions = list(collection.find({'category': category}))
        categorized_questions[category] = questions

    return render_template('categories.html', categorized_questions=categorized_questions)'''


@app.route('/view_categories')
def view_categories():
    # Get distinct categories from the database
    categories = collection.distinct('category')

    # Create a dictionary to store questions for each category
    categorized_questions = {}

    # Categorize questions based on keywords
    for question in collection.find():
        category = None
        for key, keywords in categories_keywords.items():
            if any(keyword in question['question_text'].lower() for keyword in keywords):
                category = key
                break
        if category:
            if category not in categorized_questions:
                categorized_questions[category] = []
            categorized_questions[category].append(question)



    return render_template('categories.html', categorized_questions=categorized_questions)


@app.route('/question_detail/<question_text>')
def question_detail(question_text):
    # Retrieve the question based on category and question text
    question = collection.find_one({'question_text': question_text})

    if question:
        return render_template('question_detail.html', question=question)
    else:
        return jsonify({'error': 'Question not found'}), 404

@app.route('/image/<image_id>')
def get_image(image_id):
    file = fs.get(image_id)
    return send_file(io.BytesIO(file.read()), mimetype='image/jpeg')

@app.route('/recent_questions', methods=['GET'])
def get_recent_questions():
    try:
        # Fetch recent questions from MongoDB
        recent_questions = collection.find().limit(15)  # Fetch the recent 5 questions

        # Convert MongoDB results to a list of dictionaries
        questions_list = [{"question_text": question["question_text"]} for question in recent_questions]

        # Return the questions as JSON
        return jsonify(questions_list)

    except ServerSelectionTimeoutError as server_error:
        return jsonify({"error": "Could not connect to the database"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/search_questions/<keyword>', methods=['GET'])
def search_questions(keyword):
    try:
        # Fetch questions based on the keyword from MongoDB
        # Validate the keyword
        '''if not keyword.isalnum():
            return jsonify({"error": "Invalid keyword. Please use alphanumeric characters only."}), 400'''
        search_results = collection.find(
            {'question_text': {'$regex': keyword, '$options' :'i'}}
        )

        # Convert MongoDB results to a list of dictionaries
        results_list = [{"question_text": question["question_text"]} for question in search_results]

        # Return the search results as JSON
        return jsonify(results_list)

    except ServerSelectionTimeoutError as server_error:
        return jsonify({"error": "Could not connect to the database"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/save_answer/<question_id>', methods=['POST'])
def save_answer(question_id):
    
    data = request.json

    # Validate the request data
    if not data or 'answer_text' not in data:
        return jsonify({"error": "Invalid request data. Please provide an answer text."}), 400

    # Save the answer to the database
    answer_data = {
        "question_id": question_id,
        "answer_text": data['answer_text']
    }
    try:
        db.answers.insert_one(answer_data)
        return jsonify({"success": "Answer saved successfully."})

    except Exception as e:
        # Log the error for debugging
        app.logger.error("An error occurred while saving an answer: %s", e)
        # Return a generic error response
        return jsonify({"error": "An error occurred while saving the answer"}), 500

@app.route('/vote_up/<answer_id>', methods=['POST'])
def vote_up(answer_id):
    
    # Validate the request
    if not request.json or 'user_id' not in request.json:
        return jsonify({"error": "Invalid request data. Please provide a user ID."}), 400

    # Update the vote count in the database
    try:
        db.answers.update_one(
            {"_id": answer_id},
            {"$inc": {"vote_count": 1}}
        )
        return jsonify({"success": "Vote up recorded successfully."})

    except Exception as e:
        # Log the error for debugging
        app.logger.error("An error occurred while voting up an answer: %s", e)
        # Return a generic error response
        return jsonify({"error": "An error occurred while voting"}), 500

@app.route('/vote_down/<answer_id>', methods=['POST'])
def vote_down(answer_id):
    
    # Validate the request
    if not request.json or 'user_id' not in request.json:
        return jsonify({"error": "Invalid request data. Please provide a user ID."}), 400

    # Update the vote count in the database
    try:
        db.answers.update_one(
            {"_id": answer_id},
            {"$inc": {"vote_count": -1}}
        )
        return jsonify({"success": "Vote down recorded successfully."})

    except Exception as e:
        # Log the error for debugging
        app.logger.error("An error occurred while voting down an answer: %s", e)
        # Return a generic error response
        return jsonify({"error": "An error occurred while voting"}), 500

@app.route('/vote_count/<answer_id>', methods=['GET'])
def vote_count(answer_id):
    
    # Fetch the vote count from the database
    answer = db.answers.find_one({"_id": answer_id})

    if not answer:
        return jsonify({"error": "Answer not found."}), 404

    return jsonify({"vote_count": answer["vote_count"]})


if __name__ == '__main__':
    app.secret_key = 'super_secret_key'
    app.run(debug=True)
    
