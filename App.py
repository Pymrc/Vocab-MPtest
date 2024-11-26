from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from transformers import pipeline
import pandas as pd

# Initialiser l'application Flask
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///vocab.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Modèle pour le vocabulaire
class Word(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    word = db.Column(db.String(50), nullable=False)
    definition = db.Column(db.String(200), nullable=False)
    example = db.Column(db.String(200), nullable=True)
    theme = db.Column(db.String(50), nullable=False)

# Créer la base de données
with app.app_context():
    db.create_all()

# Modèle de génération de phrases (Hugging Face)
generator = pipeline("text-generation", model="gpt2")

# Page d'accueil : Tableau de bord
@app.route('/')
def index():
    words = Word.query.all()
    return render_template('index.html', words=words)

# Ajouter un mot
@app.route('/add', methods=['POST'])
def add_word():
    word = request.form['word']
    definition = request.form['definition']
    theme = request.form['theme']
    example = generator(f"Example of the word {word} in a sentence:", max_length=30)[0]['generated_text']
    new_word = Word(word=word, definition=definition, example=example, theme=theme)
    db.session.add(new_word)
    db.session.commit()
    return redirect(url_for('index'))

# Quiz : Tester ses connaissances
@app.route('/quiz')
def quiz():
    word = Word.query.order_by(db.func.random()).first()
    return render_template('quiz.html', word=word)

@app.route('/quiz', methods=['POST'])
def check_quiz():
    word_id = request.form['word_id']
    answer = request.form['answer']
    word = Word.query.get(word_id)
    result = "Correct!" if answer.lower() in word.definition.lower() else "Incorrect."
    return render_template('quiz_result.html', result=result, word=word)

# Importer des mots via un fichier CSV
@app.route('/import', methods=['POST'])
def import_words():
    file = request.files['file']
    df = pd.read_csv(file)
    for _, row in df.iterrows():
        new_word = Word(word=row['word'], definition=row['definition'], theme=row['theme'], example=row['example'])
        db.session.add(new_word)
    db.session.commit()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
