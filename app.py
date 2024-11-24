from flask import Flask, request, render_template
import joblib
import string
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Flask app initialization
app = Flask(__name__)

# Load pre-trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    critical_words = {'good', 'bad', 'terrible', 'broke', 'excellent', 'amazing', 'worst', 'awful', 'poor'}
    text = ' '.join([
        word for word in text.split()
        if word not in stopwords.words('english') or word in critical_words
    ])
    return text

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input
        review = request.form['review']
        
        # Preprocess the review
        processed_review = preprocess_text(review)
        
        # Transform the review using the vectorizer
        review_tfidf = vectorizer.transform([processed_review])
        
        # Predict sentiment
        sentiment = model.predict(review_tfidf)
        
        # Map prediction to sentiment
        sentiment_label = "Positive" if sentiment == 1 else "Negative"
        
        # Render result
        return render_template('index.html', prediction_text=f'The sentiment is: {sentiment_label}')

if __name__ == "__main__":
    app.run(debug=True)
