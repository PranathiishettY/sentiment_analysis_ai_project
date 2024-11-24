import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt
import joblib

# Ensure stopwords are downloaded
nltk.download('stopwords')

# Sample dataset with additional examples
data = {
    'Review': [
        "I love this product, it's amazing!",
        "Terrible, broke after one use",
        "Good quality, but too expensive",
        "Excellent! Would recommend to everyone.",
        "Worst purchase ever, very disappointed",
        "good",
        "This is a good product",
        "good and satisfactory experience",
        "It broke after one use, terrible!",
        "Awful experience, will not buy again",
        "Fantastic and works perfectly",
        "Poor build quality, not worth the money","Amazing product, highly recommend!","good"
    ],
    'Sentiment': [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0,1,1]
}

# Convert the dictionary into a DataFrame
df = pd.DataFrame(data)
df.to_excel('product_reviews.xlsx', index=False)

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    # Keep "good", "bad", "terrible", etc., even if they're stopwords
    critical_words = {'good', 'bad', 'terrible', 'broke', 'excellent', 'amazing', 'worst', 'awful', 'poor'}
    text = ' '.join([
        word for word in text.split()
        if word not in stopwords.words('english') or word in critical_words
    ])
    return text

# Apply preprocessing to the review column
df['Cleaned_Review'] = df['Review'].apply(preprocess_text)

# Define features (X) and target (y)
X = df['Cleaned_Review']
y = df['Sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))  # Include bigrams

# Fit the vectorizer to the training data and transform both the train and test data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize the Naive Bayes classifier
model = MultinomialNB()

# Train the model
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Generate a classification report
print(classification_report(y_test, y_pred))

# Visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(np.arange(2), ['Negative', 'Positive'])
plt.yticks(np.arange(2), ['Negative', 'Positive'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()

# Function to predict sentiment
def predict_sentiment(review):
    review = preprocess_text(review)
    review_tfidf = vectorizer.transform([review])
    sentiment = model.predict(review_tfidf)
    return "Positive" if sentiment == 1 else "Negative"

# Test the model with new reviews
new_review = "The product is fantastic, I love it!"
print(f'Sentiment: {predict_sentiment(new_review)}')  # Expected: Positive

new_review_2 = "It broke after one use, terrible!"
print(f'Sentiment: {predict_sentiment(new_review_2)}')  # Expected: Negative

new_review_3 = "good"
print(f'Sentiment: {predict_sentiment(new_review_3)}')  # Expected: Positive

# Save the model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
