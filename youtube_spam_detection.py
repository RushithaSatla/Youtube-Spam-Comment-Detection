import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

# List of file paths for the input CSV files
file_paths = [
    '/content/Youtube01-Psy.csv',
    '/content/Youtube02-KatyPerry.csv',
    '/content/Youtube03-LMFAO.csv',
    '/content/Youtube04-Eminem.csv',
    '/content/Youtube05-Shakira.csv'
]

def load_data(file_paths):
    """Load and concatenate data from a list of file paths."""
    data = pd.concat([pd.read_csv(file) for file in file_paths], ignore_index=True)
    return data

def preprocess_text(text_series):
    """Basic text preprocessing: lowercasing, removing special characters."""
    return text_series.str.lower().str.replace(r'[^\w\s]', '', regex=True)

def visualize_class_distribution(y):
    """Visualize the class distribution in the dataset."""
    unique, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(6, 4))
    plt.bar(unique, counts, color=['skyblue', 'salmon'])
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution in the Dataset')
    plt.xticks(unique)
    plt.show()

def preprocess_data(data):
    """Split the data into features and labels, then into training and testing sets."""
    data['CONTENT'] = preprocess_text(data['CONTENT'])
    X = data['CONTENT']
    y = data['CLASS']
    visualize_class_distribution(y)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    """Vectorize the text data and train a Naive Bayes classifier."""
    vectorizer = CountVectorizer()
    X_train_counts = vectorizer.fit_transform(X_train)
    classifier = MultinomialNB()
    classifier.fit(X_train_counts, y_train)
    return vectorizer, classifier

def evaluate_model(classifier, vectorizer, X_test, y_test):
    """Evaluate the classifier on the test set and print accuracy, confusion matrix, and classification report."""
    X_test_counts = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_counts)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_test)))
    plt.xticks(tick_marks, np.unique(y_test))
    plt.yticks(tick_marks, np.unique(y_test))
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

    # Classification report
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def cross_validation_score(classifier, X, y):
    """Perform k-fold cross-validation to evaluate model reliability."""
    vectorizer = CountVectorizer()
    X_counts = vectorizer.fit_transform(X)
    cv_scores = cross_val_score(classifier, X_counts, y, cv=5)
    print("Cross-Validation Scores:", cv_scores)
    print("Mean CV Score:", np.mean(cv_scores))

def predict_comment(classifier, vectorizer, comment):
    """Predict the label of a single comment."""
    comment_counts = vectorizer.transform([comment])
    prediction = classifier.predict(comment_counts)
    return prediction[0]

def batch_predict_comments(classifier, vectorizer, comments):
    """Predict labels for a batch of comments."""
    comments_counts = vectorizer.transform(comments)
    predictions = classifier.predict(comments_counts)
    for comment, label in zip(comments, predictions):
        print(f"Comment: {comment[:30]}... -> Predicted label: {label}")

# New Function to Separate and Remove Spam Comments
def remove_spam_comments(input_file, classifier, vectorizer, output_file='non_spam_comments.csv'):
    """Load comments from a CSV file, predict spam or not, and save only non-spam comments to a new file."""
    # Load comments from input file
    comments_df = pd.read_csv(input_file)

    # Ensure 'CONTENT' column exists for comments
    if 'CONTENT' not in comments_df.columns:
        raise ValueError("The input CSV file must contain a 'CONTENT' column with comments.")

    # Preprocess and predict spam labels
    comments_df['CONTENT'] = preprocess_text(comments_df['CONTENT'])
    comment_counts = vectorizer.transform(comments_df['CONTENT'])
    predictions = classifier.predict(comment_counts)

    # Separate spam and non-spam comments
    non_spam_comments = comments_df[predictions == 0]  # Assuming 0 is the label for non-spam
    spam_comments = comments_df[predictions == 1]  # Assuming 1 is the label for spam

    # Save non-spam comments to a new CSV file
    non_spam_comments.to_csv(output_file, index=False)
    print(f"Non-spam comments have been saved to {output_file}.")
    print(f"Total comments processed: {len(comments_df)}")
    print(f"Non-spam comments: {len(non_spam_comments)}")
    print(f"Spam comments removed: {len(spam_comments)}")

# Load data
data = load_data(file_paths)

# Split data
X_train, X_test, y_train, y_test = preprocess_data(data)

# Train model
vectorizer, classifier = train_model(X_train, y_train)

# Evaluate model
evaluate_model(classifier, vectorizer, X_test, y_test)

# Cross-validation
cross_validation_score(classifier, data['CONTENT'], data['CLASS'])

# Take user input for a single comment prediction
try:
    new_comment = input("Enter a comment to classify: ")
    predicted_label = predict_comment(classifier, vectorizer, new_comment)
    print("Predicted label:", predicted_label)
except Exception as e:
    print("Error with input or prediction:", e)

# Example batch prediction
batch_comments = [
    "This video is amazing!",
    "I hated every second of this.",
    "Check out my channel for more videos.",
    "Great song and performance!"
]
batch_predict_comments(classifier, vectorizer, batch_comments)

# Remove spam comments from a CSV file
remove_spam_comments('Youtube01-Psy.csv', classifier, vectorizer)

