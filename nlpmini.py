import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import cohere
import streamlit as st

# Set up Cohere API key
cohere_api_key = "UeMwoAVVSFoRqjgeftzueODabegFZMDn5YwrSJiB"  # Replace with your Cohere API key
co = cohere.Client(cohere_api_key)

# Load dataset
df = pd.read_csv(r"C:\Users\SMITHA N\Downloads\NLP project\datanew (2).csv")
df.columns = df.columns.str.strip()
df.rename(columns={"text": "comment_text"}, inplace=True)

# Clean text
def clean_text(text):
    return text.lower()

df['clean_comment'] = df['comment_text'].astype(str).apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['clean_comment'])
y = df['toxic_label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Motivational message using Cohere's Chat API
def get_motivation_with_cohere(comment):
    response = co.chat(
        message=f"Generate a motivational response to this toxic comment: '{comment}'. Respond with kindness and encouragement.",
        temperature=0.8
    )
    return response.text.strip()

# Predict and respond
def predict_toxicity_with_motivation(comment):
    clean_comment = clean_text(comment)
    comment_vector = vectorizer.transform([clean_comment])
    prediction = model.predict(comment_vector)[0]

    if prediction == 1:
        motivation = get_motivation_with_cohere(comment)
        return f"⚠️ Toxic Comment Detected!\n\n**Motivational Message:**\n{motivation}"
    else:
        return "✅ This comment is non-toxic. Keep spreading positivity!"

# Streamlit UI
st.title("🧠 Toxicity Comment Classifier and Motivational Generator")
st.write("Enter a comment to check if it's toxic. If it is, you'll receive a motivational message!")

user_comment = st.text_area("Enter your comment here:")

if st.button("Analyze"):
    if user_comment.strip():
        result = predict_toxicity_with_motivation(user_comment)
        st.write(result)
    else:
        st.warning("Please enter a comment to analyze.")

