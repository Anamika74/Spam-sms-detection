import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()

# 1. Paste your exact text transformation logic here
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

# 2. Load the saved vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# 3. Streamlit UI Layout
st.title(" SMS Spam Detection System")
st.write("Enter an SMS message below to check if it's Spam or not.")

input_sms = st.text_area("Type your message here...", height=150)

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a valid message!")
    else:
        # Preprocess the input string
        transformed_sms = transform_text(input_sms)
        
        # Vectorize using the loaded TF-IDF vectorizer
        vector_input = tfidf.transform([transformed_sms])
        
        # Predict using the loaded Random Forest model
        result = model.predict(vector_input)[0]
        
        # Display the output
        if result == 1:
            st.error(" Warning: This is a SPAM message!")
        else:
            st.success(" Safe: This is not spam.")

if st.button("About"):
    st.write("This SMS Spam Detection System uses a Random Forest Classifier trained on a dataset of SMS messages to classify incoming messages as either 'Spam' or 'Not Spam'.")
    st.write("The model was trained using TF-IDF vectorization and achieved high precision and accuracy.")
    
if st.button("Explanation"):
    st.write("How this model predicts spam messages:")
    st.write('''1. The Text Basis (TF-IDF Features)
Your text is converted into a row of 6,708 numbers based on a mathematical balancing act:

Term Frequency (TF): Words get a higher score if they appear multiple times within a single message (e.g., repeating "FREE").

Inverse Document Frequency (IDF): Words get a lower score if they are common everywhere (like "go" or "ur"). They get a very high score if they are rare and informative (like "winner", "claim", or "cash").

2. The Decision Basis (Random Forest Ensemble)
Once the text is converted into numerical scores, it is processed by an ensemble of 50 independent decision trees:

Tree Routing: Each tree looks at a random subset of words and checks if their scores cross specific thresholds learned during training (e.g., Is the score for "claim" > 0.35?).

Majority Voting: Everyday conversational words route decisions toward Ham (0), while promotional words route them toward Spam (1). Every tree casts a single vote, and the majority winner becomes the final prediction.''')