import streamlit as s
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

nltk.download('stopwords')
ps = PorterStemmer()

v = pickle.load(open('vectorizer.pkl','rb'))
mnb_model = pickle.load(open('model.pkl','rb'))

def change_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    l = []
    for i in text:
        if i.isalnum():
            l.append(i)

    text = l[:]
    l.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            l.append(i)

    text = l[:]
    l.clear()

    for i in text:
        l.append(ps.stem(i))

    return " ".join(l)


s.title("Email Spam Classifier")
input_msg = s.text_input("Enter the email message")

if s.button('Predict'):
    changed_msg = change_text(input_msg)
    to_be_predicted_msg = v.transform([changed_msg])
    prediction = mnb_model.predict(to_be_predicted_msg)[0]
    if prediction == 1:
        s.header("It's a spam message")
    else:
        s.header("Not a spam message")