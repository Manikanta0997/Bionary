import streamlit as st
import pickle
import pandas as pd
import sklearn
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

def cleaning_fsa(data):

 import re
 #1. Remove Puncs
 # \w typically matches [A-Za-z0-9_]
 text = re.sub('[^\w\s]','', data)

 #2. Tokenize
 text_tokens = word_tokenize(text.lower())

 #3. Remove numbers
 tokens_without_punc = [w for w in text_tokens if w.isalpha()]

 #4. Removing Stopwords
 stop_words = stopwords.words('english')
 tokens_without_sw = [t for t in tokens_without_punc if t not in stop_words]


 #joining
 return " ".join(tokens_without_sw)

pipe = pickle.load(open('Bionary.pkl','rb'))
function = pickle.load(open('Function.pkl', 'rb'))
vect = pickle.load(open('vect.pkl', 'rb'))
ExerciseAngina1 = ['N','Y']
ST_Slope1 = ['Up', 'Flat', 'Down']
Sex1 = ['M', 'F']
ChestPainType1 = ['ATA', 'NAP', 'ASY', 'TA']
RestingECG1 = ['Normal', 'ST', 'LVH']
st.set_page_config(
    page_title="Bionary ML-SUBDEP",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title('A sentiment analysis model')


df = st.text_input(label = 'Enter the Review')
    

if st.button('Predict'):
    df_pred = pd.DataFrame({'Reviews':[df] })
    df_pred["Reviews1"]=df_pred["Reviews"].apply(function)
    Review=vect.transform(df_pred["Reviews1"])
    
    result = pipe.predict(Review)
	
    if int(result[0]) == 0:
        result = "Not Recommended"
    else:
        result = "Recommended"
    st.header("Predicted  - " + result)

