import streamlit as st 
import spacy_streamlit as spt 
import spacy

from transformers import pipeline

pipe = pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("lxyuan/distilbert-base-multilingual-cased-sentiments-student")
model = AutoModelForSequenceClassification.from_pretrained("lxyuan/distilbert-base-multilingual-cased-sentiments-student")


st.title('TEXT SENTIMENT ANALYSIS APP')
menu = ['Home', 'Analysis']
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Home':
    st.subheader('About')
    st.write("This model analyze digital text to determine if the emotional tone of the message is positive, negative, or neutral.")
    st.image('Home.png', width=600)
elif choice == 'Analysis':
    st.subheader('Text Analysis')
    raw_text = st.text_area('Enter Text Here') 
    docs = pipe(raw_text)
    d = docs[0]
    res=d['label']
    if st.button('Analysis'):
        st.write(res)
        if res=='positive':
            st.image('positive.jpg', width=300)
        elif res=='negative':
            st.image('negative.jpg', width=300)
        else:
            st.image('neutral.jpg', width=300)