# Import
import streamlit as st
from model_utils import inference

# Title and header
st.set_page_config(page_title='Translate EN-DE', page_icon=':1234:')
st.title('Translator [en-de]')
st.markdown('#')

# Sidebar
with st.sidebar:
    st.title('Translator')
    st.write('This EN-DE translator is powered by a seq2seq transformer model with architecture based on the original transformers paper.')
    st.header('Model Config')
    st.markdown('d<sub>model</sub> = 512<br/>d<sub>ff</sub> = 512<br/>h = 8<br/>N = 3<br/>dropout = 0.1', unsafe_allow_html=True)
    st.header('Parameters')
    st.markdown('24M Trainable Parameters')

# English input
st.subheader('English')
input_text = st.text_input('English text', label_visibility='collapsed', key='en')
output_text = ''
if(input_text):
    output_text = inference(input_text)

# Output
st.markdown('&nbsp;')
st.subheader('German')
st.write(output_text if len(output_text)>0 else ':grey[Enter english text to translate]')
# st.text_input('English text', disabled=True, label_visibility='collapsed', key='de')
# st.write('Dieser Text ist auf Deutsch verfasst')