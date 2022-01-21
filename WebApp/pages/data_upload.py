#Import libraries
import streamlit as st
import numpy as np
from PIL import Image

#Initialize variables
data = []
files = []

def app():
    #Page header/instructions
    st.markdown("### Upload images for classification")
    st.write("\n")
    
    #Allows uploaded data to be seen on other pages
    if 'files' not in st.session_state:
        st.session_state.files = []
    if 'data' not in st.session_state:
        st.session_state.data = []
    if 'dataframe' not in st.session_state:
        st.session_state.dataframe = None

    #Create area to upload files and store the data in a variable
    uploaded_files = st.file_uploader("Choose file(s)", type = ['jpeg', 'png'], accept_multiple_files=True)
    if uploaded_files is not None:
        for file in uploaded_files:
            image = Image.open(file).convert('RGB').resize((300,300))
            imageArray = np.array(image)
            imageArray = np.expand_dims(imageArray, 0)
            data.append(imageArray)
            files.append(file.name)
    st.session_state.data = data
    st.session_state.files = files
