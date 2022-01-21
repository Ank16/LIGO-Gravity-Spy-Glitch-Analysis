#Import libraries
import os
import streamlit as st
import numpy as np
from PIL import  Image

# Custom imports
from multipage import MultiPage
from pages import data_upload, machine_learning, data_analysis
# Create an instance of the app
app = MultiPage()

#Title and image of the main page
display = Image.open('Logo.png')
display = np.array(display)
st.markdown("<h1 style='text-align: center; color: #D1E8E2;'>LIGO Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;''><img src='https://www.ligo.caltech.edu/system/news_items/images/190/page/general_NSBH_merger.png?1624656509'></h2>", unsafe_allow_html=True)

#Add the webpages
app.add_page("Upload Data", data_upload.app)
app.add_page("Machine Learning", machine_learning.app)
app.add_page("Data Analysis", data_analysis.app)

#Run app
app.run()
