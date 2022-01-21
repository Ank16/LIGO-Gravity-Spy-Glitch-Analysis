#Import libraries
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd

#Dictionary that can convert neural network output (a number) to the actual class name
classDict = {0: 'Light Modulation', 1: 'Scratchy', 2: 'Low Frequency Burst', 3: 'Violin Mode', 4: 'Repeating Blips', 5: 'Air Compressor', 6: 'No Glitch', 7: '1080Lines', 8: 'Chirp', 9: 'Low Frequency Lines', 10: 'Scattered Light', 11: '1400Ripples', 12: 'Tomte', 13: 'Blip', 14: 'Extremely Loud', 15: 'Helix', 16: 'Whistle', 17: 'None', 18: 'Power Line', 19: 'Wandering Line', 20: 'Paired Doves', 21: 'Koi Fish'}

#Define function that uses the neural network model and returns the classes of inputted data
@st.cache
def classifyImages(images, fileNames):
    model = keras.models.load_model('LIGOModel.h5')
    predictionList = []
    imageData = np.concatenate(images)
    generator = ImageDataGenerator(rescale = 1./255., samplewise_center=True, samplewise_std_normalization=True)
    X = generator.flow(imageData, batch_size=64, shuffle=False)
    with tf.device('/cpu:0'):
        predictions = model.predict(X)
    for prediction in predictions:
        predictionList.append(classDict.get(prediction.argmax()))
    df = pd.DataFrame(list(zip(fileNames, predictionList)), columns=['Filename', 'Prediction'])
    return df

#Function that allows model's predictions to be downloaded as csv
@st.cache
def convert_df(df):
     return df.to_csv(index=False)

def app():
    #Ensure data has been uploaded before running the neural network
    if st.session_state.files is None or not st.session_state.files:
        st.markdown("<h3 style='text-align: center; color:#05386B;'>Please upload at least 1 file on the Upload Files page before classification.</h3>", unsafe_allow_html=True)
    else:
        #Run the classification function once "Classify" button is pressed & page formatting/headers
        st.markdown("<h3 style='text-align: left; color:#05386B; margin-left:8.25em'>Ready to classify!</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns([0.675, 1])
        with col2:
            beginClassification = st.button('Classify!')
        if beginClassification:
            predictions = classifyImages(st.session_state.data, st.session_state.files)
            st.session_state.dataframe = predictions
            col1a, col2a = st.columns([0.01, 1])
            with col2a:
                st.dataframe(predictions)
            csv = convert_df(predictions)
            col1b, col2b = st.columns([0.55, 1])
            with col2b:
                st.download_button(label='Export results to CSV', data=csv, file_name='LIGO_Predictions.csv', mime='text/csv')
