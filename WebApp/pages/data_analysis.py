#Import libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def app():
    #Make sure classification has been run before analyzing results
    if st.session_state.dataframe is None:
        st.markdown("<h3 style='text-align: center; color:Red;'>Please upload and run classification on images first.</h3>", unsafe_allow_html=True)
    else:
        #Display charts using the predictions so that the glitch types in the data can be analyzed effectively
        raw_data = st.session_state.dataframe
        classes = pd.unique(raw_data.iloc[:, 1])
        counts = [sum(raw_data.iloc[:, 1] == className) for className in classes]
        data = {'Class': classes, 'Count': counts}
        df = pd.DataFrame(data=data)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.dataframe(df)
        chartType = st.selectbox('Select chart type', ['Bar', 'Treemap', 'Pie'])
        if chartType=='Bar':
            fig = px.bar(df, x='Class', y='Count', text_auto='.2s')
            st.plotly_chart(fig)
        elif chartType=='Treemap':
            fig = px.treemap(df, path=['Class'],values='Count', color='Class')
            fig.data[0].textinfo = 'label+text+value'
            st.plotly_chart(fig)
        elif chartType=='Pie':
            fig = px.pie(df, names='Class', values='Count')
            st.plotly_chart(fig)
