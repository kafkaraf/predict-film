#!/usr/bin/env python
# coding: utf-8

# In[18]:


import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict


# In[19]:


import joblib


st.title('Classifying Movie Runtime Category')
st.markdown("This model predicts the runtime category of a movie (Short, Medium, Long, Very Long) based on its features.")

st.header("Movie Features")
col1, col2 = st.columns(2)

with col1:
    st.text("Basic Movie Information")
    runtime = st.slider('Movie Runtime (minutes)', 0, 300, 120)
    vote_average = st.slider('Vote Average (1 to 10)', 1.0, 10.0, 7.0)

with col2:
    st.text("Genre Features (Select if applicable)")
    action = st.selectbox('Action Genre', [0, 1], index=0)
    comedy = st.selectbox('Comedy Genre', [0, 1], index=0)
    drama = st.selectbox('Drama Genre', [0, 1], index=0)
    thriller = st.selectbox('Thriller Genre', [0, 1], index=0)

st.text("")

if st.button("Predict Movie Runtime Category"):
    input_data = pd.DataFrame({
        'runtime': [runtime],
        'vote_average': [vote_average],
        'Action': [action],
        'Comedy': [comedy],
        'Drama': [drama],
        'Thriller': [thriller]
    })

    input_data_scaled = scaler.transform(input_data)

    result = model.predict(input_data_scaled)

    st.text(f'Predicted Runtime Category: {result[0]}')


# In[ ]:




