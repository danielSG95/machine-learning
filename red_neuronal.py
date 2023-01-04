import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import streamlit as st
def red_neuronal(data):
    st.title('Red Neuronal')
    col1, col2 = st.columns(2)
    with col1:
        x = st.selectbox('Seleccione la columna x')
    with col2:
        y = st.selectbox('Seleccione la columna y')

