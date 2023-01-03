from numpy.core.fromnumeric import size
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.naive_bayes import GaussianNB


def gauss(data):
    st.title('Clasificador Gaussiano')
    col1, col2 = st.columns(2)
    with col1:
        param = st.selectbox('Parmetro de aproximacion', data.columns)
    with col2:
        remove_column = st.selectbox('Quitar columna', data.columns)

    listaa = data.columns.values.tolist()

    # elimino de la tabla el parametro predecir
    listaa.remove(param)
    if remove_column != 'Seleccionar':
        listaa.remove(remove_column)
    # ahora con el eliminado buscarlo y guardarlo
    result = data[param]

    listadedf = []
    for i in listaa:
        aux = data[i]
        aux = np.asarray(aux)
        
        listadedf.append(aux)
    listadedf = np.array(listadedf)

    le = preprocessing.LabelEncoder()

    listafittransform = []
    for x in listadedf:
        listafittransform.append(le.fit_transform(x))

    label = le.fit_transform(result)

    with st.expander("Resultado con etiquetas"):
        featuresencoders = list(zip((listafittransform)))
        featuresencoders = np.array(featuresencoders)
        tamcolumnas = len(listaa)
        tamfilas = featuresencoders.size
        featuresencoders = featuresencoders.reshape(int(tamfilas/tamcolumnas), tamcolumnas)

        st.dataframe(featuresencoders)

    with st.expander('Resultados sin etiquetas'):
        features = list(zip(np.asarray(listadedf)))
        features = np.asarray(features)
        tamcolumnas = len(features)
        tamfilas = features.size
        features = features.reshape(int(tamfilas/tamcolumnas), tamcolumnas)
        st.dataframe(features)

    model = GaussianNB()
    model2 = GaussianNB()

    model.fit(np.asarray(features), np.asarray(result))
    model2.fit(featuresencoders, label)

    columna = len(listaa)
    texto = "Ingrese "+str(columna)+" parametros del vector a comparar, separados por coma(,)"
    predecirresult = st.text_input(texto, '')

    if predecirresult != '':
        entrada = predecirresult.split(",")
        map_obj = list(map(int, entrada))
        map_obj = np.array(map_obj)
        predicted = model.predict(np.asarray([map_obj]))
        predicted2 = model2.predict(np.asarray([map_obj]))
        print(np.asarray([map_obj]))
        
        co1, co2, co3 = st.columns(3)
        with co2:
            st.subheader('Prediccion con etiquetas')
            st.write(predicted)

        coo1, coo2, coo3 = st.columns(3)
        with coo2:
            st.subheader('Prediccion sin etiquetas')
            st.write(predicted2)
