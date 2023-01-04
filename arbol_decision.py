from numpy.core.fromnumeric import size
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import preprocessing

def arbol_decision(data):
    param = st.selectbox('Parametro de aproximacion', data.columns)

    # data_top = data.columns.values
    listaa = data.columns.values.tolist()

    listaaux = ["Seleccionar"] + listaa
    eliminarcolumna = st.selectbox('Selecciona la columna que deseas eliminar', listaaux)

    # elimino de la tabla el parametro predecir
    listaa.remove(param)
    if eliminarcolumna != 'Seleccionar':
        listaa.remove(eliminarcolumna)
    # ahora con el eliminado buscarlo y guardarlo
    result = data[param]

    listadedf = []
    for i in listaa:
        aux = data[i]
        aux = np.asarray(aux)
        listadedf.append(aux)

    listadedf = np.array(listadedf)

    # Creacion del codificador de palabras
    le = preprocessing.LabelEncoder()

    # Se convierte los String a numeros
    listafittransform = []
    for x in listadedf:
        listafittransform.append(le.fit_transform(x))

    # Se convierte los string a numero del parametro
    label = le.fit_transform(result)
    with st.expander("Resultados"):
        st.header("Con etiquetas")
        fencoders = list(zip((listafittransform)))
        fencoders = np.array(fencoders)
        lengthcolumn = len(listaa)
        lengthrow = fencoders.size
        fencoders = fencoders.reshape(int(lengthrow / lengthcolumn), lengthcolumn)

        st.dataframe(fencoders)
        st.header("Sin etiquetas")
        features = list(zip(np.asarray(listadedf)))
        features = np.asarray(features)
        lengthcolumn = len(features)
        lengthrow = features.size
        features = features.reshape(int(lengthrow / lengthcolumn), lengthcolumn)
        st.dataframe(features)

    # se encaja con el modelo
    clf = DecisionTreeClassifier(max_depth=4).fit(features, result)
    fig, ax = plt.subplots()
    plot_tree(clf, filled=True, fontsize=10)

    with st.expander("Ver Graficas"):
        plt.figure(figsize=(50, 50))
        #st.pyplot(fig)

        clf2 = DecisionTreeClassifier(max_depth=5).fit(fencoders, label)
        fig2, ax2 = plt.subplots()
        plot_tree(clf2, filled=True)

        plt.figure(figsize=(60, 60))
        st.pyplot(fig2)
