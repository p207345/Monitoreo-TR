import streamlit as st
import pandas as pd

with st.sidebar

    genre = st.radio(
     "¿Que bases de datos quieres graficar?",
     ('1', '2', '3'))

    if genre == '1':
         st.write("#Has seleccionado la base de datos 1")

    elif database == '2':
        st.write("#Has seleccionado la base de datos 2")

    else:
         st.write("#Has seleccionado la base de datos 3")