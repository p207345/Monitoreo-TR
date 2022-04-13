import streamlit as st
import pandas as pd

with st.sidebar:
    st.write(""" # Calculadora
    ## Suma""")
    num1 = st.slider("Escoge el primer número",0.0,300.0,15.0)
    num2 = st.slider("Escoge el segundo número",0.0,300.0,15.0)
    st.write("La suma de esos números es:", num1+num2)
    st.write("## Multiplicación")
    num3 = st.number_input("Escribe el primer número")
    num4 = st.number_input("Escribe el segundo número")

    st.write("La multiplicación de esos números es:", num3*num4)