import streamlit as st
import pandas as pd

with st.sidebar:

    st.write("# Variables")
    database = st.radio(
     "Base de datos",
     ('1', '2', '3'))

if database == '1':
        st.write("""# Has seleccionado la base de datos 1""")
        df = pd.read_csv('https://raw.githubusercontent.com/IgnacioRodriguez98/Monitoreo-TR/main/Data/norm.csv', header=None)
        st.write(df)

elif database == '2':
    st.write("""# Has seleccionado la base de datos 2""")
    df = pd.read_csv('https://raw.githubusercontent.com/IgnacioRodriguez98/Monitoreo-TR/main/Data/normCH.csv', header=None)
    st.write(df)

else:
    st.write("""# Has seleccionado la base de datos 3""")
    df = pd.read_csv('https://raw.githubusercontent.com/IgnacioRodriguez98/Monitoreo-TR/main/Data/normJA.csv', header=None)
    st.write(df)

C2H2 = st.checkbox("Acetileno", check)
H2 = st.checkbox("Hidrógeno", check)
C2H4 = st.checkbox("Etileno", check)
CO = st.checkbox("Monóxiod de carbono", check)
C2H6 = st.checkbox("Etano", check)
CH4 = st.checkbox("Metano", check)
