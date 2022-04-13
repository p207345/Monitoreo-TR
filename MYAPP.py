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
    

else:
        st.write("""# Has seleccionado la base de datos 3""")