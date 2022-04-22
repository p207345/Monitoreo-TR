import streamlit as st
import pandas as pd

with st.sidebar:

    st.write("# Variables")
    database = st.radio(
     "Base de datos",
     ('1', '2', '3'))

    st.write("# Selecciona los gases a analizar:")
    C2H2 = st.checkbox("Acetileno", check)
    H2 = st.checkbox("Hidrógeno", check)
    C2H4 = st.checkbox("Etileno", check)
    CO = st.checkbox("Monóxido de carbono", check)
    C2H6 = st.checkbox("Etano", check)
    CH4 = st.checkbox("Metano", check)
if C2H2 == check and H2,C2H4,CO,C2H6,CH4 == False:
    database = database["0"]
elif C2H2 == check and H2 == check:
    database = database["0","1"]
 
if H2 == check:
    database = database[:,1]

if C2H4 == check:
    database = database[:,2]

if CO == check:
    database = database[:,3] 

if C2H6 == check:
    database = database[:,4]

if CH4 == check:
    database = database[:,5]


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



