
import streamlit as st
import pandas as pd
import plotly.express as px
#import matplotlib as plt
import time
import numpy as np



with st.sidebar:
    st.write("# Variables")
    database = st.radio(
     "Base de datos",
     ('1', '2', '3'))

    st.write("# Selecciona los gases a analizar:")
    C2H2 = st.checkbox("Acetileno")
    H2 = st.checkbox("Hidrógeno")
    C2H4 = st.checkbox("Etileno")
    CO = st.checkbox("Monóxido de carbono")
    C2H6 = st.checkbox("Etano")
    CH4 = st.checkbox("Metano")

if database == '1':
        st.write("""# Has seleccionado la base de datos 1""")
        df = pd.read_csv('https://raw.githubusercontent.com/IgnacioRodriguez98/Monitoreo-TR/main/Data/planta1.csv', header=None)
        st.write(df)

elif database == '2':
    st.write("""# Has seleccionado la base de datos 2""")
    df = pd.read_csv('https://raw.githubusercontent.com/IgnacioRodriguez98/Monitoreo-TR/main/Data/planta2.csv', header=None)
    st.write(df)

else:
    st.write("""# Has seleccionado la base de datos 3""")
    df = pd.read_csv('https://raw.githubusercontent.com/IgnacioRodriguez98/Monitoreo-TR/main/Data/planta3.csv', header=None)
    st.write(df)



##################### SELECCION DE GASES ######################

with st.sidebar:
    st.write("# Selecciona el tamaño de la ventana:")
    vent = st.slider("",1,len(df))

lista = []
p= df[0]

if C2H2:
    a = df[[0,1]]
    lista.append(a)

if H2:
    b = df[[0,2]]
    lista.append(b)

if C2H4:
    c = df[[0,3]]
    lista.append(c)

if CO:
    d = df[[0,4]]
    lista.append(d)

if C2H6:
    e = df[[0,5]]
    lista.append(e)

if CH4:
    f = df[[0,6]]
    lista.append(f)

if len(lista)> 0:

    for i in lista:
        p = pd.merge(p,i,on = 0, how='outer')
    p.drop([0],inplace=True, axis=1)

if C2H2== False |H2 == False | C2H4 == False |CO == False |C2H6 ==False |CH4 == False:
   p="""### No hay gases seleccionados, por favor selecciona al menos uno"""

######### Reproduccion tiempo real
db = []
for i, row in p.iterrows():
    #st.write(row)
    db.append(row)
    #time.sleep(.1)
db = pd.DataFrame(db)
st.write(db)
st.write(p)
############## Grafica
fig = px.line(db)#,animation_frame="index",animation_group=db)
st.write(fig)

#### AUTOENCODER
from pyod.models.auto_encoder import AutoEncoder



X_train = db[0:round((len(db)/3)*2)]
X_test = db[round((len(db)/3)*2):]
n_features = 1 #para gases
y_train = np.zeros(round((len(db)/3)*2))
y_test = np.zeros(len(db)-round((len(db)/3)*2))
y_train[round((len(db)/3)*2):] = 1
y_test[len(db)-round((len(db)/3)*2):] = 1
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

from sklearn.preprocessing import StandardScaler
X_train = StandardScaler().fit_transform(X_train)
X_train = pd.DataFrame(X_train)
X_test = StandardScaler().fit_transform(X_test)
X_test = pd.DataFrame(X_test)

clf = AutoEncoder(hidden_neurons =[25, 1, 1, 25],contamination=.01)
clf.fit(X_train)

# Get the outlier scores for the train data
y_train_scores = clf.decision_scores_  

# Predict the anomaly scores
y_test_scores = clf.decision_function(X_test)  # outlier scores
y_test_scores = pd.Series(y_test_scores)

import matplotlib.pyplot as plt

fig3 = plt.figure(figsize=(10,4))
plt.hist(y_test_scores, bins='auto')  
plt.title("Histogram for Model Clf Anomaly Scores")
plt.show();

df_test = X_test.copy()
df_test['score'] = y_test_scores
df_test['cluster'] = np.where(df_test['score']<4, 0, 1)
df_test['cluster'].value_counts()

t = df_test.groupby('cluster').mean()
indices = pd.DataFrame(np.where(y_test_scores > 1.6))
X_test = db[round((len(db)/3)*2):]
X_test.reset_index(inplace=True)


fig2 = plt.figure(2)
plt.plot(X_test.index,X_test.iloc[:, [1]])
plt.vlines([indices],0,36,"r")
#plt.xlim(400,600)
#plt.ylim(400,600)
plt.xlabel('Date Time')
plt.ylabel('CO_CarbonMonoxide')
plt.show();
st.write(fig3)
st.write(fig2)