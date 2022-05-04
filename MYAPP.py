
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib as plt
import time
import numpy as np



with st.sidebar:
    st.write("# Variables")
    database = st.radio(
     "Base de datos",
     ('1 (2432 datos)', '2 (9277 datos)', '3 (21533 datos)'))

    st.write("# Selecciona los gases a analizar:")
    C2H2 = st.checkbox("Acetileno")
    H2 = st.checkbox("Hidrógeno")
    C2H4 = st.checkbox("Etileno")
    CO = st.checkbox("Monóxido de carbono")
    C2H6 = st.checkbox("Etano")
    CH4 = st.checkbox("Metano")

if database == '1 (2432 datos)':
        st.write("""# Has seleccionado la base de datos 1""")
        df = pd.read_csv('https://raw.githubusercontent.com/IgnacioRodriguez98/Monitoreo-TR/main/Data/planta2.csv', header=None)
        st.write(df)

elif database == '2 (9277 datos)':
    st.write("""# Has seleccionado la base de datos 2""")
    df = pd.read_csv('https://raw.githubusercontent.com/IgnacioRodriguez98/Monitoreo-TR/main/Data/planta3.csv', header=None)
    st.write(df)

else:
    st.write("""# Has seleccionado la base de datos 3""")
    df = pd.read_csv('https://raw.githubusercontent.com/IgnacioRodriguez98/Monitoreo-TR/main/Data/planta1.csv', header=None)
    st.write(df)



##################### SELECCION DE GASES ######################

#with st.sidebar:
#    st.write("# Selecciona el tamaño de la ventana:")
#    vent = st.slider("",1,len(df))

lista = []
header = ["Date"]
p = df[0]
con = 0
b1 = 0
b2 = 0
b3 = 0
b4 = 0
b5 = 0
b6 = 0

if C2H2:
    a = df[[0,1]]
    lista.append(a)
    con += 1
    b1 = 1
    header.append("Acetileno")

if H2:
    b = df[[0,2]]
    lista.append(b)
    con += 1
    b2 = 1
    header.append("Hidrogeno")

if C2H4:
    c = df[[0,3]]
    lista.append(c)
    con += 1
    b3 = 1
    header.append("Etileno")

if CO:
    d = df[[0,4]]
    lista.append(d)
    con += 1
    b4 = 1
    header.append("Monóxido de carbono")

if C2H6:
    e = df[[0,5]]
    lista.append(e)
    con += 1
    b5 = 1
    header.append("Etano")

if CH4:
    f = df[[0,6]]
    lista.append(f)
    con += 1
    b6 = 1
    header.append("Metano")

if C2H2== False |H2 == False | C2H4 == False |CO == False |C2H6 ==False |CH4 == False:
   p=" "

if len(lista)> 0:

    for i in lista:
        p = pd.merge(p,i,on = 0, how='outer')
    #p.drop([0],inplace=True, axis=1)
    st.write(p)

if len(header) == 1:
    st.write("""### No hay gases seleccionados, por favor selecciona al menos uno para continuar""")
else:    
    
    q = p
    st.write(q)
    q.columns = header
    q["Date"] = pd.to_datetime(q["Date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    gs = []
    dat = []
    va = []
    c = 0
    gr = pd.DataFrame()
    for i in range(len(q)):
        for k in range(len(header[1:])):
            dat.append(q["Date"][i])
        for k in header[1:]:
            va.append(q[k][i])
        for k in header[1:]:
            gs.append(k)
            
    gr["Date"]=dat
    gr["Gas"]= gs
    gr["Valor"]= va


    st.write(q)
    st.write(gr)
    if st.button("Simulación tiempo real"):
        #st.write(len(q))
        fig = px.bar(gr, x= "Gas", y= "Valor", color="Gas",
        animation_frame= "Date", 
        animation_group= "Gas")
        fig.update_layout(width=800)
        st.write(fig)

    if st.button("Simulación tiempo real 2"):
        #st.write(len(q))
        fig = px.bar(q, x= "Gas", y= "Valor", color="Gas",
        animation_frame= "Date", 
        animation_group= "Gas")
        fig.update_layout(width=800)
        st.write(fig)

######### Reproduccion tiempo real
db = []
for i, row in p.iterrows():
    #st.write(row)
    db.append(row)
    #time.sleep(1)
st.write(db)
db.drop([],inplace=True, axis=1)
db = pd.DataFrame(db)

#st.write(db)
#st.write(p)
############## Grafica
fig = px.line(db)#,animation_frame="index",animation_group=db)
st.write(fig)

#### AUTOENCODER
from pyod.models.auto_encoder import AutoEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
gases = []
if con == 1:
    gases = [0]

elif con == 2:
    gases = [0,1]

elif con == 3:
    gases = [0,1,2]

elif con == 4:
    gases = [0,1,2,3]

elif con == 5:
    gases = [0,1,2,3,4]

elif con == 6:
    gases = [0,1,2,3,4,5]

elif con == 0:
   st.write("Por favor escoge los gases a analizar")

if con >= 2:

    q["Anomalias"]=np.zeros(len(p))
    X_train = db[0:round((len(db)/3)*2)]
    X_test = db[round((len(db)/3)*2):]
    n_features = con #para gases
    y_train = np.zeros(round((len(db)/3)*2))
    y_test = np.zeros(len(db)-round((len(db)/3)*2))
    y_train[round((len(db)/3)*2):] = 1
    y_test[len(db)-round((len(db)/3)*2):] = 1
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

#estandarizacion
    X_train = StandardScaler().fit_transform(X_train)
    X_train = pd.DataFrame(X_train)
    X_test = StandardScaler().fit_transform(X_test)
    X_test = pd.DataFrame(X_test)

    clf = AutoEncoder(hidden_neurons =[25, 2, 2, 25],contamination=.01)
    clf.fit(X_train)

# Get the outlier scores for the train data
    y_train_scores = clf.decision_scores_  

# Predict the anomaly scores
    y_test_scores = clf.decision_function(X_test)  # outlier scores
    y_test_scores = pd.Series(y_test_scores)
    fig3 = plt.figure(figsize=(10,4))   
    plt.hist(y_test_scores, bins='auto')  
    plt.title("Histogram for Model Clf Anomaly Scores")
    plt.show();
    df_test = X_test.copy()
    df_test['score'] = y_test_scores
    df_test['cluster'] = np.where(df_test['score']<4, 0, 1)
    df_test['cluster'].value_counts()

    t = df_test.groupby('cluster').mean()
    indices = pd.DataFrame(np.where(y_test_scores > (t["score"].max()-.5)))
    q["Anomalias"][indices]=1
    X_test = db[round((len(db)/3)*2):]
    X_test.reset_index(inplace=True)
    X_test.drop(["index"],axis=1,inplace=True)
    fig2 = plt.figure(2)
    plt.plot(X_test.index,X_test.iloc[:, gases])
    plt.vlines([indices],0,X_test.max().max(),"r")
    plt.xlabel('Date Time')
    plt.ylabel('Gases')
    plt.show();
    st.write(fig3)
    st.write(fig2)

    

####### ISOLATION FOREST

from sklearn.ensemble import IsolationForest
if con == 1:
    CO = db.iloc[:, [0]]

#Parámetros
    outliers_fraction = float(.1)
    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(CO.values.reshape(-1, 1))
    data = pd.DataFrame(CO.iloc[:, [0]])
    model =  IsolationForest(contamination=outliers_fraction)
    model.fit(data)

    CO['anomaly'] = model.predict(data)

    if CO.columns[0]==1:
        fig4, ax = plt.subplots(figsize=(10,6))

        a = CO.loc[CO['anomaly'] == -1, [1]] #anomaly

        ax.plot(CO.index, CO.iloc[:, [0]], color='black', label = 'Normal')
        ax.scatter(a.index,a.iloc[:, [0]], color='red', label = 'Anomaly')
        plt.title("Acetileno")
        plt.legend()
        plt.show();
        st.write(fig4,ax)

    elif CO.columns[0]==2:
        fig4, ax = plt.subplots(figsize=(10,6))

        a = CO.loc[CO['anomaly'] == -1, [2]] #anomaly

        ax.plot(CO.index, CO.iloc[:, [0]], color='black', label = 'Normal')
        ax.scatter(a.index,a.iloc[:, [0]], color='red', label = 'Anomaly')
        plt.title("Hidrogeno")
        plt.legend()
        plt.show();
        st.write(fig4,ax)

    elif CO.columns[0]==3:
        fig4, ax = plt.subplots(figsize=(10,6))

        a = CO.loc[CO['anomaly'] == -1, [3]] #anomaly

        ax.plot(CO.index, CO.iloc[:, [0]], color='black', label = 'Normal')
        ax.scatter(a.index,a.iloc[:, [0]], color='red', label = 'Anomaly')
        plt.title("Etileno")
        plt.legend()
        plt.show();
        st.write(fig4,ax)

    elif CO.columns[0]==4:
        fig4, ax = plt.subplots(figsize=(10,6))

        a = CO.loc[CO['anomaly'] == -1, [4]] #anomaly

        ax.plot(CO.index, CO.iloc[:, [0]], color='black', label = 'Normal')
        ax.scatter(a.index,a.iloc[:, [0]], color='red', label = 'Anomaly')
        plt.title("Monoxido de Carbono")
        plt.legend()
        plt.show();
        st.write(fig4,ax)
        
    elif CO.columns[0]==5:
        fig4, ax = plt.subplots(figsize=(10,6))

        a = CO.loc[CO['anomaly'] == -1, [5]] #anomaly

        ax.plot(CO.index, CO.iloc[:, [0]], color='black', label = 'Normal')
        ax.scatter(a.index,a.iloc[:, [0]], color='red', label = 'Anomaly')
        plt.title("Etano")
        plt.legend()
        plt.show();
        st.write(fig4,ax)

    elif CO.columns[0]==6:
        fig4, ax = plt.subplots(figsize=(10,6))

        a = CO.loc[CO['anomaly'] == -1, [6]] #anomaly

        ax.plot(CO.index, CO.iloc[:, [0]], color='black', label = 'Normal')
        ax.scatter(a.index,a.iloc[:, [0]], color='red', label = 'Anomaly')
        plt.title("Metano")
        plt.legend()
        plt.show();
        st.write(fig4,ax)    
# visualization
    

########## Grafica dinamica

