
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
if H2:
    b = df[[0,2]]
    lista.append(b)
    con += 1
    b2 = 1
if C2H4:
    c = df[[0,3]]
    lista.append(c)
    con += 1
    b3 = 1
if CO:
    d = df[[0,4]]
    lista.append(d)
    con += 1
    b4 = 1
if C2H6:
    e = df[[0,5]]
    lista.append(e)
    con += 1
    b5 = 1
if CH4:
    f = df[[0,6]]
    lista.append(f)
    con += 1
    b6 = 1
st.write(con)
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
    #time.sleep(1)
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
    gases = [1]

elif con == 2:
    gases = [1,2]

elif con == 3:
    gases = [1,2,3]

elif con == 4:
    gases = [1,2,3,4]

elif con == 5:
    gases = [1,2,3,4,5]

elif con == 6:
    gases = [1,2,3,4,5,6]

else:
    st.write("Por favor escoge los gases a analizar")

if con > 1:

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
    indices = pd.DataFrame(np.where(y_test_scores > 2.5))
    X_test = db[round((len(db)/3)*2):]
    X_test.reset_index(inplace=True)
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

    #Ac = db.iloc[:, [1]]
    #H2 = db.iloc[:, [2]]
    #Et = db.iloc[:, [3]]
    CO = db.iloc[:, [0]]
    #Eta = db.iloc[:, [5]]
    #Me = db.iloc[:, [6]]

#Monoxido de carbono
    outliers_fraction = float(.15)
    scaler = StandardScaler()
    np_scaled = scaler.fit_transform(CO.values.reshape(-1, 1))
    data = pd.DataFrame(CO.iloc[:, [0]])
    model =  IsolationForest(contamination=outliers_fraction)
    model.fit(data)

    CO['anomaly'] = model.predict(data)

# visualization
    fig4, ax = plt.subplots(figsize=(10,6))

    a = CO.loc[CO['anomaly'] == -1, [4]] #anomaly

    ax.plot(CO.index, CO.iloc[:, [0]], color='black', label = 'Normal')
    ax.scatter(a.index,a.iloc[:, [0]], color='red', label = 'Anomaly')
    plt.title("Monoxido de Carbono")
    plt.legend()
    plt.show();
    st.write(fig4,ax)

########## Grafica dinamica

import matplotlib.animation as animation

plt.style.use('fivethirtyeight')

set_ind=set({})

def animate(i):
    data = db
    x = data.index
    #x1 = data.iloc[:, [0]]
    y1 = data.iloc[:, [0]]
    y2 = data.iloc[:, [1]]
    y3 = data.iloc[:, [2]]
    y4 = data.iloc[:, [3]]
    y5 = data.iloc[:, [4]]
    y6 = data.iloc[:, [5]]

    

    # Get the outlier scores for the train data
    #data2 = data[['C2H2_Acetylene', 'H2_Hydrogen', 'C2H4_Ethylene',
    #       'CO_CarbonMonoxide','C2H6_Ethane', 'CH4_Methane']]
    df_test = data.copy()
    #estandarizar
    
    df_test = scaler.fit_transform(df_test)
    df_test = pd.DataFrame(df_test)
    #atc.clf.fit(df_test)
    y_train_scores = clf.decision_scores_  

    # Predict the anomaly scores
    y_test_scores = clf.decision_function(df_test)  # outlier scores
    y_test_scores = pd.Series(y_test_scores)

    df_test['score'] = y_test_scores
    df_test['cluster'] = np.where(df_test['score']<4, 0, 1)
    df_test['cluster'].value_counts()
    t = df_test.groupby('cluster').mean()
    indices = list(np.where(y_test_scores > 3))
    for v in indices:
        set_ind.update(v)
    
    rayas2 = set({})    
    rayas = pd.DataFrame(set_ind).sort_values(0)
    df = pd.DataFrame(set_ind).sort_values(0)
    df["dif"] = df.diff()
    df.rename(columns={0:"orig"},inplace = True)
    df.reset_index(inplace=True)
    df.drop("index",axis=1,inplace=True)
    defin = []

    for ind, row in df.iterrows():
        #print(df.iloc(ind,1))
        if df.dif[ind] == "":
            next
        if df.dif[ind] != 1:
            defin.append(df.orig[ind])
        else:
            if df.dif[ind+1] != 1:
                defin.append(df.orig[ind])
            else:
                next
                
                
    rayas2.update(defin)
    rayas2 = pd.DataFrame(rayas2)


    
    plt.cla()
    plt.plot(x1, y1, label='C2H2')
    plt.plot(x1, y2, label='H2')
    plt.plot(x1, y3, label='C2H4')
    plt.plot(x1, y4, label='CO')
    plt.plot(x1, y5, label='C2H6')
    plt.plot(x1, y6, label='CH4')
    plt.xticks(rotation=90)
    plt.vlines(rayas2,0,50,"r")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show();    
    
    
    
ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)
plt.tight_layout()
plt.show();
st.write(ani)