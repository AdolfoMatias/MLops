# %% [markdown]
# ### SUCESSO DE ESTUDANTES

# %%
import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.metrics import *
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

#%matplotlib inline

# %% [markdown]
# ### IMPOTANDO BASE DE DADOS

# %%
estudantes = pd.read_csv("estudantes.csv", sep=";")
estudantes.head()

# %% [markdown]
# ### EXPLORANDO DADOS

# %%
#informações dataset
estudantes.info()

# %%
#número de linhas e coluans do dataframe
estudantes.shape

# %%
#presença de nulos
estudantes.isnull().sum()

# %%
# fig, ax = plt.subplots(1, sharey=True )
# heatmap = sns.heatmap(estudantes.select_dtypes(include="float64"), ax=ax)
# fig.set_size_inches(18.5, 10.5)


# %% [markdown]
# #### PREPROCESSAMENTO

# %%
for i in range(37):
    estudantes = estudantes.rename({estudantes.columns[i]:i}, axis=1)


    

# %%
labele = LabelEncoder()
for colunas in estudantes.iloc[:,:36].select_dtypes(include="object"):
    estudantes[colunas] = labele.fit_transform(estudantes.loc[colunas])

# %%
#Redimensioando com a Padronização
padronizar = StandardScaler()
estudantes.iloc[:,:36] = padronizar.fit_transform(estudantes.iloc[:,:36])
estudantes.head()

# %%
#Separando entre clase e previsores
previsores = estudantes.iloc[:, 0:36].values
classe =estudantes.iloc[:, 36].values
classe

# %%
X_train, X_test, y_train,y_test = train_test_split(previsores, classe, test_size=0.3, random_state=42)

# %% [markdown]
# ##### CRIANDO MODELOS

# %%
clf = [ GaussianNB(), 
    DecisionTreeClassifier(random_state=42),  
    RandomForestClassifier(
    n_estimators=500, 
    min_samples_leaf=25, random_state=42), 

    GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.01,
    subsample=1,
    max_depth=6,
    random_state=42
)]
print(len(clf))

# %%

lista_clf=[]
for i in range(len(clf)):
    modelo = clf[i]
    modelo = modelo.fit(X_train, y_train)
    lista_clf.append(modelo)
    


# %%
lista_prev =[]
for item in lista_clf:
    previsao = item.predict(X_test)
    lista_prev.append(previsao)
    

# %%
lista_classificadores = ["Naive Bayes", "Árvore de Decisão","Floresta de Decisão", "XGBoost"]
contador=0
for item in lista_prev:
    acuracia = accuracy_score(y_test, item)
    mod = lista_classificadores[contador]
    print(f"{mod}: {round(acuracia,2)}")
    contador+=1


