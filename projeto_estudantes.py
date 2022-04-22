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
# for i in range(37):
#     estudantes = estudantes.rename({estudantes.columns[i]:i}, axis=1)


    

# %%
labele = LabelEncoder()
estudantes["Target"] = labele.fit_transform(estudantes["Target"])
# for colunas in estudantes.iloc[:,:37].select_dtypes(include="object"):
#     estudantes.loc[colunas] = labele.fit_transform(estudantes.loc[colunas])

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
n_estimators=500
learning_rate=0.01
subsample=1
max_depth=6
random_state=42

clf = [ GaussianNB(), 
    DecisionTreeClassifier(random_state=42),  
    RandomForestClassifier(
    n_estimators=500, 
    min_samples_leaf=25, random_state=42), 

    GradientBoostingClassifier(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    subsample=subsample,
    max_depth=max_depth,
    random_state=random_state
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

# %%
#Opção com Saborn
confusao_matrix = confusion_matrix(y_test, lista_prev[3])
matriz= sns.heatmap(confusao_matrix, annot=True, cmap="Blues")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.figure()

plt.show()
# plt.savefig("confusao.png")


# %%
#Opção 2
categorias= ["Dropout", "Enrolled", "Graduate"]
confusao = ConfusionMatrixDisplay(confusao_matrix, display_labels=categorias)
confusao.plot()
#Salvando visualização
plt.savefig("MatrizConfusao.png")

#opção3 usando plot_confusion_matrix(modelo, teste, previsao)

# %%
mlflow.set_experiment("BestModel")
with mlflow.start_run():
    #logando métricas
    mlflow.log_metric("acuracia", acuracia)

    #logando parametros
    mlflow.log_param("n_estimators",n_estimators)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("subsample", subsample)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)

    #Imagemens
    mlflow.log_artifact("MatrizConfusao.png")
   

    #Modelo
    mlflow.sklearn.log_model(lista_clf[3],"XGBoost")
    print("Modelo: ",mlflow.active_run().info.run_uuid)


    mlflow.end_run()

# %% [markdown]
# ##### use mlflow  ui --port 5000

# %%



