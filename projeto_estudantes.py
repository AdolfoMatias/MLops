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
# #### CRIANDO ESTRTURA EM CLASSE PARA BUSCAR NO XGBOOT UM BOM RESULTADO

# %%
class ModeloMl:
    def __init__(self, estimadores, aprendizado, random_state, max_depth):
        self._n_estimadores = estimadores
        self._learning_rate = aprendizado
        self._random_state=random_state
        #self._subsample = subsample
        self._max_depth = max_depth

    def procurar(self):
        mlflow.set_experiment("ProcuraBest")
        with mlflow.start_run():

            #cirando modelo
            modelo = GradientBoostingClassifier(n_estimators=self._n_estimadores,
            learning_rate=self._learning_rate,
            max_depth=self._max_depth,
            #subsample=self._subsample,
            random_state=self._random_state)

            modelo = modelo.fit(X_train, y_train)

            #criando artefatos gráficos
            previsao = modelo.predict(X_test)
            mc = confusion_matrix(y_test, previsao)
            labels_name = ["Dropout", "Enrolled", "Graduated"]
            matgraph = ConfusionMatrixDisplay(mc, display_labels=labels_name)
            matgraph.plot()
            plt.savefig("matgraph.png")

            #salvando parametros
            mlflow.log_param("n_estimators", self._n_estimadores)
            mlflow.log_param("learning_rate", self._learning_rate)
            mlflow.log_param("random_state", self._random_state)
            #mlflow.log_param("subsample", self._subsample)
            mlflow.log_param("maxdepth", self._max_depth)

            #fazendo metricas
            acuracia = accuracy_score(y_test, previsao)
            #precisoa acuracia e f1 não aceitam binary então precisa mudar para micro, macro,weighted ou sample
            precisao = precision_score(y_test, previsao, average="macro")
            recall = recall_score(y_test, previsao, average="macro")
            f1score = f1_score(y_test, previsao, average="macro")
            #salvando métricas
            mlflow.log_metric("acuracia",acuracia)
            mlflow.log_metric("precisao",precisao)
            mlflow.log_metric("recall",recall)
            mlflow.log_metric("f1_score",f1score)

            #salvando gráficos
            mlflow.log_artifact("matgraph.png")
            
            mlflow.sklearn.log_model(modelo, "XGBoost")
            print("Modelo", mlflow.active_run().info.run_uuid)
        mlflow.end_run()
    


# %% [markdown]
# #### ANALISANDO FEATURES IMPORTANTES

# %%
modelo_extra = ExtraTreeClassifier()
modelo_extra = modelo_extra.fit(X_train, y_train)

prevendo_extras = modelo_extra.predict(X_test)
modelo_extra.feature_importances_


# %% [markdown]
# #### VARREDURA DE MELHORES MODELOS COM LAÇO

# %%
contador =1000
apr = 0.01
prf= 10
for elemento in range(10):
    mp = ModeloMl(1000,apr,42,prf)
    mp.procurar()
    apr+=0.01
    prf-=1

# %% [markdown]
# #### MELHOR MODELO ATÉ O MOMENTO

# %%
mp = ModeloMl(100,0.1,42,5)
mp.procurar()

# %% [markdown]
# ##### use mlflow  ui --port 5000

# %% [markdown]
# DOCUMENTAÇÂO PRECISION: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html


