# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 17:10:54 2023

@author: langh
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import ClusterCentroids
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from collections import Counter
from sklearn.metrics import roc_curve, roc_auc_score
import xgboost as xgb
import re
from pathlib import Path
import os

directorio = Path(r'C:\Users\langh\OneDrive\Desktop\DATA Sc\Fil Rouge')
os.chdir(directorio)

df_base = pd.read_csv("weatherAUS.csv")
#df_base = pd.DataFrame(weatherAUScsv, columns=weatherAUScsv[0])
df_base = df_base.drop(df_base.index[0])

df = pd.read_excel("df_knn.xlsx", engine='openpyxl')
df  = df_knn

ville_cluster = pd.read_csv("climats.csv", index_col="Location")
#ville_cluster = pd.DataFrame(climatscsv).drop(ville_cluster.index[0])
ville_cluster.drop(columns=ville_cluster.columns[ville_cluster.columns.str.contains('Unnamed')], inplace=True)


##Ajouter la colonne contenant le numéro de la grappe de chaque ville
columnas_ite = df.loc[:, "Albany":"Woomera"].columns
for i in df.index :
    
    for col in columnas_ite :
       # print(df.at[i, col], i, col)
        
        if df.at[i, col]== 1:
            for ville in ville_cluster.index:
                if re.search(ville, col):
                   #print(ville_cluster[ ville_cluster.index == ville].values)
                   df.at[i, "Cluster"] = ville_cluster[ ville_cluster.index == ville].values
    
def train_test(df, col_target, test_size):
    data=df.drop(col_target, axis=1)
    target = df[col_target]
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, stratify=target, random_state=77)
    return X_train, X_test, y_train, y_test

def app_standard_scaler(df, col_no_scaler): ##Applique un StandardScaler à un DataFrame.
    scaler = StandardScaler()
    columnas_to_scaler = [col for col in df.columns if col not in col_no_scaler]
    df[columnas_to_scaler] = scaler.fit_transform(df[columnas_to_scaler])
    return df

def renommer_colonnes_doublons(df):
    """
    Renombra las columnas duplicadas en un DataFrame de pandas agregando un '2' al final del nombre 
    de la segunda columna duplicada, un '3' para la tercera, y así sucesivamente.
    """
    # Contador para las apariciones de los nombres de las columnas
    comptes_colonnes = {}
    
    nouvelles_colonnes = []
    for colonne in df.columns:
        if colonne in comptes_colonnes:
            # Incrementa el contador para esta columna
            comptes_colonnes[colonne] += 1
            # Agrega el número correspondiente al final del nombre de la columna
            nouvelle_colonne = f"{colonne}{comptes_colonnes[colonne]}"
        else:
            # Inicializa el contador para este nombre de columna
            comptes_colonnes[colonne] = 1
            nouvelle_colonne = colonne
        
        nouvelles_colonnes.append(nouvelle_colonne)
    
    # Crea una copia del DataFrame y actualiza los nombres de las columnas
    df_nouveau = df.copy()
    df_nouveau.columns = nouvelles_colonnes
    
    return df_nouveau

def trouver_meilleurs_parametres(X_train, y_train, metodo):
    #-Définir l'espace de recherche des hyperparamètres
    parametres_rf = {
        'n_estimators': [200],
        'max_depth': [None, 5],
        'min_samples_split': [3, 5, 7]
    }
    
    parametres_xgb = {
        'n_estimators': [75, 100, 150],
        'max_depth': [5, 7],
        'learning_rate': [0.25, 0.1, 0.05]
    }
    
    if metodo == 1:
        # Utilizar Random Forest
        clf = RandomForestClassifier()
        parametres = parametres_rf
    elif metodo == 2:
        # Utilizar XGBoost
        clf = XGBClassifier()
        parametres = parametres_xgb
        
    # Initialiser la recherche en grille avec validation croisée (cross-validation)
    recherche_en_grille = GridSearchCV(clf, parametres, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    # Trouver les meilleurs hyperparamètres
    recherche_en_grille.fit(X_train, y_train)
    best_params = recherche_en_grille.best_params_
    
    return recherche_en_grille.best_estimator_, best_params





## Test de differetns Modeles
modele = 2  # 1 for RF et 2 pour KGBOOST
## Division de X_train pour utiliser le meem avec les differents modeles

df = df.dropna()
df_base = df_base.drop(df_base.loc[:, df_base['Location'] == 'Adelaine'].columns, axis=1)
df = renommer_colonnes_doublons(df)
X_train, X_test, y_train, y_test = train_test(df, "RainTomorrow", 0.3)





## 1.1 Modele Global (un seule modele pour tous les villes sans cluster)

X_train_sc = X_train.drop(columns="Cluster")
X_test_sc = X_test.drop(columns="Cluster")
clf, best_params= trouver_meilleurs_parametres(X_train_sc, y_train, modele)
y_test_pred = clf.predict(X_test_sc)
y_test_prob = clf.predict_proba(X_test_sc)[:, 1]
print("1.1:", best_params)
print("train", clf.score(X_train_sc, y_train))
print("test", clf.score(X_test_sc, y_test))


# ######

# fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)

# # Calcular el área bajo la curva (AUC)
# roc_auc = roc_auc_score(y_test, y_test_prob)

# # Grafica la curva ROC
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (área = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('Tasa de Falsos Positivos')
# plt.ylabel('Tasa de Verdaderos Positivos')
# plt.title('Curva ROC')
# plt.legend(loc="lower right")
# plt.show()

# ##
# ######

score_global = []
f1_global = []
villes = []
score_base = []
f1_base = []

y_test_pred_df = pd.DataFrame(y_test_pred)
y_test_pred_df["Index"] = X_test.index.tolist()


for ville in df_base.Location.unique():
    try:
        x_test_aux = X_test_sc[X_test_sc[ville] > 0]
        y_test_aux = y_test.loc[x_test_aux.index]
        y_test_pred_aux = y_test_pred_df[y_test_pred_df["Index"].isin(x_test_aux.index)][0]

        #print("el score pour", ville, "est :", clf.score(x_test_aux, y_test_aux))
        #print("el F1 pour", ville, "est :", f1_score(y_test_aux, y_test_pred_aux))
        villes.append(ville)
        score_global.append(clf.score(x_test_aux, y_test_aux))
        f1_global.append(f1_score(y_test_aux, y_test_pred_aux))
        
       
        y_test_pred_aux = pd.Series(np.zeros(len(y_test_aux)))
        matrix = confusion_matrix(y_test_aux, y_test_pred_aux)
        SCORE = (matrix[0][0]+matrix[1][1])/(matrix[0][0]+matrix[0][1]+matrix[1][0]+matrix[1][1])
        score_base.append(SCORE)
               
        
    except KeyError:
        print("error avec", ville)
        

#Ajouter la moyenne à la fin
villes.append("Moyenne")
score_global.append(np.mean([i for i in score_global if i !=0]))
score_base.append(np.mean([i for i in score_base if i !=0]))
f1_global.append(np.mean([i for i in f1_global if i !=0]))

## 1.2 Graphes Modele Global
#Afficher le score par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.35
plt.bar(indi, score_global, width=space, color='r', label='ML Global')
plt.bar(indi+space, score_base, width=space, color='b', label='Pred Base')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.6,1)
plt.grid(True, axis='y', alpha=0.8)
plt.ylabel('SCORE')
plt.show()

#Afficher le f1 par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.3
plt.bar(indi, f1_global, width=space, color='r', label='ML Global')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.4,1)
plt.grid(True, axis='y', alpha=0.8)
plt.ylabel('F1')
plt.show()
     





## 2-ML Modèle global avec CLuster

clf, best_params = trouver_meilleurs_parametres(X_train, y_train, modele)
y_test_pred = clf.predict(X_test)
print("2.1:", best_params)

### 2.1 Voir le resultat par ville
score_global_cluster = []
f1_global_cluster = []
villes = []

y_test_pred_df = pd.DataFrame(y_test_pred)
y_test_pred_df["Index"] = X_test.index.tolist()


for ville in df_base.Location.unique():
    try:
        x_test_aux = X_test[X_test[ville] > 0]
        y_test_aux = y_test.loc[x_test_aux.index]
        y_test_pred_aux = y_test_pred_df[y_test_pred_df["Index"].isin(x_test_aux.index)][0]

        #print("el score pour", ville, "est :", clf.score(x_test_aux, y_test_aux))
        #print("el F1 pour", ville, "est :", f1_score(y_test_aux, y_test_pred_aux))
        villes.append(ville)
        score_global_cluster.append(clf.score(x_test_aux, y_test_aux))
        f1_global_cluster.append(f1_score(y_test_aux, y_test_pred_aux))
        
       
        y_test_pred_aux = pd.Series(np.zeros(len(y_test_aux)))
        matrix = confusion_matrix(y_test_aux, y_test_pred_aux)
                       
    except KeyError:
        print("error avec", ville)
       

#Ajouter la moyenne à la fin
villes.append("Moyenne")
score_global_cluster.append(np.mean([i for i in score_global_cluster if i !=0]))

f1_global_cluster.append(np.mean([i for i in f1_global_cluster if i !=0]))

## 2.2 Graphes Modele Global CLUSTER
#Afficher le score par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.35
plt.bar(indi, score_global, width=space, color='r', label='ML Global')
plt.bar(indi+space, score_global_cluster, width=space, color='b', label='ML Global CLUSTER')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.75,1)
plt.grid(True, axis='y', alpha=0.8)
plt.ylabel('SCORE')
plt.show()

#Afficher le f1 par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.3
plt.bar(indi, f1_global, width=space, color='r', label='ML Global')
plt.bar(indi+space, f1_global_cluster, width=space, color='b', label='ML Global CLUSTER')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.4,1)
plt.grid(True, axis='y', alpha=0.8)
plt.ylabel('F1')
plt.show()
     





## 3 Cahque ville avec modèle individuel (un modèle par ville)
score_indiv = []
f1_indiv = []
villes = []
for ville in df_base.Location.unique():
    try:
        x_train_aux = X_train[X_train[ville] > 0]
        y_train_aux = y_train.loc[x_train_aux.index]
        x_test_aux = X_test[X_test[ville] > 0]
        y_test_aux = y_test.loc[x_test_aux.index]
                        
        clf, best_params = trouver_meilleurs_parametres(x_train_aux, y_train_aux, modele)
        y_test_pred_aux = clf.predict(x_test_aux)
        score_indiv.append(clf.score(x_test_aux, y_test_aux))
        f1_indiv.append(f1_score(y_test_aux, y_test_pred_aux))
        villes.append(ville)
        print("3.1:", best_params)
               
        
    except KeyError:
        print("error avec", ville)
        

#Ajouter la moyenne à la fin
villes.append("Moyenne")
score_indiv.append(np.mean([i for i in score_indiv if i !=0]))
f1_indiv.append(np.mean([i for i in f1_indiv if i !=0]))

#Afficher le score par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.26

plt.bar(indi, score_global, width=space, color='r', label='ML Global')
plt.bar(indi+space, score_global_cluster, width=space, color='b', label='CLUSTER global')
plt.bar(indi+space*2, score_indiv, width=space, color='g', label='ML Individuel')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.6,1)
plt.grid(True, axis='y', alpha=0.8)
plt.ylabel('SCORE')
plt.show()

#Afficher le f1 par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.3
plt.bar(indi, f1_global, width=space, color='r', label='ML Global')
plt.bar(indi+space, f1_global_cluster, width=space, color='b', label='CLUSTER global')
plt.bar(indi+space*2, f1_indiv, width=space, color='g', label='ML Individuel')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.3,1)
plt.grid(True, axis='y', alpha=0.8)
plt.ylabel('F1')
plt.show()




## 4.Calculer le Score et le F1 pour cahque Cluster ville avec modèle CLUSTER (un modèle par Cluster)
### 4.1 Calcul per Cluster
y_test_pred_CLUSTER = pd.DataFrame()   #pour le stockage des résultats

for cluster in df.Cluster.unique():
    try:
        x_train_aux = X_train[X_train["Cluster"] == cluster]
        y_train_aux = y_train.loc[x_train_aux.index]
        x_test_aux = X_test[X_test["Cluster"] == cluster]
        y_test_aux = y_test.loc[x_test_aux.index]
                        
        clf, best_params = trouver_meilleurs_parametres(x_train_aux, y_train_aux, modele)
        y_test_pred_aux = clf.predict(x_test_aux)  #Prediction per CLuester
        temp_df = pd.DataFrame({'index': x_test_aux.index, 'Predicciones':  y_test_pred_aux})  #temp DF avec le resultad del cluster et les Index original (index to be used to get resultat/ville)
        y_test_pred_CLUSTER = pd.concat([y_test_pred_CLUSTER, temp_df], ignore_index=True)  #Stockage de resultat
        print("4.1:", best_params)            
        
    except KeyError:
        score_indiv.append(0)
        f1_indiv.append(0)
        villes.append(ville)



## 4.2 Séparer les résultats par ville pour les afficher
score_cluster_ind = []
f1_cluster_ind = []
villes = []


for ville in df_base.Location.unique():
    try:
        x_test_aux = X_test[X_test[ville] > 0]
        y_test_aux = y_test.loc[x_test_aux.index]
        y_test_pred_aux = y_test_pred_CLUSTER[y_test_pred_CLUSTER["index"].isin(x_test_aux.index)][["Predicciones"]]

        #print("el score pour", ville, "est :", clf.score(x_test_aux, y_test_aux))
        #print("el F1 pour", ville, "est :", f1_score(y_test_aux, y_test_pred_aux))
        villes.append(ville)
        score_cluster_ind.append(clf.score(x_test_aux, y_test_aux))
        f1_cluster_ind.append(f1_score(y_test_aux, y_test_pred_aux))
                      
        
    except KeyError:
        print("error avec", ville)
        
        
#Ajouter la moyenne à la fin
villes.append("Moyenne")
score_cluster_ind.append(np.mean([i for i in score_cluster_ind if i !=0]))
f1_cluster_ind.append(np.mean([i for i in f1_cluster_ind if i !=0]))


## 4.3 Graphes Modele per CLUSTER
#Afficher le score par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.3
plt.bar(indi, score_global, width=space, color='r', label='Global sans Cluster')
plt.bar(indi+space, score_global_cluster, width=space, color='b', label='Cluster Global')
plt.bar(indi+space*2, score_cluster_ind, width=space, color='g', label='Cluster Ind')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.6,1)
plt.grid(True, axis='y', alpha=0.8)
plt.ylabel('SCORE')
plt.show()

#Afficher le f1 par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.3
plt.bar(indi, f1_global, width=space, color='r', label='ML Global')
plt.bar(indi+space, f1_global_cluster, width=space, color='b', label='ML Individuel')
plt.bar(indi+space*2, f1_cluster_ind, width=space, color='g', label='Global sans Cluster')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.3,1)
plt.grid(True, axis='y', alpha=0.8)
plt.ylabel('F1')
plt.show()