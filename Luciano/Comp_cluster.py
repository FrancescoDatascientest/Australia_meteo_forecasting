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

#definir directorio
new_directorio = "C:/Users/langh/OneDrive/Desktop/DATA Sc/Fil Rouge"
os.chdir(new_directorio)

#importation des donnéess
df_base = pd.read_csv("weatherAUS.csv")
#df_base = pd.DataFrame(weatherAUScsv, columns=weatherAUScsv[0])
# df_base = df_base.drop(df_base.index[0])
# df_base.index = df_base.index - 1
df_date = df_base[df_base.index.isin(df.index)]["Date"]
df_date = pd.to_datetime(df_date)

#charger le fichier df_knn.py
df = df_knn.copy()


# ville_cluster = pd.DataFrame(climatscsv, columns=climatscsv[0])
# ville_cluster.set_index("Location", inplace=True)
# ville_cluster.drop(columns=ville_cluster.columns[0], inplace=True)
# ville_cluster = ville_cluster.drop(ville_cluster.index[0])
ville_cluster = pd.read_csv("climats.csv", index_col="Location")
ville_cluster.drop(columns=ville_cluster.columns[ville_cluster.columns.str.contains('Unnamed')], inplace=True)


##Aajouter la colonne contenant le numéro de cluster de chaque ville
columnas_ite = df.loc[:, "Adelaide":"Woomera"].columns
for i in df.index :
    
    for col in columnas_ite :
                
        if df.at[i, col]== 1:
            for ville in ville_cluster.index:
                if re.search(ville, col):
                    
                   #print(ville, pd.to_numeric(ville_cluster[ville_cluster.index == ville].values[0], errors='coerce') )
                   df.at[i, "Cluster"] = pd.to_numeric(ville_cluster[ville_cluster.index == ville].values[0], errors='coerce') 
    

def diviser_par_date(df, cible, df_date, test_size=0.2, random_state=42):
    # Obtenez une liste unique de dates
    dates_uniques = df_date['Date'].unique()
    
    # Choisissez aléatoirement certaines dates pour l'ensemble de test
    dates_test = pd.to_datetime(pd.Series(dates_uniques).sample(frac=test_size, random_state=random_state))
    
    # Assurez-vous qu'il n'y a pas de dates en double dans dates_test
    dates_test = pd.Series(dates_test).unique()
    
    # Filtrez le DataFrame pour obtenir les lignes correspondant aux dates de test
    df_test = df[df_date['Date'].isin(dates_test)]
    
    # Les lignes restantes sont pour l'ensemble d'entraînement
    df_train = df[~df_date['Date'].isin(dates_test)]
    
    # Extraire les caractéristiques et les étiquettes
    X_train, X_test = df_train.drop(columns=[cible]), df_test.drop(columns=[cible])
    y_train, y_test = df_train[cible], df_test[cible]
    
    return X_train, X_test, y_train, y_test

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
        'n_estimators': [100],
        'max_depth': [None, 5],
        'min_samples_split': [3, 5, 7]
    }
    
    parametres_xgb = {
        'n_estimators': [75, 100, 250],
        'max_depth': [5, 7], #, 7
        'learning_rate': [0.25, 0.1] #, 0.01
    }
    
    parametres_xgb_2 = {
        'n_estimators': [50, 85, 120],
        'max_depth': [3, 5], #, 7
        'learning_rate': [0.3, 0.1] #, 0.01
    }
    
    if metodo == 1:
        # Utilizar Random Forest
        clf = RandomForestClassifier()
        parametres = parametres_rf
    elif metodo == 2:
        # Utilizar XGBoost
        clf = XGBClassifier(eval_metric='auc')
        parametres = parametres_xgb
    elif metodo == 3:
        # Utilizar XGBoost
        clf = XGBClassifier(eval_metric='auc')
        parametres = parametres_xgb_2
        
    # Initialiser a recherche en grille avec validation croisée (cross-validation)
    recherche_en_grille = GridSearchCV(clf, parametres, cv=6, scoring='accuracy', verbose=1, n_jobs=-1)
    # Trouver les meilleurs hyperparamètres
    recherche_en_grille.fit(X_train, y_train)
    best_params = recherche_en_grille.best_params_
    
    return recherche_en_grille.best_estimator_, best_params





## Test de differetns Modeles
modele = 3  # 1 for RF et 2 pour KGBOOST
## Division de X_train pour utiliser le meem avec les differents modeles

df = df.dropna()
#df_base = df_base.drop(df_base.loc[:, df_base['Location'] == 'Adelaine'].columns, axis=1)
df = renommer_colonnes_doublons(df)
X_train, X_test, y_train, y_test = train_test(df, "RainTomorrow", 0.3)
#X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)





## 1.1 Modele Global (un seule modele pour tous les villes sans cluster)

X_train_sc = X_train.drop(columns="Cluster")
X_test_sc = X_test.drop(columns="Cluster")
clf, best_params= trouver_meilleurs_parametres(X_train_sc, y_train, modele)
y_test_pred = clf.predict(X_test_sc)
y_test_prob = clf.predict_proba(X_test_sc)[:, 1]
print("1.1:", best_params)
print("train", clf.score(X_train_sc, y_train))
print("test", clf.score(X_test_sc, y_test))


score_global = []
f1_global = []
villes = []
score_base = []
f1_base = []

y_test_pred_df = pd.DataFrame(y_test_pred)
y_test_pred_df["Index"] = X_test.index.tolist()

print(df_base.Location.unique())

for ville in df_base.Location.unique():
    
    
    try:
        #ville = "Albany"
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
               
        
    except ValueError:
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
     
#Afficher Curve ROC
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
roc_auc = roc_auc_score(y_test, y_test_prob)

# Graf
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.4f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()




## 2-ML Modèle global avec CLuster

clf, best_params = trouver_meilleurs_parametres(X_train, y_train, modele)
y_test_pred = clf.predict(X_test)
y_test_prob = clf.predict_proba(X_test)[:, 1]
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
     
#Afficher Curve ROC
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
roc_auc = roc_auc_score(y_test, y_test_prob)

# Graf
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.4f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()


#Graf importances de features 
noms_variables = list(X_train.columns)
importances = clf.feature_importances_
indices = importances.argsort()[::-1]
noms_variables_tries = [noms_variables[i] for i in indices]

# Ggraphique à barres
plt.figure(figsize=(10, 25))  # Ajuster la taille selon les besoins
plt.barh(noms_variables_tries, importances[indices])
plt.xlabel('Importance')
plt.title('Importance des Variables')
plt.gca().invert_yaxis()  # Inverser l'ordre des variables
plt.show()





## 3 Cahque ville avec modèle individuel (un modèle par ville)

score_indiv = []
f1_indiv = []
villes = []
for ville in df_base.Location.unique():
    try:
        x_train_aux, x_test_aux, y_train_aux, y_test_aux = train_test(df[df[ville] > 0], "RainTomorrow", 0.3)
        
        # x_train_aux = X_train[X_train[ville] > 0]
        # y_train_aux = y_train.loc[x_train_aux.index]
        # x_test_aux = X_test[X_test[ville] > 0]
        # y_test_aux = y_test.loc[x_test_aux.index]
                        
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
plt.figure(figsize=(35, 12))
indi = np.arange(len(villes))
space = 0.26
plt.bar(indi, score_base, width=space, color='b', label='Pred Bas')
plt.bar(indi+space, score_global, width=space, color='r', label='ML Global')
plt.bar(indi+space*2, score_indiv, width=space, color='g', label='ML Individuel')
plt.xticks(indi, villes, rotation=90, fontsize=21)
plt.yticks(fontsize=21)
plt.legend(fontsize=20, loc='upper left')
plt.ylim(0.6,1)
plt.grid(True, axis='y', alpha=0.8)
plt.ylabel('ACCURACY', fontsize=21)
plt.xlabel('VILLES', fontsize=21)
#plt.tight_layout()  
plt.show()

#Afficher le f1 par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.33
plt.bar(indi, f1_global, width=space, color='r', label='ML Global')
#plt.bar(indi+space, f1_global_cluster, width=space, color='b', label='CLUSTER global')
plt.bar(indi+space, f1_indiv, width=space, color='g', label='ML Individuel')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.4,1)
plt.grid(True, axis='y', alpha=0.8)
plt.ylabel('F1')
plt.show()




## 4.0 Calculer si on ajoute les donnes de la veille

df_dup = df.copy()
aux = df_dup.shift(1)

for index in aux.index:
    # Vérifier si l'index-1 correspond à un jour de moins dans df_date
    try:
        if df_date.iloc[index-1] != (df_date.iloc[index] - pd.Timedelta(days=1)):
            aux.drop(index, inplace=True)
    except IndexError:
        print(index)

aux.columns = [f'{col}-1' for col in aux.columns]
df_veille = pd.merge(df_knn, aux, left_index=True, right_index=True, how='right')
#df_veille.dropna(inplace=True)
df_veille = renommer_colonnes_doublons(df_veille)

score_veille = []
f1_veille = []
villes = []
for ville in df_base.Location.unique():
    try:
        x_train_aux, x_test_aux, y_train_aux, y_test_aux = train_test(df_veille[df_veille[ville] > 0], "RainTomorrow", 0.3)
        # x_train_aux = X_train[X_train[ville] > 0]
        # y_train_aux = y_train.loc[x_train_aux.index]
        # x_test_aux = X_test[X_test[ville] > 0]
        # y_test_aux = y_test.loc[x_test_aux.index]
                        
        clf, best_params = trouver_meilleurs_parametres(x_train_aux, y_train_aux, modele)
        y_test_pred_aux = clf.predict(x_test_aux)
        score_veille.append(clf.score(x_test_aux, y_test_aux))
        f1_veille.append(f1_score(y_test_aux, y_test_pred_aux))
        villes.append(ville)
        print(ville, best_params)
               
        
    except KeyError:
        score_veille.append(0)
        f1_veille.append(0)
        villes.append(ville)
    
    except ValueError:
        score_veille.append(0)
        f1_veille.append(0)
        villes.append(ville)

#Ajouter la moyenne à la fin
villes.append("Moyenne")
score_veille.append(np.mean([i for i in score_veille if i !=0]))
f1_veille.append(np.mean([i for i in f1_veille if i !=0]))


#Afficher le score par ville
plt.figure(figsize=(32, 15))
indi = np.arange(len(villes))
space = 0.2
plt.bar(indi, score_base, width=space, color='b', label='Pred Bas')
plt.bar(indi+space, score_global, width=space, color='r', label='ML Global')
plt.bar(indi+space*2, score_indiv, width=space, color='g', label='ML Individuel')
plt.bar(indi+space*3, score_veille, width=space, color='black', label='ML Veille -ind-')
plt.xticks(indi, villes, rotation=90, fontsize=21)
plt.yticks(fontsize=21)
plt.legend(fontsize=21)
plt.ylim(0.6,1)
plt.grid(True, axis='y', alpha=0.8)
plt.ylabel('ACCURACY', fontsize=21)
plt.xlabel('VILLES', fontsize=21)
plt.tight_layout()  
plt.show()







## 4.Calculer le Score et le F1 pour cahque Cluster ville avec modèle CLUSTER (un modèle par Cluster)
### 4.1 Calcul per Cluster
y_test_pred_CLUSTER = pd.DataFrame()   #pour le stockage des résultats
y_test_prob_CLUSTER = pd.DataFrame()
for cluster in df.Cluster.unique():
    try:
        x_train_aux = X_train[X_train["Cluster"] == cluster]
        y_train_aux = y_train.loc[x_train_aux.index]
        x_test_aux = X_test[X_test["Cluster"] == cluster]
        y_test_aux = y_test.loc[x_test_aux.index]
                        
        clf, best_params = trouver_meilleurs_parametres(x_train_aux, y_train_aux, modele)
        y_test_pred_aux = clf.predict(x_test_aux)  #Prediction per CLuester
        
        
        temp_df = pd.DataFrame({'index': x_test_aux.index, 'Predicciones':  y_test_pred_aux})  #temp DF avec le resultad del cluster et les Index original (index to be used to get resultat/ville)
        y_test_pred_CLUSTER = pd.concat([y_test_pred_CLUSTER, temp_df])  #Stockage de resultat
        print("4.1:", best_params)
        
        #¹Prob pour courve ROC
        y_test_prob_aux = clf.predict_proba(x_test_aux)[:, 1]  
        temp_df_prob = pd.DataFrame({'Probabilidades':  y_test_prob_aux}, index=x_test_aux.index)  # Crear DataFrame con probabilidades
             
        y_test_prob_CLUSTER = pd.concat([y_test_prob_CLUSTER, temp_df_prob])

         
        
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
plt.ylim(0.75,1)
plt.grid(True, axis='y', alpha=0.8)
plt.ylabel('SCORE')
plt.show()

#Afficher le f1 par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.3
plt.bar(indi, f1_global, width=space, color='r', label='Global')
plt.bar(indi+space, f1_global_cluster, width=space, color='b', label='Global CLUSTER')
plt.bar(indi+space*2, f1_cluster_ind, width=space, color='g', label='Modele per Cluster')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.4,1)
plt.grid(True, axis='y', alpha=0.8)
plt.ylabel('F1')
plt.show()

#Afficher Curve ROC
y_test_prob_CLUSTER = y_test_prob_CLUSTER.sort_index()
y_test_aux2 = y_test.sort_index() ##Pour avoir le meme order que y_test_prob_CLUSTER

fpr, tpr, thresholds = roc_curve(y_test_aux2, y_test_prob_CLUSTER)
roc_auc = roc_auc_score(y_test_aux2, y_test_prob_CLUSTER)

# Graf
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.4f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()







### 5 Ajouter predccion des autres villes qu features - 2 ML en serie -
## Completer X_test avec valor reel


df_date = df_base.loc[df.index, ["Date"]]
df_date.Date = pd.to_datetime(df_date["Date"]) 


print( df_ext.RainTomorrow.isnull().sum() )

#♣Ajouter colonnes avec RainTOmorrow des autres villes
df_ext = df.copy()
for index, row in df_ext.iterrows():
    
    # Obtener la fecha correspondiente en df_date
    fecha_actual = df_date.loc[index, 'Date']  # Reemplaza 'tu_columna_de_fecha' con el nombre real de la columna
    
    # Obtener todos los índices en df_date que tienen la misma fecha
    indices_misma_fecha = df_date[df_date['Date'] == fecha_actual].index.tolist()
    #indices_misma_fecha = [i for i in indices_misma_fecha if i in df_ext.index]
    
    # Obtener los valores correspondientes en y_train
    valores_RainTomorrow = df_ext.RainTomorrow.loc[indices_misma_fecha].drop(index=index) #copie tous les jour row de la meme date et suprime l'index de la row a remplir
    
    # Agregar los valores como nuevas columnas a la fila actual de X_train
    for i, valor in enumerate(valores_RainTomorrow):
        ville = df_base.at[valores_RainTomorrow.index[i], "Location"]
        columna_nueva = f'Rain_{ville}'
        df_ext.loc[index, columna_nueva] = valor

df_ext.fillna(0, inplace=True)
        

print(df_ext.isnull().sum())
"Suprimmer la dernier colonne parce que il y plusier NaN"
df_ext = df_ext.drop(columns=["Rain_ville47"])
df_ext = df_ext.dropna()


## 5.2-ML Modèle global avec CLuster
X_train_ext, X_test_ext, y_train_ext, y_tes_ext = diviser_par_date(df_ext, "RainTomorrow", df_date, 0.3, 77)

clf_ext, best_params = trouver_meilleurs_parametres(X_train_ext, y_train_ext, modele)
y_test_pred = clf_ext.predict(X_test_ext)
y_test_prob = clf_ext.predict_proba(X_test_ext)[:, 1]
print("2.1:", best_params)

### 5.1 Voir le resultat par ville
score_global_cluster_ext = []
f1_global_cluster_ext = []
villes = []

y_test_pred_df = pd.DataFrame(y_test_pred)
y_test_pred_df["Index"] = X_test_ext.index.tolist()


for ville in df_base.Location.unique():
    try:
        x_test_aux = X_test_ext[X_test_ext[ville] > 0]
        y_test_aux = y_tes_ext.loc[x_test_aux.index]
        y_test_pred_aux = y_test_pred_df[y_test_pred_df["Index"].isin(x_test_aux.index)][0]

        #print("el score pour", ville, "est :", clf.score(x_test_aux, y_test_aux))
        #print("el F1 pour", ville, "est :", f1_score(y_test_aux, y_test_pred_aux))
        villes.append(ville)
        score_global_cluster_ext.append(clf_ext.score(x_test_aux, y_test_aux))
        f1_global_cluster_ext.append(f1_score(y_test_aux, y_test_pred_aux))
        
       
        y_test_pred_aux = pd.Series(np.zeros(len(y_test_aux)))
        matrix = confusion_matrix(y_test_aux, y_test_pred_aux)
                       
    except KeyError:
        print("error avec", ville)
       

#Ajouter la moyenne à la fin
villes.append("Moyenne")
score_global_cluster_ext.append(np.mean([i for i in score_global_cluster_ext if i !=0]))

f1_global_cluster_ext.append(np.mean([i for i in f1_global_cluster_ext if i !=0]))

## 5.2 Graphes Modele Global CLUSTER
#Afficher le score par ville
plt.figure(figsize=(25, 7))
indi = np.arange(len(villes))
space = 0.35
plt.bar(indi, score_global, width=space, color='r', label='ML Global')
plt.bar(indi+space, score_global_cluster_ext, width=space, color='b', label='ML Global CLUSTER')
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
plt.bar(indi+space, f1_global_cluster_ext, width=space, color='b', label='ML Global CLUSTER')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.4,1)
plt.grid(True, axis='y', alpha=0.8)
plt.ylabel('F1')
plt.show()
     
#Afficher Curve ROC
fpr, tpr, thresholds = roc_curve(y_test_ext, y_test_prob_ext)
roc_auc = roc_auc_score(y_test_ext, y_test_prob_ext)

# Graf
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.4f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()


#Graf importances de features 
noms_variables = list(X_train.columns)
importances = clf.feature_importances_
indices = importances.argsort()[::-1]
noms_variables_tries = [noms_variables[i] for i in indices]

# Ggraphique à barres
plt.figure(figsize=(10, 25))  # Ajuster la taille selon les besoins
plt.barh(noms_variables_tries, importances[indices])
plt.xlabel('Importance')
plt.title('Importance des Variables')
plt.gca().invert_yaxis()  # Inverser l'ordre des variables
plt.show()






## 6 Cahque ville avec modèle individuel (un modèle par ville)
## Avec donnes reel
score_indiv = []
f1_indiv = []
villes = []
for ville in df_base.Location.unique():
    try:
        x_train_aux = X_train_ext[X_train_ext[ville] > 0]
        y_train_aux = y_train_ext.loc[x_train_aux.index]
        x_test_aux = X_test_ext[X_test_ext[ville] > 0]
        y_test_aux = y_tes_ext.loc[x_test_aux.index]
                        
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
#plt.bar(indi+space, score_global, width=space, color='b', label='CLUSTER global')
plt.bar(indi+space, score_indiv, width=space, color='g', label='ML Individuel')
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
plt.bar(indi+space, f1_global, width=space, color='b', label='CLUSTER global')
plt.bar(indi+space*2, f1_indiv, width=space, color='g', label='ML Individuel')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.3,1)
plt.grid(True, axis='y', alpha=0.8)
plt.ylabel('F1')
plt.show()


#### Avec predicitons
##Construir X_test_ext_pred a partir de las predicciones de premier Modele


clf_ext, best_params = trouver_meilleurs_parametres(X_train_ext, y_train_ext, modele)
y_test_pred = clf_ext.predict(X_test_ext)
y_test_pred = pd.DataFrame(y_test_pred, index=X_test_ext.index, columns=['Predicciones'])

#♣Ajouter colonnes avec RainTOmorrow des autres villes
X_test_ext_pred = X_test_ext.copy() 

X_test_ext_pred = X_test_ext_pred.iloc[:, :-46] #Je supprime les 46 dernières colonnes (les Raintomorrow des autres villes).
for index, row in X_test_ext_pred.iterrows():
    
    # Obtener la fecha correspondiente en df_date
    fecha_actual = df_date.loc[index, 'Date']  # Reemplaza 'tu_columna_de_fecha' con el nombre real de la columna
    
    # Obtener todos los índices en df_date que tienen la misma fecha
    indices_misma_fecha = df_date[df_date['Date'] == fecha_actual].index.tolist()
    indices_misma_fecha = [i for i in indices_misma_fecha if i in df_ext.index]
    
    # Obtener los valores correspondientes en y_train
    valores_RainTomorrow = y_test_pred.Predicciones.loc[indices_misma_fecha].drop(index=index)
    
    # Agregar los valores como nuevas columnas a la fila actual de X_train
    for i, valor in enumerate(valores_RainTomorrow):
        ville = df_base.at[valores_RainTomorrow.index[i], "Location"]
        columna_nueva = f'Rain_{ville}'
        X_test_ext_pred.loc[index, columna_nueva] = valor
X_test_ext_pred.fillna(0, inplace=True)

## Calcules Deuxieme ML
score_indiv = []
f1_indiv = []
villes = []
for ville in df_base.Location.unique():
    try:
        x_train_aux = X_train_ext[X_train_ext[ville] > 0]
        y_train_aux = y_train_ext.loc[x_train_aux.index]
        x_test_aux = X_test_ext_pred[X_test_ext[ville] > 0]
        y_test_aux = y_tes_ext.loc[x_test_aux.index]
                        
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
#plt.bar(indi+space, score_global, width=space, color='b', label='CLUSTER global')
plt.bar(indi+space, score_indiv, width=space, color='g', label='ML Individuel')
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
#plt.bar(indi+space, f1_global, width=space, color='b', label='CLUSTER global')
plt.bar(indi+space, f1_indiv, width=space, color='g', label='ML Individuel')
plt.xticks(indi, villes, rotation=90, fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=15)
plt.ylim(0.3,1)
plt.grid(True, axis='y', alpha=0.8)
plt.ylabel('F1')
plt.show()

