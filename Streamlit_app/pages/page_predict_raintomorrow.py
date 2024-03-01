# -*- encoding: utf-8 -*-
import pickle
from json import dump

import shap
import streamlit as st

import os

import pandas as pd
import xgboost as xgb
from joblib import load
from matplotlib import pyplot as plt
from sklearn import ensemble, tree, metrics
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder
from streamlit import components

print(os.getcwd())

path_data = "./data/RainTomorrow"
list_metric_considered = ['accuracy', 'recall', 'precision', 'f1', 'roc_auc']
list_model = ["LogisticRegression", "TreeDecision", "RandomForest", "XGBoost"]

k = 4
path_classification = path_data
data_original = f"{path_data}/data_original_pre_processing_without_location_knn_imputed_{k}.joblib"
data_scaled = f"{path_data}/data_scaled_pre_processing_without_location_knn_imputed_{k}.joblib"


class MachineLearningModels:
    def __init__(self, path_data):
        self.path_classification = path_data
        self.data_original = f"{path_data}/data_original_pre_processing_without_location_knn_imputed_{k}.joblib"
        self.data_scaled = f"{path_data}/data_scaled_pre_processing_without_location_knn_imputed_{k}.joblib"

    def get_results(self, metric_name):
        file_scores = f"{self.path_classification}/Scores_of_modeles_best_{metric_name}.csv"
        file_rocauc_rain = f"{self.path_classification}/ROCCurve_of_Rain_of_modeles_best_{metric_name}.png"
        file_rocauc_no_rain = f"{self.path_classification}/ROCCurve_of_No_Rain_of_modeles_best_{metric_name}.png"

        df_scores = pd.read_csv(file_scores, sep=";", index_col=0)
        del df_scores["Model"]
        return df_scores, file_rocauc_rain, file_rocauc_no_rain

    def graph_mean_shap_xgboost(self, metric_name):
        X_train, X_test, y_train, y_test = load(data_original)

        file_best_model = f"{path_classification}/xgboost/best_model_{metric_name}.joblib"
        xgb_best = load(file_best_model)

        shap_explainer = shap.TreeExplainer(xgb_best)
        shap_values = shap_explainer(X_test)

        # =============================================================================================
        # SHAP - Mean SHAP Plot
        # =============================================================================================

        fig_mean_shap = plt.figure(figsize=(8, 12))
        shap.plots.bar(shap_values, max_display=20, show=False)
        plt.title(f'Mean Shap XGBoost for {metric_name}', fontsize=20)
        fig_mean_shap.tight_layout()

        # =============================================================================================
        # SHAP - Beeswarm Plot
        # =============================================================================================
        # print("bees")
        fig_beeswarm = plt.figure(figsize=(8, 20))
        shap.plots.beeswarm(shap_values, max_display=20, show=False)
        plt.title(f'Beeswarm XGBoost for {metric_name}', fontsize=20)
        fig_beeswarm.tight_layout()

        # =============================================================================================
        # SHAP - Dependence Plots
        # =============================================================================================

        list_variables = ["Humidity3pm", "Pressure3pm", "WindGustSpeed", "Sunshine"]
        fig_dependences, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 11))
        axes = axes.ravel()
        for i, variable in enumerate(list_variables):
            shap.plots.scatter(shap_values[:, variable], ax=axes[i])
            axes[i].set_title(variable)
        fig_dependences.suptitle("Dépendence de la valeur SHAP à une seule variable", fontsize=20)
        plt.tight_layout()

        return fig_mean_shap, fig_beeswarm, fig_dependences

    def predict(self, metric_name, s):
        X_train, X_test, y_train, y_test = load(data_original)
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        file_best_model = f"{path_classification}/xgboost/best_model_{metric_name}.joblib"
        xgb_best = load(file_best_model)

        shap_explainer = shap.TreeExplainer(xgb_best)
        shap_values = shap_explainer(X_test)
        fig_waterfall = plt.figure(figsize=(6, 10))
        shap.plots.waterfall(shap_values[s], max_display=10, show=False)

        if s == 0:
            title = f"RainTomorrow prédit = 0, RainTomorrow observé = 0"
        elif s == 1:
            title = f"RainTomorrow prédit = 1, RainTomorrow observé = 1"
        else:
            title = "Waterfall Plot"

        plt.title(title, fontsize=20)
        fig_waterfall.tight_layout()
        is_raintomorrow = True if y_test_encoded[s] == 1 else False
        print(is_raintomorrow)
        return fig_waterfall, is_raintomorrow

# ================================================================================================
# Le bloc ci-dessous sert à générer les fichiers de résultats pour l'application
# ================================================================================================
# import pickle
# metric_name = "roc_auc"
# mlm = MachineLearningModels(path_data)
#
# fig_mean_shap, fig_beeswarm, fig_dependences = mlm.graph_mean_shap_xgboost(metric_name)
# with open(f"{path_classification}/xgboost/mean_shap_{metric_name}.pickle", 'wb') as f:
#     pickle.dump(fig_mean_shap, f)
# with open(f"{path_classification}/xgboost/beeswarm_{metric_name}.pickle", 'wb') as f:
#     pickle.dump(fig_beeswarm, f)
# with open(f"{path_classification}/xgboost/dependence_{metric_name}.pickle", 'wb') as f:
#     pickle.dump(fig_dependences, f)
#
# fig_warterfall, is_raintomorrow = mlm.predict(metric_name, s=0)
# with open(f"{path_classification}/xgboost/prediction_{is_raintomorrow}_example_{metric_name}.pickle", 'wb') as f:
#     pickle.dump(fig_warterfall, f)
#
# fig_warterfall, is_raintomorrow = mlm.predict(metric_name, s=1)
# with open(f"{path_classification}/xgboost/prediction_{is_raintomorrow}_example_{metric_name}.pickle", 'wb') as f:
#     pickle.dump(fig_warterfall, f)

# ================================================================================================
#
# ================================================================================================
sous_pages = ["***Approche de Machine Learning***",
              "***Approche de Deep Learning***"]


def set_sous_pages():
    page = st.sidebar.radio("", sous_pages)
    st.header(page)
    st.sidebar.markdown(page)
    return page


class ApprocheMachineLearning:
    @staticmethod
    def show_different_ML_models():
        st.markdown(
            '''
            ## 1. Modèles classiques de classification du Machine Learning

            - Modèles utilisés: **Logistic Regression**, **Decision Tree**, **Random Forest** et **XGBoost**.

            - Données utilisés: toutes les observations après avoir traité les données manquantes par KNN imputation

            - Hyperparamètres tunning: Les hyperparamètres de chaque modèle seront optimisés via des tests manuels,
             des GridSearch, mais aussi à l’aide de la bibliothèque Hyperopt, en cherchant à maximiser diverses 
             métriques telles que l’*accuracy*, la *précision*, le *recall*, le score *F1* et le *ROC AUC*.

            '''
        )

        metric_name = st.selectbox(
            "***Quelle métrique de classification que vous voulez optimiser?***",
            list_metric_considered,
            index=len(list_metric_considered) - 1
        )

        mlm = MachineLearningModels(path_data)
        df_scores, file_rocauc_rain, file_rocauc_no_rain = mlm.get_results(metric_name)

        col1, col2 = st.columns(2)

        col1.write("\n\n\n\n\n\n\n\n\n")
        col1.write(f"Les scores des modèles en optimisant le {metric_name}")
        col1.dataframe(df_scores.style.highlight_max(axis=0))
        col2.image(file_rocauc_rain, width=500)


    @staticmethod
    def show_XGBoost_different_levels():
        st.markdown(
            '''
            ## 2. Modèle XGBoost avec trois niveau de finesse
            
            -	Un niveau **macro**: la modélisation est réalisée en utilisant l’ensemble des données australiennes 
            du jeu de données -> un seul modèle de prédiction pour tout l'Australie
            -	Un niveau **micro**: pour chaque *Location*, la modélisation est réalisée en utilisant que les données
            collectées à cette *Location* -> 49 modèles de prédiction
            -	Un niveau **intermédiaire**: pour chaque zone climatique, la modélisation est réalisée en utilisant que 
            les données collectées à cette cette zone -> 7 modèles de prédiction
            
            Comparons maintenant les performances d’un XGBoost entraîné en optimisant l’*AUC-ROC* aux trois niveaux 
            en-dessus.
            
            '''
        )
        fichier = f"{path_classification}/xgboost/Comparaison_XGBoost_3niveaux.xlsx"
        df_comparaison = pd.read_excel(fichier)
        st.dataframe(df_comparaison)

        st.markdown(
            '''
            De façon plus détaillée, observons, dans la Figure 40, l’accuracy pour chacun des 49 lieux en comparant :
            
            -	Un modèle global, entraîné sur l’ensemble du dataset (« ML Global »)
            -	Un modèle local, entraîné spécifiquement sur les données du lieu concerné (« ML Individuel »)
            -	Un modèle naïf se contentant de prédire qu’il ne pleuvra jamais, qui nous permettra de relativiser les 
            scores obtenus par les deux premiers modèles (« Pred Bas »)

            '''
        )
        fichier = f"{path_classification}/xgboost/Accuracy_des_trois_modeles_par_Location.png"
        st.image(fichier, caption=f"L'accuracy des trois modèles par Location", width=800)

    @staticmethod
    def show_SHAP_XGBoost():

        st.markdown(
            '''
            ## 3. Interprétabilité et explicabilité du modèle XGBoost
            Nous utilisons **SHAP**, une méthode d'interprétation mesurant l'impact des variables explicatives sur les
            prédictions du modèle XGBoost.
            
            ### 3.1. L'interprétabilité avec SHAP
            Nous allons maintenant montrer comment SHAP nous permet concrètement d’interpréter les résultats d’un
            algorithme XGBoost.
            '''
        )
        col1, col2, col3 = st.columns(3)

        metric_name = "roc_auc"

        with open(f"{path_classification}/xgboost/mean_shap_{metric_name}.pickle", 'rb') as f:
            fig_mean_shap = pickle.load(f)
        with open(f"{path_classification}/xgboost/beeswarm_{metric_name}.pickle", 'rb') as f:
            fig_beeswarm = pickle.load(f)
        with open(f"{path_classification}/xgboost/dependence_{metric_name}.pickle", 'rb') as f:
            fig_dependences = pickle.load(f)

        col1.pyplot(fig_mean_shap)
        col2.pyplot(fig_beeswarm)
        col3.pyplot(fig_dependences)

        st.markdown(
            '''
            On constate que:
            - Une augmentation de la valeur de *Humidity3pm* aujourd'hui indique une probabilité accrue de pluie pour 
            demain.
            - De même, des vents forts aujourd'hui sont associés à une probabilité plus élevée de pluie demain.
            - En revanche, une pression atmosphérique élevée aujourd'hui est généralement corrélée à une diminution de 
            la probabilité de pluie pour demain.
            '''
        )
        st.markdown(
            '''
            ### 3.2 L'explicabilité avec SHAP
            Dans les graphes ci-dessous, on peut voir l’impact de chacune des caractéristiques de l’observation choisie
            et comment ces caractéristiques impactent la prédiction de la pluie du lentement. 
            '''
        )

        with open(f"{path_classification}/xgboost/prediction_False_example_{metric_name}.pickle", 'rb') as f:
            fig_false = pickle.load(f)
        with open(f"{path_classification}/xgboost/prediction_True_example_{metric_name}.pickle", 'rb') as f:
            fig_true = pickle.load(f)

        col1, col2 = st.columns(2)
        col1.pyplot(fig_false)
        col2.pyplot(fig_true)


    @staticmethod
    def show_prediction_with_XGBoost():
        st.markdown(
            '''
            ## 4. Prédiction example avec XGBoost
            
            '''
        )

        st.markdown("Vous pouvez faire des prédiction avec vos propres données")
        input_file = f"{path_classification}/xgboost/meteo_example.csv"
        st.markdown(f"""[Example CSV input file]({input_file})""")

        # Collects user input features into dataframe
        uploaded_file = st.file_uploader("Charger votre input CSV file", type=["csv"])
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            st.dataframe(input_df)


    def show(self):
        # Résultats des différents modèles de Machine Learning
        self.show_different_ML_models()
        # Résultats du modèle XGBoost aux 3 niveaux de finesse
        self.show_XGBoost_different_levels()
        # Interprétabilité du modèle XGBoost
        self.show_SHAP_XGBoost()
        # Prédiction example
        self.show_prediction_with_XGBoost()


class ApprocheDeepLearning:
    @staticmethod
    def show_parametres_tunning():
        st.markdown(
            '''
            Nous utilisons un réseau de neuronnes profond (Deep Neural Network) pour la prédiction de la variable 
            *RainTomorrow*.
            
            ### Détermination de la structure du modèle DNN
            
            Pour structurer notre modèle DNN, nous avons exploré différentes configurations en ajustant le nombre de 
            couches et de neurones, ainsi que le taux d'apprentissage et les fonctions d'activation. Les tests ont 
            révélé que l'utilisation de couches cachées avec un nombre de neurones varié impacte significativement les 
            performances du modèle. L'ajustement du taux d'apprentissage via une callback a permis d'optimiser la 
            convergence du modèle, tandis que l'expérimentation avec différentes fonctions d'activation a montré des 
            effets sur la dispersion de l'apprentissage.

            En conclusion, nous opterons pour un réseau comportant une première couche de 50 neurones activés par *Tanh*,
            suivie d'une seconde couche de 50 neurones activés par *ReLU*.
            
            ### Entraînement du modèle
            
            L'entraînement du modèle nécessite une approche empirique étant donné l'absence de moyen analytique pour 
            déterminer les hyperparamètres optimaux tels que le nombre d'epochs, l'optimizer ou la taille de batch. 
            Nous avons exploré différentes configurations en faisant varier un hyperparamètre à la fois, constatant que 
            le modèle avec un batch_size de 16 était le plus performant mais aussi le plus lent. Malgré une légère 
            baisse de la loss à chaque changement de learning rate, nous préférons opter pour un batch_size de 128 en
            raison de son temps d'entraînement plus raisonnable et de l'écart moindre entre les échantillons de train 
            et de test.
            
            Nous examinons les courbes d'apprentissage de notre meilleur modèle DNN, caractérisé par 
            - une première couche cachée de 50 neurones avec activation tanh, 
            - une seconde couche cachée de 50 neurones avec activation ReLU, 
            - un entraînement sur 300 époques.
            - un learning rate dynamique via une callback personnalisée, 
            - un batch size de 128, 
            '''
        )
        fichier = f"{path_classification}/Comparaison_XGBoost_DNN.xlsx"
        df_comparaison = pd.read_excel(fichier)
        df_comparaison = df_comparaison.style.highlight_max(subset=df_comparaison.columns[1:], axis=0)
        st.dataframe(df_comparaison)

    def show(self):
        self.show_parametres_tunning()


# La fonction principale qui est appellée dans streamlit_app.py pour afficher la page
def app():
    sous_page = set_sous_pages()

    if sous_page == sous_pages[0]:
        ApprocheMachineLearning().show()

    if sous_page == sous_pages[1]:
        ApprocheDeepLearning().show()
