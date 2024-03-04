# -*- encoding: utf-8 -*-

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
#import pickle

def app(pas):
    st.subheader("Décomposition")
    
    #st.image('img/maxtemp_decomposition.png', use_column_width=True)

    st.write("- A partir de mars 2013 uniquement: pas de données pour les mois d'avril 2011, décembre 2012 et février 2013")

    locations = pas.df.Location.unique()
    
    # Histogramme temp / pluviometrie
    option = st.selectbox(
        'Sélectionnez une Location',
        (locations)
    )

    pas.prepare_serie_temporelle(location=option)
    st.pyplot(plt.gcf())
    """    
    st.subheader('Observations')
    st.line_chart(season_dec.observed)
    
    st.subheader('Tendance')
    st.line_chart(season_dec.trend)
    
    st.subheader('Saisonnalité')
    st.line_chart(season_dec.seasonal)
    
    st.subheader('Résidus')
    st.scatter_chart(season_dec.resid)
    
    st.pyplot(season_dec)
    """    
    
    st.subheader("RNN monovarié")
    
    st.write("- Une première couche cachée de 30 neurones LSTM, avec une fonction d’activation ReLU")
    st.write("- Une seconde couche cachée de 10 neurones LSTM, avec une fonction d’activation ReLU")
    st.write("- Une couche dense de sortie de 1 neurone, sans fonction d’activation")
    st.write("- Loss: MSE")
    st.write("- Batch size: 1 (pour meilleure qualité)")
    st.write("- Fenêtre: 15 jours précédents")
    
    #st.image('img/maxtemp_multi_loss.png', use_column_width=True)
    #st.image('img/maxtemp_multi_pred.png', use_column_width=True)
    df_resultats_rnn_mono = pd.read_csv("data/rnn_resultats_Mildura_mono.csv", index_col=0)
    df_resultats_rnn_mono.index = pd.to_datetime(df_resultats_rnn_mono.index)
    trace_pred_rnn(df_resultats_rnn_mono.loc['2014-03-01':], "monovariée, Mildura")


    st.subheader("RNN multivarié")
    
    st.write("- Seconde couche cachée de 100 neurones LSTM")
    st.write("- Insertion d'une troisième couche cachée dense de 100 neurones, avec une fonction d’activation ReLU")
    
    
    #st.image('img/maxtemp_mono_loss.png', use_column_width=True)
    #st.image('img/maxtemp_mono_pred.png', use_column_width=True)
    df_resultats_rnn_multi = pd.read_csv("data/rnn_resultats_Mildura_multi.csv", index_col=0)
    df_resultats_rnn_multi.index = pd.to_datetime(df_resultats_rnn_multi.index)
    trace_pred_rnn(df_resultats_rnn_multi.loc['2014-03-01':], "multivariée, Mildura")    

    st.subheader("Comparaison mono/multivarié (Mildura, MaxTemp, 15 jours)")
    #st.image('img/maxtemp_multi_comp.png', use_column_width=True)

    df_comparaison = pd.read_excel("data/Comparaison_RNN.xlsx", index_col=0)
    df_comparaison = df_comparaison.style.highlight_min(subset=df_comparaison.columns[0:], axis=0)
    st.table(df_comparaison)

    st.subheader("Prédictions itératives")
    #st.image('img/maxtemp_mono_futur.png', use_column_width=True)

    """
    st.subheader("Fenêtre de 7 jours")
    with open('img/rnn_iterative_07.pkl', 'rb') as f:
        reloaded_figure = pickle.load(f)
    st.pyplot(plt.gcf())        

    st.subheader("Fenêtre de 15 jours")
    with open('img/rnn_iterative_15.pkl', 'rb') as f:
        reloaded_figure = pickle.load(f)
    st.pyplot(plt.gcf())        

    st.subheader("Fenêtre de 30 jours")
    with open('img/rnn_iterative_30.pkl', 'rb') as f:
        reloaded_figure = pickle.load(f)
    st.pyplot(plt.gcf())        
    """
    
    df_rnn2016 = pd.read_csv("data/rnn_pred2016.csv", index_col=0)    
    df_rnn2016.index = pd.to_datetime(df_rnn2016.index)
    
    plt.figure(figsize=(16, 4))
    plt.plot(df_rnn2016.reel_unscaled, label="Donnees reelles")
    plt.plot(df_rnn2016.pred_unscaled_07, label="Predictions incrémentales - Fenêtre de 7 jours")
    plt.plot(df_rnn2016.pred_unscaled_15, label="Predictions incrémentales - Fenêtre de 15 jours")
    plt.plot(df_rnn2016.pred_unscaled_30, label="Predictions incrémentales - Fenêtre de 30 jours")
    plt.legend()
    plt.title("Prediction itérative des températures sur période non vue à l'entraînement")  
    st.pyplot(plt.gcf())
    
    
def trace_pred_rnn(df:pd.DataFrame, titre:str):
        plt.figure(figsize=(25, 5))
#        plt.figure(figsize=(30, 6))
        plt.plot(df.train_orig_unscaled, label="Train (réel)", alpha=.75)
        plt.plot(df.train_pred_unscaled, label="Train (prédictions)", alpha=.75)
        plt.plot(df.validation_orig_unscaled, label="Test (réel)", alpha=.75)
        plt.plot(df.validation_pred_unscaled, label="Test (prédictions)", alpha=.75)
        #plt.xticks(rotation=90, ha='right')
        plt.title("Prédictions avec RNN sur MaxTemp\napproche "+titre)
        plt.legend()
        
        st.pyplot(plt.gcf())
        #st.line_chart(df, height=500)