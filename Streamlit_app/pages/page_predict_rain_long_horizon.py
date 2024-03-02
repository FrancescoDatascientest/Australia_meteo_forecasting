# -*- encoding: utf-8 -*-

import streamlit as st
import matplotlib.pyplot as plt
import pickle

def app(pas):

    st.subheader("Principe")
    
    st.write("- Ajout de variables cibles : 'RainToday' avec shift de n jours")
    st.write("- Entrainement d'un XGBoost par décalage de prévision de la pluie")

    st.subheader("Métriques des prévisions sur 15 jours")
    #st.image('img/horizon_perfs.png', use_column_width=True)
    pas.affiche_perfs_rainj_macro(nbj=15)
    st.pyplot(plt.gcf())

    st.subheader("AUC par zone climatique")   
    #st.image('img/horizon_auc_climat.png', use_column_width=True)
    pas.AUC_par_climat(nbj=15)
    st.pyplot(plt.gcf())
    pas.AUC_par_climat(nbj=360)
    st.pyplot(plt.gcf())
    
    st.subheader("Test X² par zone climatique")   
    #st.image('img/horizon_pvalue_climat.png', use_column_width=True)
    show_plot = st.checkbox("Afficher les tests X²")
    if show_plot:
        pas.affiche_pvalue_rainj_climats()
        st.pyplot(plt.gcf())

    st.subheader("Seuil de probabilité optimal")
    
    col1, col2 = st.columns(2)

    with open('img/auc_australie.pkl', 'rb') as f:
        reloaded_figure = pickle.load(f)
    col1.pyplot(reloaded_figure)
    
    with open('img/auc_uluru.pkl', 'rb') as f:
        reloaded_figure = pickle.load(f)
    col2.pyplot(reloaded_figure)
    
    st.subheader("Prévisions sur une année, uniquement à partir des données du 4 janvier 2016")
    #st.image('img/horizon_pred_darwin.png', use_column_width=True)

    with open('img/horz2016_Darwin_climat.pkl', 'rb') as f:
        reloaded_figure = pickle.load(f)
    st.pyplot(reloaded_figure)

    with open('img/horz2016_Adelaide_micro.pkl', 'rb') as f:
        reloaded_figure = pickle.load(f)
    st.pyplot(reloaded_figure)
    
    st.subheader("Interprétabilité")   
    # st.image('img/horizon_explicabilite.png', use_column_width=True)
    
    col1, col2 = st.columns(2)
    with open('img/fi_Australie_Rain_J_15.pkl', 'rb') as f:
        reloaded_figure = pickle.load(f)
    col1.pyplot(reloaded_figure)
    
    #with open('img/fi_Darwin_Rain_J_15.pkl', 'rb') as f:
    #    reloaded_figure = pickle.load(f)
    #col2.pyplot(reloaded_figure)

    with open('img/fi_Darwin_Rain_J_100.pkl', 'rb') as f:
        reloaded_figure = pickle.load(f)
    col2.pyplot(reloaded_figure)
    