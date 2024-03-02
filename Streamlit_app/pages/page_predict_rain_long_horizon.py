# -*- encoding: utf-8 -*-

import streamlit as st
import matplotlib.pyplot as plt

def app(pas):
    st.subheader("Préambule : importance du seuil de probabilité")
    st.subheader("Ajout de variables cibles : 'RainToday' avec shift de n jours")

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
    pas.affiche_pvalue_rainj_climats()
    st.pyplot(plt.gcf())
    
    st.subheader("Prévisions sur une année")      
    st.image('img/horizon_pred_darwin.png', use_column_width=True)
    
    st.subheader("Explicabilité")   
    st.image('img/horizon_explicabilite.png', use_column_width=True)
    