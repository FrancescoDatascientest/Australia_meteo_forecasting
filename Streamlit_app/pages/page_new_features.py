# -*- encoding: utf-8 -*-

import streamlit as st
import matplotlib.pyplot as plt


def app(pas):
    
    st.subheader("Clusterisation par zones climatiques")
    affichage_clusterisation(pas)

    st.subheader("Nouvelles variables")

    st.write("- Zone climatique")
    st.write("- Latitude et longitude de chaque Location")
    st.write("- Amplitude thermique (=MaxTemp-MinTemp)")
    st.write("- Approche trigonométrique du jour de l'année (=cos(2 pi * numéro/365), et idem avec 4 pi)")
    st.write("- Approche trigonométrique de la direction du vent (cos et sin)")

    #st.header("Corrélations nouvelles variables")
    #pas.matrice_corr_quyen(pas.data)
    #st.pyplot(plt.gcf())  
    
    
def affichage_clusterisation(pas):
    # Clusterisation
    fig = pas.clusterisation_groupee()
    st.pyplot(plt.gcf())
    st.caption("Dendrogramme de clusterisation")

    st.plotly_chart(fig)
    st.caption("Résulat de la clusterisation en 7 zones climatiques")

    st.image('img/climats_wiki.png', caption="Climats australiens (Wikipédia)", width=512)

