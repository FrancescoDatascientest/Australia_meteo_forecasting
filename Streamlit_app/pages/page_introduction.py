# -*- encoding: utf-8 -*-

import streamlit as st

def app(pas):
    
    st.markdown(
        """
        <style>
    .big-font {
        font-size:20px !important; 
        }
    </style>
    
    <div class='big-font'>
        Ce projet consiste à prédire des variables météorologiques à partir d’un jeu de données contenant dix ans de
         relevés sur de nombreuses stations météo australienne.
    </div>""", unsafe_allow_html=True
    )

    # ajouter lien Kaggle
    st.markdown(
        """   
    <div class='big-font'>
        Les données utilisées pour le développement du modèle sont disponibles sur Kagle pour le projet "Rain in Australia", voir le lien ci-dessous.
        
    </div>""", unsafe_allow_html=True
    )
    st.markdown(f'<a href="{"https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package"}" target="_blank">lien base de données</a>', unsafe_allow_html=True)
    
    ##
   
    
    st.image('img/sc_project.jpg', width=768) #, use_column_width=True)
    
    
    # l'Australie, c'est grand !
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """   
    <div class='big-font'>
        L'Australie, un vaste pays aux paysages diversifiés comme des déserts, forêts tropicales et côtes, peut contenir l'Europe 1,3 fois. Cette comparaison souligne son immense étendue et la richesse de sa géographie.
        </div>""", unsafe_allow_html=True
    )
    st.image('img/carte-australie-europe.jpg',  width=768) #, use_column_width=True)
    
    
    # evolution de RainToday => peu de cohérence géo et temporelle
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """   
    <div class='big-font'>
        Le modèle a été entraîné à partir d'observations météorologiques sur la période allant du 01/11/2007 au 25/06/2017. Vous trouverez ci-dessous une animation de la répartition des précipitations sur cette période. elle montre que les villes du centre de l'australie, situées dans la zone sèche, ont beaucoup moins de jours de pluie que le reste du pays.
        </div>""", unsafe_allow_html=True
    )
    fig = pas.animation_variable("RainToday")
    st.plotly_chart(fig)
