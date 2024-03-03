# -*- encoding: utf-8 -*-

import streamlit as st
import matplotlib.pyplot as plt

def app(pas):
    
    st.markdown(
        """
        <style>
     .big-font, .big-font ul, .big-font li {
         font-size:20px !important; 
     }
     </style>
    
    <div class='big-font'>
        L'identification des valeurs manquantes est cruciale dans l'analyse de données, car elle impacte l'exactitude des résultats. 
        Dans notre projet, nous constatons un nombre significatif de données absentes pour certaines variables, comme le montre le graphique suivant
    </div>""", unsafe_allow_html=True
    )
    pas.graphique_valeurs_manquantes()
    st.pyplot(plt.gcf())
    
    
    st.markdown("<br>", unsafe_allow_html=True)    
    st.markdown(
        """
          
    <div class='big-font'>
        En ajoutant à l'analyse la répartition de ces données manquantes par ville, on observe que plusieurs villes n'ont aucune valeur pour certaines variables.
    </div>""", unsafe_allow_html=True
    )
    
    pas.graphe_taux_na_location_feature()
    st.pyplot(plt.gcf())
    
    
    # ajoute valeur manqantes vs temp
    
    
    