# -*- encoding: utf-8 -*-

import streamlit as st
import matplotlib.pyplot as plt


def app(pas):
    
    ##Valeurs Extremes
    st.markdown(
        """
        <style>
     .big-fontplus, .big-fontplus ul, .big-fontplus li {
         font-size:25px !important; 
     }
     </style>
     
     <style>
  .big-font, .big-font ul, .big-font li {
      font-size:20px !important; 
  }
  </style>
    
    <div class='big-fontplus'>
        Valeurs extrêmes
    </div>
    
    <div class='big-font'>
        Comme on le voit dans le graphique suivant, les valeurs extrêmes sont des outliers mathématiques, mais pertinentes pour les échelles et types de données météorologiques.
        Nous faisons donc le choix de conserver l’intégralité des outliers après les avoir analysés.
    </div>""", unsafe_allow_html=True    
    )
    
    fig = pas.generer_boxplots()
    st.pyplot(fig)
    
    
    
    ##Supresion Observations
    st.markdown("<br><br>", unsafe_allow_html=True)    
    st.markdown(
        """    
    <div class='big-fontplus'>
       Suppression des observations
    </div>
    
    <div class='big-font'>
    Nous avons adopté une approche progressive pour la suppression des lignes contenant des données manquantes. 
    <ul>
            <li>Suppression des lignes avec des données manquantes pour la variable cible (représentant 2.2% de l’ensemble de données).</li>
            <li>Suppression des lignes avec une forte proportion de données manquantes.</li> 
    </ul> 
        
    </div>""", unsafe_allow_html=True    
    )
    
    
    
    
    ##Traitement de valeur manquantes
    st.markdown("<br><br>", unsafe_allow_html=True)    
    st.markdown(
        """
        <style>
    .big-font2 {
        font-size:24px !important; 
        }
    </style>
    
    <div class='big-font2'>
       Traitement des valeurs manquantes avec KNN_Imputer.
    </div>
    
    <div class='big-font'>
        Les valeurs manquantes ont été traitées avec la méthode KNN Imputer, basée sur l'approximation des k plus proches voisins. Ce processus remplace chaque donnée absente par une moyenne pondérée de ses voisins les plus proches, assurant ainsi une imputation adaptée et efficace.
        La image suivante illustre les distributions des variables avant et après l'imputation via la méthode KNN, tout en les comparant aux méthodes traditionnelles de moyenne et médiane, lesquelles, bien que plus simples, se révèlent être moins précises.
    </div>
    
    """, unsafe_allow_html=True)
    
    
    fig = pas.generate_density_plots()
    st.pyplot(fig)
    
    
    ##Transformation des données
    st.markdown("<br><br>", unsafe_allow_html=True)    
    st.markdown(
        """
        <style>
    .big-font2 {
        font-size:24px !important; 
        }
    </style>
    
    <div class='big-font2'>
       Transformation des données.
    </div>
    
    <div class='big-font'>
        Trois variables catégorielles, représentant la direction du vent avec 16 options, sont présentes. L'encodage OneHot transformerait celles-ci en 45 variables distinctes, un volume important. En alternative, l'approche trigonométrique propose de résumer l'information en 6 variables numériques basées sur les composantes X et Y, calculées à partir du cosinus et du sinus des angles du vent.
    </div>
    
    """, unsafe_allow_html=True)
    
    st.image('img/Vent_conv.png', width=1200)
 
    
   
    ##Traitement KNN_imputer
    
    
    
    
    
