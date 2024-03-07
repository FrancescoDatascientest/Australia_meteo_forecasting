# -*- encoding: utf-8 -*-

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

    
def app(pas):
     
    st.markdown(
        """
       <style>
    .big-font, .big-font ul, .big-font li {
        font-size:20px !important; 
    }
    </style>
    
    <div class='big-font'>
        La base de données est composée de 22 variables qui sont divisées en 3 types:
        <ul>
            <li>4 variables catégorielles : Location, WindGustDir, WindDir9am et WindDir3pm</li>
            <li>2 variables booléennes : RainToday, RainTomorrow</li>
            <li>16 variables numériques.</li>
        </ul>
    </div>""", unsafe_allow_html=True)


    st.dataframe(pas.df.describe())

    # RainTomorrow vs ville
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
         """
         <div class='big-font'>         
             Une disparité significative est observée dans la variable cible en fonction de la localisation, commençant à 6% pour Woomera et se terminant à environ 35% pour Portland.
         </div>""", unsafe_allow_html=True
     )  
    pas.grap_rain_bar_ville()
    st.pyplot(plt.gcf(), use_container_width=True)



    # géographie - raintomorrow / MaxTemp
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
         """   
     <div class='big-font'>
         La diversité des climats en Australie se reflète également dans d'autres variables telles que la température maximale. Dans l'image suivante, vous pouvez voir les différences de température maximale moyenne par ville en fonction de la couleur, et la fréquence des précipitations en fonction du diamètre.
         </div>""", unsafe_allow_html=True
     )
    fig = pas.synthetise_villes()
    st.plotly_chart(fig)

    locations = pas.df.Location.unique()
    
    
    
    # Histogramme temp / pluviometrie
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
         """   
     <div class='big-font'>
         En raison de sa position géographique, s'étendant des tropiques jusqu'aux latitudes tempérées du sud, l'Australie connaît des variations significatives de température moyenne au fil des saisons, comme le montre ce graphique.
         </div>""", unsafe_allow_html=True
     )
    option = st.selectbox(
        'Sélectionnez une Location',
        (locations)
    )
    pas.histogramme_temperatures_precipitations(option)
    st.pyplot(plt.gcf(), use_container_width=True)  
    
    
    
    # Direction des vents
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
         """   
     <div class='big-font'>
        La figure suivante montre la distribution des directions du vent: WinDir9am, WindDir3pm, et WindGustDir, représentant la direction à 9h, 15h et la rafale la plus forte, respectivement.
        </div>""", unsafe_allow_html=True
     )
    location_aus = np.append('', locations)
    option = st.selectbox(
        'Sélectionnez une Location',
        (location_aus)
    )   
    pas.graphe_vent(option)
    st.pyplot(plt.gcf())  
    
    
    
    
    # Matrice Correlation
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
         """   
     <div class='big-font'>   
     La figure ci-dessus représente la corrélation entre les variables numériques et RainTomorrow, montrant des liens variables. Observons que :
     <ul>
         <li>Aucune variable quantitative n'est fortement corrélée avec RainTomorrow, les corrélations se situant majoritairement entre 0,25 et 0,5 pour Sunshine, Humidity3pm, Humidity9am, Cloud9am, Cloup9pm, RainToday.</li>
         <li>Malheureusement Cloud9am et Cloud9pm présentent un fort pourcentage de données manquantes.</li>
         <li>Les corrélations avec les températures sont faibles, variant de 0,03 à 0,19.</li>
         <li>Il existe des corrélations fortes entre d'autres variables, notamment MaxTemp avec Temp3pm, MinTemp avec Temp9am et les pressions mesurées à 3am et 9p, soulignant une consistance dans les mesures de température au cours de la journée.</li>         
     </ul>
         
     </div>""", unsafe_allow_html=True
     )
    # pas.matrice_corr_quyen(pas.df, "Corrélations entre les variables du dataset initial")
    # st.pyplot(plt.gcf()) 
    st.image('img/Corr_jeu_donnes.png', width=1300)




