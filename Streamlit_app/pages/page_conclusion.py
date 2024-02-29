# -*- encoding: utf-8 -*-

import streamlit as st


def app():

    st.markdown(
        """
        ### Constats
        
        1. **Robustesse de XGBoost**: XGBoost s'est avéré être un modèle robuste et efficace pour prédire la météo 
        australienne, rivalisant même avec des réseaux de neurones plus complexes, tout en étant rapide à entraîner.

        2. **Performances malgré le déséquilibre des classes**: Malgré un déséquilibre dans les classes de la variable 
        cible, la plupart des modèles ont présenté de bonnes performances.
        
        3. **Apport limité du feature engineering**: Le feature engineering a montré un intérêt limité dans ce contexte,
        avec un écart de performances faible entre les données initiales et celles après plusieurs mois de manipulation. 
        Cela souligne l'importance de budgéter correctement le temps consacré à cette tâche.
        
        4. **Prédiction à partir d'une seule journée**: Une surprise intéressante a été la capacité de prédire la pluie 
        pour toute une année à partir d'une seule journée d'observation, bien que les performances restent faibles.

        ### Limites et perspectives 
        
        1. **Performance finale mitigée**: Bien que satisfaisants, les performances finales du meilleur modèle ne 
        répondent pas entièrement aux attentes, avec une précision de seulement 86,6% sur l'ensemble de l'Australie,
         n raison notamment du grand nombre de jours sans pluie.
        
        2. **Pistes d'amélioration**: Plusieurs pistes d'amélioration ont été identifiées, notamment la collecte de 
        données manquantes telles que l'indice de Sunshine, ainsi que l'exploration de l'ajout de données sur l'humidité
        et la pression atmosphérique à d'autres heures de la journée, en tenant compte des différents fuseaux horaires 
        en Australie.        
        """
    )