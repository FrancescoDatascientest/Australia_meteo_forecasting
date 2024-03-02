import streamlit as st

from pages import page_introduction, page_conclusion, page_jeu_de_donnees, page_predict_raintomorrow, \
    page_predict_rain_long_horizon, page_predict_maxtemp, page_preprocessing_feature_engineering, \
    page_missing_values, page_new_features

from backend.streamlit_backend_sb import ProjetAustralieSoutenance

# affichage large
st.set_page_config(layout="wide")

# chargement des données et instanciation des classes de backend
pas = ProjetAustralieSoutenance()

st.title("Prévision météo en Australie")

st.sidebar.title("Météo en Australie")

pages = ["Introduction",
         "Le jeu de données",
         "Valeurs manquantes",
         "Pre-processing",
         "Ajout de features",
         "Prédiction de RainTomorrow",
         "Prédiction sur horizon de temps",
         "Prédiction de MaxTemp",
         "Conclusion"]
page = st.sidebar.radio("Aller vers ", pages)

# Info box
st.sidebar.info(
    '''
    Réalisé par:
    
        Sophie BERTHIER
        
        Luciano LANGHI
        
        Quyen THIEU MARCAUD
        
    Sous la supervision de: Francesco MADRISOTTI de Datascientest.
    '''
)

st.header(page)

# Introduction à la page 1
if page == pages[0]:
    page_introduction.app(pas)

elif page == pages[1]:
    page_jeu_de_donnees.app(pas)

elif page == pages[2]:
    page_missing_values.app(pas)

elif page == pages[3]:
    page_preprocessing_feature_engineering.app(pas)

elif page == pages[4]:
    page_new_features.app(pas)

elif page == pages[5]:
    page_predict_raintomorrow.app()

elif page == pages[6]:
    page_predict_rain_long_horizon.app(pas)

elif page == pages[7]:
    page_predict_maxtemp.app(pas)

elif page == pages[8]:
    page_conclusion.app()

# si pas de page valide selectionee, on reviens sur l'intro
else:
    st.write("**Point de menu non relié (" + page + ")**")
    page_introduction.app(pas)
