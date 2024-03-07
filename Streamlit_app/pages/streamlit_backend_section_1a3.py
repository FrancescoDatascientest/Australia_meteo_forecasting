# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 16:40:35 2024

@author: langh
"""
## Copie de backend_Sophie pour adapter les graps des section 1, 2, 3.1/2 ##


import pandas as pd
import numpy as np
import streamlit as st
import os

# dataviz
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# pour couleurs
import plotly.express as px
import plotly.colors as pc

# scaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import FuncFormatter

path_datasets="data/"

class ProjetAustralieSoutenance_1a3:
   

    def __init__(self):
       self.disp_width = 1200
       self.disp_height = 800

       # donnees originelles
       self.df = pd.read_csv(path_datasets+"weatherAUS.csv")
       self._preprocessing()
       self.is_preprocessing_apres_analyse=False

       self.df_orig = self.df.copy()

       # donnees preprocessees
       
       self.X=None
       self.y=None
       #self.data = data.dropna()
       #self.data = data
       self.data = pd.read_csv(path_datasets+"data_process3_knnim_resample_J2.csv", index_col=0)
       
       self.data.index = pd.to_datetime(self.data.index)
       
       #self.data = self.data.drop(columns=["SaisonCos"])
       
       # si SaisonCos existe, alors on la renomme en 4pi et on ajoute SaisonCos2pi
       if hasattr(self.data, "SaisonCos"):
           self.data = self.data.rename(columns={'SaisonCos':'SaisonCos4pi'})
           self.data["SaisonCos2pi"] = np.cos(2*np.pi*(self.data.index.day_of_year-1)/365)
           
       #self.data = self.data.drop(columns=["SaisonCos2pi", "SaisonCos4pi"])

       # s'il n'y a que mount ginini en climat 5, on degage
       if (self.data[self.data.Climat==5].Location.nunique()==1):
           self.data = self.data[self.data.Climat!=5]
           
       # palette
       palette_set1 = px.colors.qualitative.Set1
       self.palette=[]
       for i in range(7):
           self.palette.append(pc.unconvert_from_RGB_255(pc.unlabel_rgb(palette_set1[i])))
           
       # libelle des climats
       self.lib_climats = {0:"Côte Est", 1:"Nord", 2:"Centre", 3:"Sud-Est", 4:"Intermédiaire", 5:"Mount Ginini", 6:"Côte Sud"}
       
       # evite de reclaculer au vol
       self.df_resample = self.data.copy()
       self._ajoute_colonnes_dates()


    def _preprocessing(self):
                # remplace Yes/No par 0/1
        self.df = self.df.replace({"No":0, "Yes":1})
        self.df.Date = pd.to_datetime(self.df.Date)
        self.df = self.df.set_index("Date", drop=False)
        
        
    # ajoute les proprietes des villes au DF
    def _ajoute_prop_locations(self):
        # si la colonne de la latitude a déja été ajoutée, on sort
        if 'lat' in self.df.columns:
            return
        # sinon, on charge les props des villes et on les ajoute au df
        self._charge_villes()
        self.df = pd.merge(self.df, self.df_cities, on='Location')
        self.df.index = self.df.Date
               
    
    
    def _ajoute_colonnes_dates(self):
        self.df_resample["Date"] = self.df_resample.index
        self.df_resample["Annee"] = self.df_resample.Date.dt.year
        self.df_resample["Mois"] = self.df_resample.Date.dt.month
        self.df_resample["AAMM"] = self.df_resample.Annee.astype(str)+"-"+self.df_resample.Mois.astype(str)    



    def grap_rain_bar_ville(self):
           
            # donnees originelles
            self.df2 = pd.read_csv(path_datasets+"weatherAUS.csv")
            df2 = self.df2.groupby("Location")["RainToday"].value_counts().unstack(fill_value=0)  # Usar fill_value=0 para reemplazar NaN por 0 directamente en unstack
            total = df2["No"] + df2["Yes"]
            df2_porc = df2.copy()  # Trabajar con una copia para los cálculos
            df2_porc["No"] = df2["No"] / total * 100
            df2_porc["Yes"] = df2["Yes"] / total * 100
            df2_porc = df2_porc.sort_values(by="Yes")
    
            indices = np.arange(len(df2_porc))
            esp = 0.4  # Espaciado entre barras
    
            plt.figure(figsize=(20, 7))
            plt.bar(indices, df2_porc["No"], width=esp-0.1, color='orange', label='No')
            plt.bar(indices + esp, df2_porc["Yes"], width=esp-0.1, color='royalblue', label='Yes')
            plt.xticks(indices + esp / 2, df2_porc.index, rotation=90, fontsize=13)
    
            plt.legend(title='RainTomorrow', loc='upper right', fontsize=16)
            plt.xlabel('Villes', fontsize=15)
            plt.ylabel('Pourcentage des journées (%)', fontsize=15)
            plt.title('Nombre de journées sèches et pluvieuses (=RainTomorrow) par Location', fontsize=14)
            plt.ylim(0, 100)
            plt.tick_params(axis='y', labelsize=14)
    
            valores_lineas_horizontales = [20, 40, 60, 80]  # Líneas horizontales para referencia
            for valor in valores_lineas_horizontales:
                plt.axhline(y=valor, color='gray', linestyle='--', linewidth=0.5)
    
            plt.show()


    
    
    def generer_boxplots(self):
        # Caract. quant. pour boxplots
        vars_quant = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed']
        
        # Filtrage des vars présentes dans le DF
        vars_quant = [var for var in vars_quant if var in self.df.columns]
        
        # Config fig & axes
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Style
        sns.set_theme(style="whitegrid")
        
        for i, var in enumerate(vars_quant):
            # Boxplot vertical
            sns.boxplot(y=var, data=self.df, ax=axes[i], color="maroon", width=0.5, orient='v')
            axes[i].set_title(f'Boxplot {var}')
        
        plt.tight_layout()
        return fig


    def graphique_valeurs_manquantes(self):
        self.df = pd.read_csv(path_datasets + "weatherAUS.csv")
        # Calcul des valeurs manquantes par colonne
        valeurs_manquantes = self.df.isnull().sum()
        pourcentage = (valeurs_manquantes / len(self.df)) * 100
        
        # Filtrage des colonnes avec valeurs manquantes et tri
        pourcentage = pourcentage[pourcentage > 0].sort_values(ascending=True)
        
        
        fig, ax = plt.subplots(figsize=(14, 10))
        norm = plt.Normalize(pourcentage.min(), pourcentage.max())
        cmap = plt.get_cmap("coolwarm")
        colors = cmap(norm(pourcentage.values))
        
        bars = ax.barh(pourcentage.index, pourcentage.values, color=colors)
        
        # Ajout du pourcentage à la fin
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 1.5  
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center', fontsize=9)
        
        ax.set_title('Pourcentage de valeurs manquantes avant nettoyage')
        plt.tight_layout()
        
        return fig


    def generate_density_plots(self):      
        
        # Charger dataframes
        df_cleaned_data = pd.read_csv(os.path.join(path_datasets, "weatherAUS.csv"))
        df_imputed_inverse_2 = pd.read_csv(os.path.join(path_datasets, "df_imputed_inverse_2.csv"), sep=';')
        df_imputed_inverse_3 = pd.read_csv(os.path.join(path_datasets, "df_imputed_inverse_3.csv"), sep=';')
        df_imputed_inverse_4 = pd.read_csv(os.path.join(path_datasets, "df_imputed_inverse_4.csv"), sep=';')
        df_imputed_by_mean = pd.read_csv(os.path.join(path_datasets, "df_imputed_by_mean.csv"), sep=';')
        df_imputed_by_median = pd.read_csv(os.path.join(path_datasets, "df_imputed_by_median.csv"), sep=';')

        # Définir les caractéristiques quantitatives for graph
        quantitative_features = ['Sunshine', 'Cloud9am', 'Temp3pm']  

        # Generar gráficos en deux colonnes
        fig, axes = plt.subplots(nrows=len(quantitative_features), ncols=2, figsize=(14, 22))

        sns.set(style="darkgrid")

        for i, col in enumerate(quantitative_features):
            # Columna izquierda: Imputaciones por media y mediana
            sns.kdeplot(df_cleaned_data[col], linewidth=4, fill=True, color='black', ax=axes[i, 0], label="Avant")
            sns.kdeplot(df_imputed_by_mean[col], linewidth=2, fill=True, color='red', ax=axes[i, 0], label="Imputation Moyenne")
            sns.kdeplot(df_imputed_by_median[col], linewidth=1, fill=True, color='blue', ax=axes[i, 0], label="Imputation Médiane")
            axes[i, 0].set_title(f'Distribution de {col} - Moyenne et Médiane')
            axes[i, 0].legend()

            # Columna derecha: Imputaciones KNN
            sns.kdeplot(df_cleaned_data[col], linewidth=4, fill=True, color='black', ax=axes[i, 1], label="Avant")
            sns.kdeplot(df_imputed_inverse_2[col], linewidth=3, fill=True, color='green', ax=axes[i, 1], label="k = 2")
            sns.kdeplot(df_imputed_inverse_3[col], linewidth=2, fill=True, color='orange', ax=axes[i, 1], label="k = 3")
            sns.kdeplot(df_imputed_inverse_4[col], linewidth=1, fill=True, color='purple', ax=axes[i, 1], label="k = 4")
            axes[i, 1].set_title(f'Distribution de {col} - KNN')
            axes[i, 1].legend()

        plt.tight_layout(pad=3.0)
        return fig

 
    # histogramme des temperatures / precipitations pour une location
    def histogramme_temperatures_precipitations(self, location:str):
        # si l'attribut n'a pas encore été créé, alors on fait la reindexation temporelle
        #if not hasattr(self, "df_resample"):
        #    self.reindexation_temporelle()
        
        
        # retrait des données < 1/1/2009 et >31/12/2016 pour avoir des années complètes
#        df_resample = self.df_resample.loc['2009-01-01':'2016-12-31']

        df_filtre = self.df_resample[self.df_resample.Location == location]
        
        # retrait des données < 1/1/2009 et >31/12/2016 pour avoir des années complètes
        df_filtre = df_filtre.loc['2009-01-01':'2016-12-31']
        
        
        self.da = df_filtre.groupby('AAMM').agg({'MaxTemp': 'mean', 'Rainfall': 'sum', 'Annee':'max', 'Mois':'max'}).reset_index()
        self.da2 = self.da.groupby("Mois").agg({'MaxTemp': 'mean', 'Rainfall': 'mean'}).reset_index()
        
        fig, ax1 = plt.subplots(figsize=(20, 8))
        
        ax1.plot(np.array(self.da2['Mois']), np.array(self.da2['MaxTemp']), label='Température (°C)', color='r')
        ax1.set_yticks(np.arange(0,41,5))
        ax1.set_ylabel("Température (°C)")

        ax2 = ax1.twinx()
        ax2.bar(self.da2['Mois'], self.da2['Rainfall'], label='Précipitations (mm)', color='#06F', alpha=.5)
        ax2.set_yticks(np.arange(0,201,10))
        ax2.set_ylabel("Précipitations (mm)")

        
        nom_mois = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
        plt.xticks(ticks=np.arange(len(nom_mois))+1, labels=nom_mois)
        nom_titre = 'Températures moyennes mensuelles et cumul des précipations\nAnnées 2009 à 2016'
        nom_titre = nom_titre+" - "+location
        
        plt.title(nom_titre)
        plt.show()

    # anime une variable sur avril 2014
    def animation_variable(self, variable:str="RainToday", discrete:bool=False):
        
        data = self.data.loc[(self.data.index>='2014-04-01')&(self.data.index<='2014-09-30'),:].copy()
        data["Date"] = data.index
        
        if discrete:
            cible = data[variable].astype(str)
        else:
            cible = data[variable]
        
        fig = px.scatter_mapbox(data, 
                                lat='lat', 
                                lon='lng', 
                                hover_name='Location', 
                                color=cible, 
                                #text='Location', 
                                #labels=modele.labels_, 
                                animation_frame="Date",
#                                animation_group="Location",
                                size_max=30, 
                                opacity=.8,
                                #color_continuous_scale=px.colors.qualitative.Plotly
                                color_discrete_sequence=px.colors.qualitative.Set1,
                                #color_discrete_sequence=px.colors.qualitative.T10,
                                range_color=[data[variable].min(), data[variable].max()]
                                ).update_traces(marker=dict(size=30))
                
        #fig.update_layout(mapbox_style='open-street-map', width=self.disp_width, height=self.disp_height, mapbox_zoom=3.3, mapbox_center={"lat": (data.lat.min() + data.lat.max()) / 2, "lon": (data.lng.min() + data.lng.max()) / 2})
        fig = self.update_layout_australia(fig)

        #fig.show(renderer='browser')      
        return fig

    # affiche climats
    def affiche_climats(self):
        #df = self.data.drop("Date", axis=1).reset_index()[['Location', 'Climat', 'lng', 'lat']].drop_duplicates()
        
        fig = px.scatter_mapbox(self.df_moyenne.sort_values(by="Climat"), 
                            lat='lat', 
                            lon='lng', 
                            hover_name='Location', 
                            #text='Location', 
                            size_max=30, 
                            opacity=.8,
                            #color_continuous_scale=px.colors.qualitative.Plotly
                            color_discrete_sequence=px.colors.qualitative.Set1
                            #color_discrete_sequence=px.colors.qualitative.T10
                            ).update_traces(marker=dict(size=30))
    
        #fig.update_layout(mapbox_style='open-street-map', width=self.disp_width, height=self.disp_height, mapbox_zoom=3.3, mapbox_center={"lat": (self.data.lat.min() + self.data.lat.max()) / 2, "lon": (self.data.lng.min() + self.data.lng.max()) / 2})
        fig = self.update_layout_australia(fig)

        #fig.show(renderer='browser')      
        return fig


    # représentation geographique sur une carte    
    def synthetise_villes(self):
        self._charge_villes()
        
        #s_rt = self.df.groupby("Location")["RainTomorrow"].mean()
        df_rt = self.df.groupby("Location").agg({"RainTomorrow":'mean', "MaxTemp":'mean', "Pressure9am":'mean', 'Location':'count'})
        df_rt = df_rt.rename(columns={"Location":"Nb"})
        
        df_rt = df_rt.merge(self.df_cities, left_index=True, right_on='Location')
        
        fig = px.scatter_mapbox(df_rt, lat='lat', lon='lng', hover_name='Location', color='MaxTemp', size='RainTomorrow', color_continuous_scale='thermal')
#        fig = px.scatter_geo(df_rt, lat='lat', lon='lng', hover_name='Location', color='MaxTemp', size='RainTomorrow')
        
#fig.update_layout(mapbox_style='open-street-map', width=self.disp_width, height=self.disp_height, mapbox_zoom=3.3, mapbox_center={"lat": (self.data.lat.min() + self.data.lat.max()) / 2, "lon": (self.data.lng.min() + self.data.lng.max()) / 2})
        fig = self.update_layout_australia(fig)

        #fig.show(renderer='browser')             
        return fig

    # affichage layout Australie
    def update_layout_australia(self, fig):
        fig.update_layout(mapbox_style='open-street-map', width=self.disp_width, height=self.disp_height, mapbox_zoom=3.6, mapbox_center={"lat": (self.data.lat.min() + self.data.lat.max()) / 2, "lon": (self.data.lng.min() + self.data.lng.max()) / 2}, margin={"r":0,"t":0,"l":0,"b":0})
        return fig

    # charge infos sur villes
    def _charge_villes(self):
        # si deja chargé, on sort
        if hasattr(self, "df_cities"):
            return
        # Créer un DataFrame avec les coordonnées de la ville de Paris        
        self.df_cities = pd.read_csv(path_datasets+"villes_coordonnees.csv",sep=";")     

    
    # graphe des NA par couple Location/feature - si calcule=False, on charge un csv precedemment calculé, sinon, on le calcule
    def graphe_taux_na_location_feature(self, calcule=False):
        fig, axes = plt.subplots(1,1,figsize=(40,22))
        
        if calcule:
            # affiche le nb de NA pour chaque variable et pour chaque Location
            df_nb_na = self.df.groupby('Location').apply(lambda x: x.isna().sum()).drop(columns=["Location"])
            df_nb = self.df.groupby('Location').size().reset_index(name='NbTotEnregistrements\n(nuls ou non)')
            
            df_nb_na = df_nb_na.merge(df_nb, left_index=True, right_on='Location').set_index("Location")
            df_nb_na.iloc[:, :-1] = df_nb_na.iloc[:, :-1].div(df_nb_na.iloc[:, -1], axis=0) * 100
            # modifie pour avoir des %
            df_nb_na.iloc[:,:-1] = df_nb_na.iloc[:,:-1].div(df_nb_na.iloc[:,-1], axis=0)*100
            
            print("\n Taux de valeurs nulles par Location pour chaque variable\n")
            print (df_nb_na.iloc[:,1:])
        else:
            df_nb_na = pd.read_csv(path_datasets+"df_nb_na.csv", index_col=0)
        
        sns.heatmap(df_nb_na.iloc[:,1:-1], cmap='gnuplot2_r', annot=True, fmt=".0f", ax=axes, annot_kws={"size": 22})
        cbar = axes.collections[0].colorbar
        cbar.ax.tick_params(labelsize=22)
        axes.set_title("Taux de valeurs nulles pour chaque couple Location/variable", fontsize=28)
        axes.tick_params(axis='x', labelsize=22, rotation=45)
        axes.tick_params(axis='y', labelsize=22) 
        plt.show();

    # affiche matrice corr comme Quyen
    def matrice_corr_quyen(self, df, titre:str="Corrélations entre variables après ajout des nouvelles variables"):
        
        # si oui, c'est qu'on est sur le df initial
        if hasattr(df, "WindGustDir"):       
            df = df.drop(columns=['Location', 'Date', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
            
        # là, c'est qu'on est sur la dataset apres processing
        if hasattr(df, "Climat"):       
            df = df.drop(columns=['Location'])
            df = df.loc[:,~df.columns.str.startswith("Rain_J_")]
            df = df.loc[:,~df.columns.str.startswith("MaxTemp_J_")]
        
        cmap = sns.diverging_palette(260, 20, as_cmap=True)

        fig_corr, ax = plt.subplots(figsize=(20,13))
        corr_mat = df.corr()
        mask = np.triu(np.ones_like(corr_mat))

        sns.heatmap(corr_mat,
                    mask=mask,
                    annot=True,
                    fmt='.2f',
                    cmap=cmap,
                    vmin=-1, vmax=1, 
                    annot_kws={"size": 15})
        ax.set_title(titre, fontsize=15)

    # affiche repartition vent    
    def graphe_vent(self, location:str=""):
        if not hasattr(self.df, "WindGustDir_RAD"):       
            self.remplace_direction_vent()
        
        if (location==""):
            df_filtre = self.df
        else:
            df_filtre = self.df[self.df.Location == location]
       
        plt.figure(figsize=(30, 10))
#        fig, ax = plt.subplots(1,3)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, subplot_kw={'polar': True}, figsize=(8, 3.5))
        fig.subplots_adjust(wspace=0.6)
       
#        ax1 = fig.add_subplot(131, polar=True)
#        ax1.set_axes_off()
        self.graphe_vent_feature(df_filtre, location, "WindGustDir", ax1)
#        ax1 = fig.add_subplot(132, polar=True)
        self.graphe_vent_feature(df_filtre, location, "WindDir9am", ax2)
#        ax1 = fig.add_subplot(133, polar=True)
        self.graphe_vent_feature(df_filtre, location, "WindDir3pm", ax3)
        
        nom_title="Distribution des directions du vent - "
        if (location==""):
            nom_title+="Australie complète"
        else:
            nom_title+=location

        plt.suptitle(nom_title)
        plt.subplots_adjust(right=0.9)
        
        # on remet le df tel qu'il etait initialement
        self.df = self.df_orig.copy()     

    # affiche le vent pour une feature
    def graphe_vent_feature(self, df_filtre, location:str, variable:str, ax):
        vc = df_filtre[variable+"_RAD"].value_counts(normalize=True).sort_index()
        #vc.append(vc[0]) # pour fermer le tracé
        
        print (vc)
        self.vc = vc
        
        ax.fill(np.array(vc.index), vc.values, '#48A', alpha=.8)

        ax.set_xticks(np.arange(2*np.pi, 0, -np.pi/8))
        ax.set_xticklabels(["E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "N", "NNE", "NE", "ENE"])
        ax.set_yticklabels([])
        
        ax.set_title(variable)
        

    # remplace les variables categorielles de direction de vent par les composantes x et y
    def remplace_direction_vent(self):
        # si deja substitué: on sort
        if hasattr(self.df, "WindGustDir_X"):
            return
        
        print (self.df.columns)
        self._remplace_direction_vent("WindGustDir", "WindGustSpeed")
        print (self.df.columns)
        self._remplace_direction_vent("WindDir3pm", "WindSpeed3pm")
        print (self.df.columns)
        self._remplace_direction_vent("WindDir9am", "WindSpeed9am")       
        print (self.df.columns)
    
    # remplace une colonne categorielle de direction de vent par deux colonnes numeriques
    def _remplace_direction_vent(self, nom_colonne_dir: str, nom_colonne_speed: str):
        
        # on retire les NA de cette variable
        self.df = self.df.dropna(subset=[nom_colonne_dir])
        
        df_direction = pd.DataFrame()
        df_direction["dir"]=["E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "N", "NNE", "NE", "ENE"]
        
        # increment pour chacune des directions (sens horaire)
        increment=-np.pi/8
        
        df_direction["rad"]=increment*df_direction.index
        df_direction["sin"]=np.sin(df_direction.rad)
        df_direction["cos"]=np.cos(df_direction.rad)
        
        df_direction.loc[len(df_direction)]=[None, 0, 0, 0]
        
        self.df_dir_vent = df_direction
        
        # jointure pour deduire les cos et sin multipliés par la vitesse
        df_temp = self.df.merge(df_direction, left_on=nom_colonne_dir, right_on="dir")
        df_temp.index = df_temp.Date
        df_temp = df_temp.sort_index()
        
        df_temp[nom_colonne_dir+"_X"]=df_temp.cos*df_temp[nom_colonne_speed]
        df_temp[nom_colonne_dir+"_Y"]=df_temp.sin*df_temp[nom_colonne_speed]
        df_temp[nom_colonne_dir+"_RAD"]=df_temp["rad"]  # on garde rad pour les graphes
        
        self.df = df_temp.drop(columns=["cos", "sin", "rad", "dir", nom_colonne_dir])
     
    
