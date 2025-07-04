import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from PIL import Image
from datetime import datetime, timedelta

# --------------------------------------------------------------------
# FONCTIONS UTILITAIRES
# --------------------------------------------------------------------
@st.cache_data
def charger_donnees_base():
    """Charger et effectuer un prétraitement basique des données d'émissions CO2 pour la page d'accueil"""
    try:
        # Vérifier si le répertoire existe
        if not os.path.exists("raw_data"):
            return None, "Le dossier 'raw_data' n'a pas été trouvé."
        
        # Lister les fichiers et les charger
        fichiers_csv = [f for f in os.listdir("raw_data") if f.endswith(".csv")]
        if not fichiers_csv:
            return None, "Aucun fichier CSV trouvé dans le dossier 'raw_data'."
        
        dataframes = {}
        dataframes_individuels = {}  # Pour stocker les DataFrames individuels
        
        for nom_fichier in fichiers_csv:
            annee = nom_fichier.split(".")[0]
            chemin_fichier = os.path.join("raw_data", nom_fichier)
            try:
                df = pd.read_csv(chemin_fichier, sep=";", encoding="ISO-8859-1")
                
                # Gérer les colonnes dupliquées dès le chargement
                if df.columns.duplicated().any():
                    # Identifier les colonnes dupliquées
                    colonnes_dupliquees = df.columns[df.columns.duplicated()].tolist()
                    # Supprimer les colonnes dupliquées (garder la première occurrence)
                    df = df.loc[:, ~df.columns.duplicated()]
                
                dataframes[annee] = df
                dataframes_individuels[int(annee)] = df  # Stocker avec clé numérique
            except Exception as e:
                continue
        
        if not dataframes:
            return None, "Aucun fichier CSV valide n'a pu être chargé.", None
        
        # Prétraitement basique
        dataframes_traites = {}
        for annee, df_annee in dataframes.items():
            df_traite = df_annee.copy()
            
            # CORRECTION : Vérifier et supprimer les doublons de colonnes avant toute opération
            if df_traite.columns.duplicated().any():
                df_traite = df_traite.loc[:, ~df_traite.columns.duplicated()]
            
            df_traite.drop_duplicates(inplace=True)
            df_traite.reset_index(drop=True, inplace=True)
            df_traite["année"] = int(annee)
            dataframes_traites[annee] = df_traite
        
        # Concaténer tous les dataframes
        df_final = pd.concat(dataframes_traites.values(), ignore_index=True)
        
        # CORRECTION : Vérifier les colonnes dupliquées après concaténation
        if df_final.columns.duplicated().any():
            df_final = df_final.loc[:, ~df_final.columns.duplicated()]
        
        # Standardisation très basique des colonnes pour la page d'accueil
        mapping_colonnes = {}
        colonnes_existantes = list(df_final.columns)
        
        for col in colonnes_existantes:
            col_minuscule = str(col).lower()
            if 'co2' in col_minuscule and 'co2' not in mapping_colonnes.values():
                mapping_colonnes[col] = "co2"
            elif ('marque' in col_minuscule or 'brand' in col_minuscule) and 'lib_mrq_utac' not in mapping_colonnes.values():
                mapping_colonnes[col] = "lib_mrq_utac"
            elif ('carburant' in col_minuscule or 'fuel' in col_minuscule) and 'typ_crb' not in mapping_colonnes.values():
                mapping_colonnes[col] = "typ_crb"
            elif ('ptcl' in col_minuscule or 'particule' in col_minuscule) and 'ptcl' not in mapping_colonnes.values():
                mapping_colonnes[col] = "ptcl"
        
        # CORRECTION : Renommer les colonnes de manière sécurisée
        if mapping_colonnes:
            # Vérifier qu'aucun nouveau nom ne créera de doublons
            nouveaux_noms = list(mapping_colonnes.values())
            colonnes_finales = [mapping_colonnes.get(col, col) for col in df_final.columns]
            
            # Si des doublons sont détectés dans les nouveaux noms, les éviter
            if len(set(colonnes_finales)) != len(colonnes_finales):
                # Créer un mapping sécurisé sans doublons
                mapping_securise = {}
                noms_utilises = set()
                
                for ancien_nom, nouveau_nom in mapping_colonnes.items():
                    if nouveau_nom not in noms_utilises:
                        mapping_securise[ancien_nom] = nouveau_nom
                        noms_utilises.add(nouveau_nom)
                
                df_final.rename(columns=mapping_securise, inplace=True)
            else:
                df_final.rename(columns=mapping_colonnes, inplace=True)
        
        # CORRECTION FINALE : Vérifier une dernière fois les colonnes dupliquées
        if df_final.columns.duplicated().any():
            df_final = df_final.loc[:, ~df_final.columns.duplicated()]
        
        # Conversion numérique basique pour CO2
        if "co2" in df_final.columns:
            try:
                df_final["co2"] = pd.to_numeric(df_final["co2"], errors='coerce')
            except:
                try:
                    valeurs_co2 = []
                    for val in df_final["co2"]:
                        try:
                            if pd.isna(val):
                                valeurs_co2.append(np.nan)
                            else:
                                val_str = str(val).replace(',', '.')
                                val_num = float(val_str)
                                valeurs_co2.append(val_num)
                        except:
                            valeurs_co2.append(np.nan)
                    
                    df_final["co2"] = valeurs_co2
                except:
                    pass
        
        return df_final, None, dataframes_individuels
        
    except Exception as e:
        return None, f"Erreur lors du chargement des données : {str(e)}", None

def nettoyer_colonne_string(serie):
    """Fonction utilitaire pour nettoyer une colonne de type string de manière sécurisée"""
    try:
        # S'assurer que c'est une Series et la convertir en string
        if isinstance(serie, pd.Series):
            return serie.astype(str).str.upper().str.strip()
        else:
            # Si ce n'est pas une Series, la convertir d'abord
            return pd.Series(serie).astype(str).str.upper().str.strip()
    except:
        # En cas d'erreur, retourner la série originale
        return serie

def convertir_numerique_securise(serie, remplacer_virgule=True):
    """Fonction utilitaire pour convertir une colonne en numérique de manière sécurisée"""
    try:
        if remplacer_virgule:
            # Convertir en string et remplacer les virgules
            if isinstance(serie, pd.Series):
                serie_str = serie.astype(str).str.replace(',', '.')
            else:
                serie_str = pd.Series(serie).astype(str).str.replace(',', '.')
            return pd.to_numeric(serie_str, errors='coerce')
        else:
            return pd.to_numeric(serie, errors='coerce')
    except:
        return serie

def generer_dataframe_recapitulatif_nunique(dataframes_dict):
    """Générer un DataFrame récapitulatif du nombre de valeurs uniques par colonne et par année"""
    resultats = []
    
    for annee, df in dataframes_dict.items():
        for nom_colonne in df.columns:
            nb_valeurs_uniques = df[nom_colonne].nunique()
            resultats.append({
                'DataFrame': annee,
                'Colonne': nom_colonne,
                'Nombre de valeurs uniques': nb_valeurs_uniques
            })
    
    return pd.DataFrame(resultats)

def creer_graphique_cardinalite(dataframes_dict):
    """Créer le graphique de cardinalité par colonne et par année"""
    try:
        # Générer le DataFrame récapitulatif
        df_recap_nunique = generer_dataframe_recapitulatif_nunique(dataframes_dict)
        
        # Obtenir les années disponibles et les trier
        annees_disponibles = sorted(df_recap_nunique['DataFrame'].unique())
        
        # Calculer le nombre de lignes et colonnes pour les subplots
        n_annees = len(annees_disponibles)
        if n_annees <= 2:
            rows, cols = 1, n_annees
        elif n_annees <= 4:
            rows, cols = 2, 2
        else:
            rows = (n_annees + 2) // 3
            cols = 3
        
        # Créer les subplots
        fig = make_subplots(
            rows=rows, 
            cols=cols, 
            subplot_titles=[f'{annee}' for annee in annees_disponibles]
        )
        
        for i, annee in enumerate(annees_disponibles):
            data_annee = df_recap_nunique[df_recap_nunique['DataFrame'] == annee]
            
            # Calcul de l'index de ligne et de colonne pour le subplot
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            fig.add_trace(go.Bar(
                x=data_annee['Colonne'],
                y=data_annee['Nombre de valeurs uniques'],
                name=str(annee),
                showlegend=False
            ), row=row, col=col)
            
            fig.update_xaxes(
                tickangle=-45,
                tickmode="array",
                tickvals=data_annee['Colonne'],
                ticktext=data_annee['Colonne'],
                row=row, col=col
            )
            
            fig.update_yaxes(
                nticks=5,
                row=row, col=col
            )
        
        fig.update_layout(
            height=600 * rows,
            title_text="📊 Cardinalité des colonnes par année (nb valeurs uniques)",
            title_x=0.5,
        )
        
        return fig, df_recap_nunique
        
    except Exception as e:
        st.error(f"Erreur lors de la création du graphique de cardinalité : {e}")
        return None, None

def creer_graphiques_apercu_simple(df):
    """Créer des graphiques d'aperçu simple pour la page d'accueil"""
    graphiques = {}
    
    # Vérifier l'intégrité du DataFrame
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    
    # 1. Taille du jeu de données par année
    if "année" in df.columns:
        try:
            comptes_annees = df["année"].value_counts().sort_index()
            fig_annees = px.bar(
                x=comptes_annees.index,
                y=comptes_annees.values,
                title="📊 Nombre d'enregistrements par année",
                labels={"x": "Année", "y": "Nombre d'enregistrements"},
                color=comptes_annees.values,
                color_continuous_scale="blues"
            )
            fig_annees.update_layout(height=400, showlegend=False)
            graphiques["annees"] = fig_annees
        except Exception as e:
            pass
    
    # 2. Top 5 des marques
    if "lib_mrq_utac" in df.columns:
        try:
            colonne_marque = df["lib_mrq_utac"].dropna()
            
            if not colonne_marque.empty:
                valeurs_marque = []
                for val in colonne_marque:
                    try:
                        if pd.notna(val):
                            val_nettoyee = str(val).strip()
                            if val_nettoyee and val_nettoyee.lower() != 'nan':
                                valeurs_marque.append(val_nettoyee)
                    except:
                        continue
                
                if valeurs_marque:
                    serie_marques = pd.Series(valeurs_marque)
                    top_marques = serie_marques.value_counts().head(5)
                    
                    if len(top_marques) > 0:
                        fig_marques = px.pie(
                            values=top_marques.values,
                            names=top_marques.index,
                            title="🚗 Top 5 des marques automobiles"
                        )
                        fig_marques.update_layout(height=400)
                        graphiques["marques"] = fig_marques
                        
        except Exception as e:
            pass
    
    # 3. Distribution CO2
    if "co2" in df.columns:
        try:
            colonne_co2 = df["co2"]
            co2_numerique = pd.to_numeric(colonne_co2, errors='coerce')
            co2_nettoye = co2_numerique.dropna()
            co2_fini = co2_nettoye[np.isfinite(co2_nettoye)]
            co2_filtre = co2_fini[(co2_fini >= 0) & (co2_fini <= 1000)]
            
            if len(co2_filtre) > 10:
                fig_co2 = px.histogram(
                    x=co2_filtre,
                    title="🌍 Distribution des émissions CO2",
                    labels={"x": "CO2 (g/km)", "y": "Fréquence"},
                    nbins=50
                )
                fig_co2.update_layout(height=400)
                graphiques["co2"] = fig_co2
                
        except Exception as e:
            pass
    
    # 4. NOUVEAU : Top 5 des types de carburant
    if "typ_crb" in df.columns:
        try:
            colonne_carburant = df["typ_crb"].dropna()
            
            if not colonne_carburant.empty:
                valeurs_carburant = []
                for val in colonne_carburant:
                    try:
                        if pd.notna(val):
                            val_nettoyee = str(val).strip().upper()
                            if val_nettoyee and val_nettoyee.lower() != 'nan':
                                valeurs_carburant.append(val_nettoyee)
                    except:
                        continue
                
                if valeurs_carburant:
                    serie_carburants = pd.Series(valeurs_carburant)
                    top_carburants = serie_carburants.value_counts().head(5)
                    
                    if len(top_carburants) > 0:
                        fig_carburants = px.pie(
                            values=top_carburants.values,
                            names=top_carburants.index,
                            title="⛽ Top 5 des types de carburant",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        fig_carburants.update_layout(height=400)
                        graphiques["carburants"] = fig_carburants
                        
        except Exception as e:
            pass
    
    return graphiques

def creer_gantt_projet():
    """Version ultra-simplifiée du Gantt"""
    try:
        # Créer un graphique en barres horizontales simple
        phases = ['Exploration & Préparation', 'Modélisation', 'Finalisation']
        durees = [60, 60, 52]  # Durées en jours
        couleurs = ['#3498db', '#e74c3c', '#2ecc71']
        
        fig = go.Figure()
        
        for i, (phase, duree, couleur) in enumerate(zip(phases, durees, couleurs)):
            fig.add_trace(go.Bar(
                y=[phase],
                x=[duree],
                orientation='h',
                marker_color=couleur,
                name=phase,
                showlegend=False
            ))
        
        fig.update_layout(
            title='📅 Durée des Phases du Projet (en jours)',
            xaxis_title='Durée (jours)',
            height=300,
            plot_bgcolor='white'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Erreur graphique simple : {e}")
        return None

# --------------------------------------------------------------------
# PAGE D'ACCUEIL PRINCIPALE
# --------------------------------------------------------------------
def afficher_accueil():
    # Ajouter le logo dans la sidebar
    try:
        if os.path.exists("datascientest.png"):
            logo = Image.open("datascientest.png")
            st.sidebar.image(logo, width=500)
        else:
            st.sidebar.warning("Logo datascientest.png non trouvé")
    except Exception as e:
        st.sidebar.error(f"Erreur lors du chargement du logo : {e}")
    
    # Titre principal avec image CO2
    col_titre, col_image = st.columns([3, 1])
    
    with col_titre:
        st.title("🚗💨 Projet d'Analyse des Émissions CO2")
     
    st.markdown("---")
    st.markdown("**Soutenu par : Jonathan LANGNER & Jonathan CHICHEPORTICHE**")
    st.markdown("**Promotion : Octobre 2024 - Parcours Data Scientist**")
    st.markdown("---")
    
    # Créer les onglets
    tabs = st.tabs([" Contexte", "Méthodologie du Projet", " Aperçu des Données"])

    # Onglet 1 : Contexte
    with tabs[0]:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("📝 Description du projet")
            st.markdown("""
            <div style="text-align: justify; padding: 10px; background-color: #f8f9fa; border-radius: 10px; margin: 10px 0;">
            Identifier les véhicules qui émettent le plus de CO2 est important pour identifier les caractéristiques techniques qui jouent un rôle dans la pollution. Prédire à l'avance cette pollution permet de prévenir dans le cas de l'apparition de nouveaux types de véhicules
            (nouvelles séries de voitures par exemple).
            </div>
            """, unsafe_allow_html=True)
        
        # NOUVEAU : Info-bulle ajoutée ici
        st.success("**A noter** : ce projet s'intéresse au parc automobile français entre les années 2012 à 2015")
        
        with col2:
            st.subheader("📊 Sources de données")
            st.markdown("""
            <div style="text-align: justify; padding: 10px; background-color: #f8f9fa; border-radius: 10px; margin: 10px 0;">
            <strong>• Bases de données gouvernementales </strong> : ADEME, Data.gouv<br>
            <strong>• Classification ACRISS</strong><br>
            <strong>• Résultats des tests d'émissions</strong> : Carlabelling
            </div>
            """, unsafe_allow_html=True)
        
        # Objectifs du projet sur toute la largeur
        st.markdown("---")
        st.info("""
        **🎯 Objectifs du projet :**
        - Analyser les données pour les émissions CO2
        - Construire deux modèles prédictifs (un linéaire & un logistique)
        - Identifier les facteurs clés influençant les émissions de CO2
        - Classer les véhicules par classe d'efficacité CO2 (référence ACRISS)
        - Émettre des conclusions et recommandations
        """)

    # Onglet 2 : Méthodologie du Projet
    # Onglet 2 : Méthodologie du Projet
    with tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Principales étapes")
            st.write()
            st.write("""
            **1.  Chargement et intégration des données**
            
            **2.  Preprocessing**
            
            **3.  Feature Engineering**
            """)
        
        with col2:
            st.subheader("")
            st.write("""
            **4.  Analyse exploratoire des données**
            
            **5.  Développement de modèles**
            
            **6.  Évaluation et interprétation**
            """)
        
        st.markdown("---")
        
        # NOUVEAU : Section Gestion de Projet
        st.subheader("Gestion de Projet")
        
        # Afficher le diagramme de Gantt
        fig_gantt = creer_gantt_projet()
        if fig_gantt is not None:
            st.plotly_chart(fig_gantt, use_container_width=True)
            
            # Ajouter une analyse du planning
            st.info("""
            **📊 Analyse du Planning :**
            - **Durée totale** : 6 mois (Janvier à Juin 2025)
            - **3 Phases principales** : Exploration, Modélisation, Finalisation
            - **2 Livrables majeurs** : Rendu 1 (Mars) et Rendu Final (Juin)
            - **Jalons réguliers** : Checkpoints hebdomadaires pour suivi d'avancement
            """)
        else:
            # Fallback si le Gantt ne fonctionne pas
            st.subheader("📅 Planning du Projet")
            st.write("""
            **Janvier-Février :** Exploration et préparation des données
            **Mars-Avril :** Modélisation et tests des algorithmes  
            **Mai-Juin :** Finalisation et livrable final
            """)
        
        st.markdown("---")
        
        # NOUVEAU : Section Communication/Collaboration
        st.subheader("Communication/Collaboration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            - **Slack** : Communication hebdomadaire et partage de fichiers
            - **Zoom** : Réunions d'équipe et sessions de travail collaboratif
            """)
        
        with col2:
            st.markdown("""
            - **GitHub** : Versioning du code et collaboration
            - **VS Code** : Développement et tests des notebooks
            """)
        
        st.markdown("---")
        
        # NOUVEAU : Section Technologies utilisées
        st.subheader("Technologies utilisées")
        
        # Créer le tableau des technologies
        technologies_data = {
            "Module/Librairie": [
                "Streamlit",
                "Scikit-learn", 
                "Matplotlib",
                "Seaborn",
                "Plotly",
                "Pandas",
                "NumPy",
                "Pickle/Joblib"
            ],
            "Finalité": [
                "Interface web interactive et déploiement de l'application",
                "Modélisation machine learning (régression, classification, preprocessing)",
                "Visualisations graphiques statiques et personnalisées",
                "Visualisations statistiques avancées et matrices de corrélation",
                "Graphiques interactifs et tableaux de bord dynamiques",
                "Manipulation et analyse des données structurées",
                "Calculs numériques et opérations sur les arrays",
                "Sauvegarde et chargement des modèles entraînés"
            ],
            "Usage Principal": [
                "Interface utilisateur",
                "Machine Learning",
                "Visualisation",
                "Analyse statistique", 
                "Interactivité",
                "Data Processing",
                "Calcul numérique",
                "Persistance des modèles"
            ]
        }
        
        df_technologies = pd.DataFrame(technologies_data)
        
        # Afficher le tableau avec style
        st.dataframe(
            df_technologies, 
            use_container_width=True,
            hide_index=True
        )
        

    # Onglet 3 : Aperçu des données
    with tabs[2]:
        st.markdown("---")
        
        # Charger les données de base
        df_base, msg_erreur, dataframes_individuels = charger_donnees_base()
        
        if df_base is None:
            st.error(msg_erreur)
            st.info("💡 Veuillez vous assurer d'avoir un dossier 'raw_data' avec des fichiers CSV dans le même répertoire que ce script.")
            return
        
        if df_base.empty:
            st.warning("⚠️ Aucune donnée disponible pour la visualisation.")
            return
        
        # CORRECTION : Vérifier les colonnes dupliquées avant utilisation
        if df_base.columns.duplicated().any():
            st.warning("⚠️ Colonnes dupliquées détectées et supprimées.")
            df_base = df_base.loc[:, ~df_base.columns.duplicated()]
        
        # Afficher les informations de base du jeu de données
        st.success(f"✅ Données chargées avec succès ! **{len(df_base):,}** enregistrements trouvés.")
        
        # Métriques de résumé du jeu de données
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total des enregistrements", f"{len(df_base):,}")
        with col2:
            st.metric("📋 Total des colonnes", len(df_base.columns))
        with col3:
            if 'année' in df_base.columns:
                annees = df_base['année'].nunique()
                plage_annees = f"{df_base['année'].min()}-{df_base['année'].max()}"
                st.metric("📅 Années couvertes", f"{annees} ({plage_annees})")
            else:
                st.metric("📅 Années couvertes", "N/A")
        with col4:
            pourcentage_manquant = (df_base.isnull().sum().sum() / (len(df_base) * len(df_base.columns)) * 100)
            st.metric("❓ Données manquantes", f"{pourcentage_manquant:.1f}%")
        
        st.markdown("---")
        
        # Créer et afficher des graphiques simples
        graphiques = creer_graphiques_apercu_simple(df_base)
        
        if graphiques:
            # Organiser l'affichage des graphiques
            graphiques_items = list(graphiques.items())
            
            # Première ligne : années et marques
            if "annees" in graphiques or "marques" in graphiques:
                col1, col2 = st.columns(2)
                if "annees" in graphiques:
                    with col1:
                        st.plotly_chart(graphiques["annees"], use_container_width=True)
                if "marques" in graphiques:
                    with col2:
                        st.plotly_chart(graphiques["marques"], use_container_width=True)
            
            # Deuxième ligne : CO2 et carburants
            if "co2" in graphiques or "carburants" in graphiques:
                col1, col2 = st.columns(2)
                if "co2" in graphiques:
                    with col1:
                        st.plotly_chart(graphiques["co2"], use_container_width=True)
                if "carburants" in graphiques:
                    with col2:
                        st.plotly_chart(graphiques["carburants"], use_container_width=True)
                        
                        # Ajouter une analyse textuelle pour les carburants
                        if "typ_crb" in df_base.columns:
                            try:
                                carburant_counts = df_base["typ_crb"].value_counts()
                                total_vehicles = len(df_base)
                                
                                st.markdown("**📊 Analyse des carburants :**")
                                
                                # Calculer les pourcentages pour les principaux carburants
                                if "GO" in carburant_counts.index:
                                    go_percentage = (carburant_counts["GO"] / total_vehicles) * 100
                                    st.write(f"- **Gazole (GO)** : {go_percentage:.1f}% du parc automobile")
                                
                                if "ES" in carburant_counts.index:
                                    es_percentage = (carburant_counts["ES"] / total_vehicles) * 100
                                    st.write(f"- **Essence (ES)** : {es_percentage:.1f}% du parc automobile")
                                                              
                            except Exception as e:
                                pass
            
            # Afficher les autres graphiques s'il y en a
            autres_graphiques = [item for item in graphiques_items if item[0] not in ["annees", "marques", "co2", "carburants"]]
            for nom, fig in autres_graphiques:
                st.plotly_chart(fig, use_container_width=True)
                
        else:
            st.info("📊 Aucun graphique n'a pu être généré avec les données disponibles.")
        
        st.markdown("---")
        
        # ================================================================
        # Graphique de cardinalité par colonne et par année
        # ================================================================
        st.subheader("📊 Analyse de la Cardinalité des Variables par Année")
        
        if dataframes_individuels:
            fig_cardinalite, df_recap_nunique = creer_graphique_cardinalite(dataframes_individuels)
            
            if fig_cardinalite is not None:
                st.plotly_chart(fig_cardinalite, use_container_width=True)
                
                st.info("""
                💡 **Analyse :** La variable `cnit` ressort dans chaque DataFrame. 
                Chaque numéro CNIT est différent en fonction du modèle de véhicule.
                """)
            else:
                st.warning("Impossible de créer le graphique de cardinalité.")
        else:
            st.warning("Données individuelles par année non disponibles pour l'analyse de cardinalité.")
        
        st.markdown("---")
        
        # Informations sur les prochaines étapes
        st.info("""
        **🚀 Prochaines étapes :**
        
        Naviguez vers la page **Problème Régression** pour :
        - Effectuer un preprocessing détaillé des données
        - Explorer des visualisations de données complètes
        - Construire et évaluer des modèles de prédiction
        - Analyser les feature importance et les performances du modèle
        """)

# --------------------------------------------------------------------
# EXÉCUTION PRINCIPALE
# --------------------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(
        page_title="Analyse des Émissions CO2", 
        page_icon="🚀",
        layout="wide"
    )
    afficher_accueil()
