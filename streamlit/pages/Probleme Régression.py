import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import nbformat
from scipy.stats import pearsonr, f_oneway, kruskal
from sklearn.model_selection import train_test_split, GridSearchCV
import collections
import tabulate
import os
from scipy.stats import chi2_contingency
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sklearn.datasets
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import plotly.figure_factory as ff
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import shap
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import io
import pickle
import joblib
import time
from concurrent.futures import ThreadPoolExecutor
import threading

# Supprimer les avertissements
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# Fonction pour afficher les DataFrames de manière sûre
def safe_dataframe_display(df, title="DataFrame", max_rows=1000):
    """Affiche un DataFrame de manière sûre en gérant les problèmes de sérialisation Arrow"""
    try:
        if len(df) > max_rows:
            st.warning(f"DataFrame trop volumineux ({len(df)} lignes). Affichage des {max_rows} premières lignes.")
            df_display = df.head(max_rows).copy()
        else:
            df_display = df.copy()
        
        # Convertir les colonnes problématiques
        for col in df_display.columns:
            if df_display[col].dtype == 'object':
                try:
                    df_display[col] = df_display[col].astype(str)
                except:
                    pass
        
        st.dataframe(df_display)
    except Exception as e:
        st.error(f"Erreur lors de l'affichage du {title}: {e}")
        st.write(f"**{title} - Informations de base :**")
        st.write(f"Forme : {df.shape}")
        st.write(f"Colonnes : {list(df.columns)}")
        st.write("Échantillon des données :")
        st.text(str(df.head()))

# Fonction pour calculer le Cramer's V
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1)) / (n-1))
    rcorr = r - ((r-1)**2) / (n-1)
    kcorr = k - ((k-1)**2) / (n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# Fonction optimisée pour charger les modèles avec cache
@st.cache_data(ttl=3600)  # Cache pendant 1 heure
def load_models_and_scalers_cached(models_dir="saved_models_regression"):
    """Version mise en cache de la fonction de chargement des modèles"""
    return load_models_and_scalers(models_dir)

# Fonction pour charger les modèles et scalers
def load_models_and_scalers(models_dir="saved_models_regression"):
    """Charge uniquement les modèles de ML depuis le dossier spécifié"""
    loaded_models = {}
    loaded_scalers = {}
    
    if not os.path.exists(models_dir):
        st.error(f"Le dossier '{models_dir}' n'existe pas.")
        return {}, {}
    
    try:
        files = os.listdir(models_dir)
        pkl_files = [f for f in files if f.endswith('.pkl')]
        
        # Types d'objets qui sont des modèles de ML
        ml_model_types = [
            'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet',
            'RandomForestRegressor', 'GradientBoostingRegressor', 
            'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor',
            'SVR', 'DecisionTreeRegressor', 'ExtraTreesRegressor',
            'AdaBoostRegressor', 'BaggingRegressor', 'VotingRegressor',
            'StackingRegressor', 'MLPRegressor', 'KNeighborsRegressor'
        ]
        
        def load_single_model(pkl_file):
            file_path = os.path.join(models_dir, pkl_file)
            try:
                loaded_object = joblib.load(file_path)
                object_type = type(loaded_object).__name__
                
                if object_type in ml_model_types:
                    model_name = pkl_file.replace('.pkl', '')
                    if 'conventionnel' in model_name:
                        display_name = model_name.replace('reg_conventionnel_', '').replace('_', ' ').title()
                    elif 'gridsearch' in model_name:
                        display_name = model_name.replace('reg_gridsearch_', '').replace('_', ' ').title() + ' (GridSearch)'
                    else:
                        display_name = model_name.replace('reg_', '').replace('_', ' ').title()
                    
                    return ('model', display_name, loaded_object)
                
                elif 'scaler' in pkl_file.lower():
                    scaler_name = pkl_file.replace('.pkl', '').replace('_scaler', '').replace('scaler_', '')
                    return ('scaler', scaler_name, loaded_object)
                
            except Exception as e:
                return ('error', pkl_file, str(e))
            
            return None
        
        # Chargement parallèle des modèles
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(load_single_model, pkl_files))
        
        for result in results:
            if result is None:
                continue
            
            result_type, name, obj = result
            
            if result_type == 'model':
                loaded_models[name] = obj
            elif result_type == 'scaler':
                loaded_scalers[name] = obj
            elif result_type == 'error':
                st.warning(f"⚠️ Impossible de charger {name}: {obj}")
        
        return loaded_models, loaded_scalers
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement : {str(e)}")
        return {}, {}

# Fonction optimisée pour évaluer les modèles
def evaluate_models_parallel(models_dict, X_test, y_test):
    """Évalue les modèles en parallèle pour améliorer les performances"""
    
    def evaluate_single_model(model_item):
        name, model = model_item
        try:
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            return name, {'MAE': mae, 'R2': r2}
        except Exception as e:
            return name, {'MAE': np.nan, 'R2': np.nan, 'error': str(e)}
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(evaluate_single_model, models_dict.items()))
    
    return dict(results)

# Fonction pour créer le tableau Plotly amélioré
def create_enhanced_plotly_table(df_results, title, best_model_name=None):
    """Crée un tableau Plotly avec mise en surbrillance et code couleur"""
    
    # Préparer les données
    df_display = df_results.copy()
    
    # Créer le tableau de base
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Modèle</b>', '<b>MAE</b>', '<b>R²</b>'],
            fill_color='darkslategray',
            font=dict(color='white', size=14),
            align='left'
        ),
        cells=dict(
            values=[
                df_display['Modèle'],
                df_display['MAE'],
                df_display['R²']
            ],
            fill_color=[
                ['white'] * len(df_display),  # Colonne Modèle
                ['white'] * len(df_display),  # Colonne MAE
                ['white'] * len(df_display)   # Colonne R²
            ],
            align='left',
            font=dict(size=12)
        )
    )])
    
    # Appliquer les couleurs
    for i, row in df_display.iterrows():
        # Mise en surbrillance du meilleur modèle
        if best_model_name and row['Modèle'] == best_model_name:
            for col in range(3):
                fig.data[0].cells.fill.color[col][i] = 'gold'
        
        # Code couleur pour R²
        r2_val = row['R²']
        if r2_val != 'Erreur':
            try:
                r2_float = float(r2_val)
                if r2_float > 0.95:
                    color = 'lightgreen'
                elif r2_float < 0.7:
                    color = 'lightcoral'
                elif r2_float < 0.9:
                    color = 'lightyellow'
                else:
                    color = 'lightblue'
                
                # Appliquer la couleur à la colonne R² (index 2)
                if best_model_name and row['Modèle'] != best_model_name:
                    fig.data[0].cells.fill.color[2][i] = color
            except:
                pass
    
    # Mise en forme du tableau
    fig.update_layout(
        title=f"<b>{title}</b>",
        title_x=0.5,
        margin=dict(l=20, r=20, t=60, b=20),
        height=300
    )
    
    return fig

# Page de régression linéaire avec onglets
def show_linear_regression():
    st.markdown('<div id="top"></div>', unsafe_allow_html=True)
    
    st.title("Identification des facteurs favorisant les émissions de CO2")
    st.markdown("Exploration et modélisation de la relation entre les caractéristiques et les émissions de CO2.")
    st.markdown("---")

    # Initialiser les variables
    dataframes = None
    df = None
    df_final = None
    df_final_ml = None
    numerical_cols = None
    X_train, X_test, y_train, y_test = None, None, None, None
    feature_columns = None
    lr_model = None

    tabs = st.tabs([
        "Chargement des Données",
        "Preprocessing", 
        "Visualisation",
        "Feature Engineering",
        "Modélisation ML",
        "Feature Importance",
        "Conclusions / Recommandations"
    ])

    # ----------------------------------------------------------------
    # Onglet Chargement des Données (index 0)
    # ----------------------------------------------------------------
    with tabs[0]:
        st.markdown("Chargement des fichiers sources et analyse exploratoire")
        
        dataframes = {}
        try:
            if not os.path.exists("raw_data"):
                st.error("Le dossier 'raw_data' n'a pas été trouvé. Veuillez créer un dossier nommé 'raw_data' et y placer vos fichiers CSV.")
                st.stop()

            csv_files = [f for f in os.listdir("raw_data") if f.endswith(".csv")]
            if not csv_files:
                st.error("Aucun fichier CSV trouvé dans le dossier 'raw_data'.")
                st.stop()

            # ÉTAPE 1: Chargement des fichiers
            st.subheader(" Étape 1 : Chargement des Fichiers CSV")
            
            loading_results = []
            
            for filename in csv_files:
                year = filename.split(".")[0]
                filepath = os.path.join("raw_data", filename)
                
                try:
                    df_temp = pd.read_csv(filepath, sep=";", encoding="ISO-8859-1")
                    dataframes[year] = df_temp
                    
                    memory_usage = df_temp.memory_usage(deep=True).sum() / 1024**2
                    
                    loading_results.append({
                        'Fichier': filename,
                        'Année': year,
                        'Lignes': f"{len(df_temp):,}",
                        'Colonnes': len(df_temp.columns),
                        'Taille (MB)': f"{memory_usage:.2f}",
                        'Statut': '✅ Succès'
                    })
                    
                except Exception as e:
                    loading_results.append({
                        'Fichier': filename,
                        'Année': year,
                        'Lignes': 'N/A',
                        'Colonnes': 'N/A',
                        'Taille (MB)': 'N/A',
                        'Statut': f'❌ Erreur: {str(e)[:50]}...'
                    })
            
            # Afficher le tableau récapitulatif
            df_loading_summary = pd.DataFrame(loading_results)
            safe_dataframe_display(df_loading_summary, "Résumé du chargement")
            
            # Métriques globales
            successful_loads = len([r for r in loading_results if '✅' in r['Statut']])
            total_rows = sum(len(df) for df in dataframes.values())
            total_memory = sum(df.memory_usage(deep=True).sum() for df in dataframes.values()) / 1024**2
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📁 Fichiers chargés", f"{successful_loads}/{len(csv_files)}")
            with col2:
                st.metric("📊 Total lignes", f"{total_rows:,}")
            with col3:
                st.metric("📅 Années couvertes", len(dataframes))
            with col4:
                st.metric("💾 Mémoire totale", f"{total_memory:.1f} MB")

            st.success(f"✅ Chargement réussi de {len(dataframes)} DataFrame(s) depuis le dossier 'raw_data' !")

            # ÉTAPE 2: Exploration par DataFrame
            st.markdown("---")
            st.subheader(" Étape 2 : Exploration par Année")
            
            if dataframes:
                selected_year = st.selectbox(
                    "🎯 Sélectionnez une année pour l'exploration détaillée :",
                    options=sorted(dataframes.keys()),
                    key="year_selector"
                )
                
                if selected_year:
                    df_selected = dataframes[selected_year]
                    
                    # Onglets pour différents types d'exploration
                    explore_tabs = st.tabs([
                        "📊 Vue d'ensemble", 
                        "📈 Statistiques", 
                        "🔍 Colonnes", 
                        "📋 Échantillon"
                    ])
                    
                    # Sous-onglet: Vue d'ensemble
                    with explore_tabs[0]:
                        st.markdown(f"### 📊 Vue d'ensemble - Année {selected_year}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("📏 Dimensions", f"{df_selected.shape[0]} × {df_selected.shape[1]}")
                        
                        with col2:
                            memory_mb = df_selected.memory_usage(deep=True).sum() / 1024**2
                            st.metric("💾 Mémoire", f"{memory_mb:.2f} MB")
                        
                        with col3:
                            null_count = df_selected.isnull().sum().sum()
                            st.metric("❓ Valeurs manquantes", f"{null_count:,}")
                        
                        with col4:
                            duplicates = df_selected.duplicated().sum()
                            st.metric("🔄 Doublons", f"{duplicates:,}")
                        
                        # Informations sur les types de données - Tableau seulement
                        st.markdown("#### 📋 Répartition des Types de Données")
                        
                        type_counts = df_selected.dtypes.value_counts()
                        
                        # Tableau étiré sur toute la largeur
                        type_details = pd.DataFrame({
                            'Type': type_counts.index.astype(str),
                            'Nombre de colonnes': type_counts.values,
                            'Pourcentage': (type_counts.values / len(df_selected.columns) * 100).round(1)
                        })
                        safe_dataframe_display(type_details, "Détail des types")

                    # Sous-onglet: Statistiques
                    with explore_tabs[1]:
                        st.markdown(f"### 📈 Statistiques Descriptives - Année {selected_year}")
                        
                        numeric_cols = df_selected.select_dtypes(include=[np.number]).columns
                        categorical_cols = df_selected.select_dtypes(include=['object']).columns
                        
                        if len(numeric_cols) > 0:
                            st.markdown("#### 🔢 Variables Numériques")
                            
                            numeric_stats = df_selected[numeric_cols].describe()
                            safe_dataframe_display(numeric_stats, "Statistiques numériques")
                            
                            if len(numeric_cols) > 1:
                                missing_numeric = df_selected[numeric_cols].isnull().sum()
                                missing_numeric = missing_numeric[missing_numeric > 0]
                                
                                if len(missing_numeric) > 0:
                                    fig_missing_num = px.bar(
                                        x=missing_numeric.index,
                                        y=missing_numeric.values,
                                        title="Valeurs manquantes - Variables numériques",
                                        labels={'x': 'Variables', 'y': 'Nombre de valeurs manquantes'},
                                        color=missing_numeric.values,
                                        color_continuous_scale='Reds'
                                    )
                                    fig_missing_num.update_layout(height=400)
                                    st.plotly_chart(fig_missing_num, use_container_width=True)
                        
                        if len(categorical_cols) > 0:
                            st.markdown("#### 🏷️ Variables Catégorielles")
                            
                            cat_summary = []
                            for col in categorical_cols:
                                cat_summary.append({
                                    'Variable': col,
                                    'Valeurs uniques': df_selected[col].nunique(),
                                    'Valeurs manquantes': df_selected[col].isnull().sum(),
                                    'Mode': df_selected[col].mode().iloc[0] if not df_selected[col].mode().empty else 'N/A',
                                    'Fréquence du mode': df_selected[col].value_counts().iloc[0] if not df_selected[col].empty else 0
                                })
                            
                            df_cat_summary = pd.DataFrame(cat_summary)
                            safe_dataframe_display(df_cat_summary, "Résumé variables catégorielles")

                    # Sous-onglet: Colonnes
                    with explore_tabs[2]:
                        st.markdown(f"### 🔍 Analyse des Colonnes - Année {selected_year}")
                        
                        col_analysis = []
                        for col in df_selected.columns:
                            unique_count = df_selected[col].nunique()
                            null_count = df_selected[col].isnull().sum()
                            null_pct = (null_count / len(df_selected)) * 100
                            
                            if df_selected[col].dtype in ['int64', 'float64']:
                                var_type = "🔢 Numérique"
                            else:
                                var_type = "🏷️ Catégorielle"
                            
                            if unique_count == 1:
                                cardinality = "🔒 Constante"
                            elif unique_count == len(df_selected):
                                cardinality = "🔑 Identifiant"
                            elif unique_count / len(df_selected) > 0.95:
                                cardinality = "📈 Très haute"
                            elif unique_count > 50:
                                cardinality = "📊 Haute"
                            elif unique_count > 10:
                                cardinality = "📋 Moyenne"
                            else:
                                cardinality = "📝 Faible"
                            
                            col_analysis.append({
                                'Colonne': col,
                                'Type': var_type,
                                'Cardinalité': cardinality,
                                'Valeurs uniques': unique_count,
                                'Valeurs manquantes': null_count,
                                '% manquantes': round(null_pct, 2),
                                'Ratio unicité': round(unique_count / len(df_selected), 4)
                            })
                        
                        df_col_analysis = pd.DataFrame(col_analysis)
                        safe_dataframe_display(df_col_analysis, "Analyse détaillée des colonnes")
                        
                        st.markdown("#### 🎯 Analyse Ciblée")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            high_missing = df_col_analysis[df_col_analysis['% manquantes'] > 10]
                            if not high_missing.empty:
                                st.warning("⚠️ **Colonnes avec > 10% de valeurs manquantes:**")
                                safe_dataframe_display(
                                    high_missing[['Colonne', '% manquantes']].sort_values('% manquantes', ascending=False),
                                    "Colonnes problématiques"
                                )
                            else:
                                st.success("✅ Aucune colonne avec > 10% de valeurs manquantes")
                        
                        with col2:
                            low_variance = df_col_analysis[df_col_analysis['Valeurs uniques'] <= 2]
                            if not low_variance.empty:
                                st.info("ℹ️ **Colonnes avec peu de variance:**")
                                safe_dataframe_display(
                                    low_variance[['Colonne', 'Valeurs uniques', 'Cardinalité']],
                                    "Colonnes faible variance"
                                )
                            else:
                                st.success("✅ Toutes les colonnes ont une variance suffisante")

                    # Sous-onglet: Échantillon
                    with explore_tabs[3]:
                        st.markdown(f"### 📋 Échantillon des Données - Année {selected_year}")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            sample_size = st.selectbox(
                                "📏 Nombre de lignes à afficher:",
                                [5, 10, 20, 50, 100],
                                index=1,
                                key=f"sample_size_{selected_year}"
                            )
                        
                        with col2:
                            sample_type = st.selectbox(
                                "🎯 Type d'échantillon:",
                                ["Premières lignes", "Dernières lignes", "Échantillon aléatoire"],
                                key=f"sample_type_{selected_year}"
                            )
                        
                        with col3:
                            show_info = st.checkbox(
                                "📊 Afficher info() du DataFrame",
                                value=False,
                                key=f"show_info_{selected_year}"
                            )
                        
                        if sample_type == "Premières lignes":
                            sample_df = df_selected.head(sample_size)
                            st.markdown(f"**🔝 {sample_size} premières lignes:**")
                        elif sample_type == "Dernières lignes":
                            sample_df = df_selected.tail(sample_size)
                            st.markdown(f"**🔚 {sample_size} dernières lignes:**")
                        else:
                            sample_df = df_selected.sample(n=min(sample_size, len(df_selected)), random_state=42)
                            st.markdown(f"**🎲 {sample_size} lignes aléatoires:**")
                        
                        safe_dataframe_display(sample_df, f"Échantillon {selected_year}")
                        
                        if show_info:
                            st.markdown("#### 📊 Informations Détaillées du DataFrame")
                            
                            buffer = io.StringIO()
                            df_selected.info(buf=buffer)
                            info_str = buffer.getvalue()
                            
                            st.text(info_str)

            # ÉTAPE 3: Comparaison entre années
            st.markdown("---")
            st.subheader(" Étape 3 : Comparaison entre Années")
            
            if len(dataframes) > 1:
                st.markdown("#### 📊 Analyse comparative")
                
                comparison_data = []
                for year, df_year in sorted(dataframes.items()):
                    comparison_data.append({
                        'Année': year,
                        'Lignes': f"{len(df_year):,}",
                        'Colonnes': len(df_year.columns),
                        'Mémoire (MB)': f"{df_year.memory_usage(deep=True).sum() / 1024**2:.2f}",
                        'Valeurs manquantes': f"{df_year.isnull().sum().sum():,}",
                        'Doublons': f"{df_year.duplicated().sum():,}"
                    })
                
                df_comparison = pd.DataFrame(comparison_data)
                safe_dataframe_display(df_comparison, "Comparaison par année")
                
                # Analyse des colonnes communes et différentes
                all_columns = set()
                for df_year in dataframes.values():
                    all_columns.update(df_year.columns)
                
                common_columns = set(dataframes[list(dataframes.keys())[0]].columns)
                for df_year in list(dataframes.values())[1:]:
                    common_columns = common_columns.intersection(set(df_year.columns))
                
                if len(common_columns) < len(all_columns):
                    st.markdown("#### 🔍 Analyse des Différences de Colonnes")
                    
                    diff_analysis = []
                    for year, df_year in dataframes.items():
                        missing_cols = all_columns - set(df_year.columns)
                        extra_cols = set(df_year.columns) - common_columns
                        
                        diff_analysis.append({
                            'Année': year,
                            'Colonnes manquantes': len(missing_cols),
                            'Colonnes spécifiques': len(extra_cols),
                            'Détail manquantes': ', '.join(list(missing_cols)[:3]) + ('...' if len(missing_cols) > 3 else ''),
                            'Détail spécifiques': ', '.join(list(extra_cols)[:3]) + ('...' if len(extra_cols) > 3 else '')
                        })
                    
                    df_diff_analysis = pd.DataFrame(diff_analysis)
                    safe_dataframe_display(df_diff_analysis, "Différences de colonnes")
            
            else:
                st.info("ℹ️ Une seule année chargée. Comparaison non disponible.")

        except FileNotFoundError:
            st.error("Le dossier 'raw_data' n'a pas été trouvé. Veuillez créer un dossier nommé 'raw_data' et y placer vos fichiers CSV.")
        except Exception as e:
            st.error(f"Une erreur s'est produite lors du chargement des données : {e}")

        # Informations sur les prochaines étapes
        st.markdown("---")
        st.info("""
        **🚀 Prochaines étapes :**
        
        Naviguez vers l'onglet **Preprocessing** pour :
        -  Nettoyer et standardiser les données
        -  Harmoniser les colonnes entre les années
        -  Gérer les valeurs manquantes
        -  Préparer les données pour l'analyse
        -  Créer le DataFrame final unifié
        """)

    # ----------------------------------------------------------------
    # Onglet preprocessing (index 1)
    # ----------------------------------------------------------------
    with tabs[1]:
        st.markdown("Traitement des données et création de la 1ère phase du dataframe cible")
       
        if dataframes is None or not dataframes:
            st.info("⚠️ Veuillez d'abord charger les données dans l'onglet 'Chargement des Données'.")
        else:
            processed_dataframes = {year: df_year.copy() for year, df_year in dataframes.items()}

            # ÉTAPE 1: Nettoyage initial des doublons
            st.subheader(" Étape 1 : Suppression des Doublons")
            
            st.markdown("#### 🔍 Détails de la suppression des doublons")
            
            # Créer un tableau récapitulatif
            duplicates_data = []
            total_duplicates_removed = 0
            
            for year, df_cleaning in sorted(processed_dataframes.items()):
                longueur_initiale = len(df_cleaning)
                nb_lignes_dupliquees = df_cleaning.duplicated().sum()
                
                df_cleaning.drop_duplicates(inplace=True)
                longueur_finale = len(df_cleaning)
                removed = longueur_initiale - longueur_finale
                total_duplicates_removed += removed
                
                df_cleaning.reset_index(drop=True, inplace=True)
                processed_dataframes[year] = df_cleaning
                
                duplicates_data.append({
                    'Année': year,
                    'Lignes initiales': f"{longueur_initiale:,}",
                    'Doublons détectés': nb_lignes_dupliquees,
                    'Lignes finales': f"{longueur_finale:,}",
                    'Lignes supprimées': removed,
                    'Statut': '✅ Nettoyé' if removed > 0 else 'ℹ️ Aucun doublon'
                })
            
            # Afficher le tableau
            df_duplicates_summary = pd.DataFrame(duplicates_data)
            safe_dataframe_display(df_duplicates_summary, "Résumé suppression des doublons")
            
            # Résumé global
            st.markdown("#### 📊 Résumé Global")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🗑️ Total doublons supprimés", total_duplicates_removed)
            with col2:
                total_rows = sum(len(df) for df in processed_dataframes.values())
                st.metric("📈 Total lignes conservées", f"{total_rows:,}")

            st.markdown("---")

            # ÉTAPE 2: Ajout des colonnes manquantes
            st.subheader(" Étape 2 : Harmonisation des Colonnes")
            
            st.markdown("#### ➕ Ajout des colonnes manquantes")
            if '2015' in processed_dataframes:
                cols_added = []
                
                if "Carrosserie" not in processed_dataframes['2015'].columns:
                    processed_dataframes['2015']["Carrosserie"] = np.nan
                    cols_added.append("Carrosserie")
                
                if "gamme" not in processed_dataframes['2015'].columns:
                    processed_dataframes['2015']["gamme"] = np.nan
                    cols_added.append("gamme")
                
                if cols_added:
                    st.success(f"✅ Colonnes ajoutées à 2015 : {', '.join(cols_added)}")
                    for col in cols_added:
                        st.info(f"📋 '{col}' initialisée avec des valeurs NaN dans 2015")
                else:
                    st.info("ℹ️ Toutes les colonnes requises existent déjà dans 2015")
            else:
                st.warning("⚠️ DataFrame 2015 non trouvé")

            st.markdown("---")

            # Ajout de la colonne 'année' avec message après le tableau
            st.markdown("#### 📅 Ajout de la colonne 'année'")
            for year, df_year in processed_dataframes.items():
                df_year["année"] = int(year)
            
            year_counts = {}
            for year, df_year in processed_dataframes.items():
                year_counts[year] = len(df_year)
            
            df_year_summary = pd.DataFrame(list(year_counts.items()), 
                                         columns=['Année', 'Nombre de lignes'])
            safe_dataframe_display(df_year_summary, "Répartition par année")
            
            # Message après le tableau
            st.success("✅ Colonne 'année' ajoutée à tous les DataFrames")

            st.markdown("---")

            # ÉTAPE 3: Standardisation des noms de colonnes
            st.subheader(" Étape 3 : Mapping et uniformisation")
            
            column_mapping = {
                "Boîte de vitesse": "typ_boite_nb_rapp",
                "Carburant": "typ_crb",
                "Carrosserie": "Carrosserie",
                "Champ V9": "champ_v9",
                "CNIT" : "cnit",
                "CO type I (g/km)": "co_typ_1",
                "co_typ_1": "co_typ_1",
                "CO2 (g/km)": "co2",
                "co2": "co2",
                "co2_mixte": "co2",
                "cod_cbr": "typ_crb",
                "conso_urb_93": "conso_urb",
                "Consommation extra-urbaine (l/100km)": "conso_exurb",
                "Consommation mixte (l/100km)": "conso_mixte",
                "Consommation urbaine (l/100km)": "conso_urb",
                "Date de mise à jour": "date_maj",
                "Désignation commerciale": "dscom",
                "dscom": "dscom",
                "energ": "typ_crb",
                "HC (g/km)": "hc",
                "HC+NOX (g/km)": "hcnox",
                "hibride": "hybride",
                "Hybride" : "hybride",
                "lib_mod": "lib_mod",
                "lib_mrq": "lib_mrq_utac",
                "lib_mrq_doss": "lib_mrq_utac",
                "Marque": "lib_mrq_utac",
                "masse vide euro max (kg)": "masse_ordma_max",
                "masse vide euro min (kg)": "masse_ordma_min",
                "mod_utac": "lib_mod",
                "Modèle dossier": "lib_mod_doss",
                "Modèle UTAC": "lib_mod",
                "mrq_utac": "lib_mrq_utac",
                "NOX (g/km)": "nox",
                "Particules (g/km)": "ptcl",
                "puiss_admin_98": "puiss_admin",
                "puiss_heure": "puiss_h",
                "Puissance administrative": "puiss_admin",
                "Puissance maximale (kW)": "puiss_max",
                "typ_cbr": "typ_crb",
                "typ-crb" : "typ_crb",
                "Type Variante Version (TVV)": "tvv",
            }

            def standardize_column_names(df, column_mapping):
                df = df.copy()
                new_columns = df.columns.to_list()
                for keys, new_name in column_mapping.items():
                    if isinstance(keys, str):
                        keys = [keys]
                    for key in keys:
                        if key in new_columns:
                            new_columns[new_columns.index(key)] = new_name
                df.columns = new_columns
                return df

            mapping_results = {}
            for year, df_year in processed_dataframes.items():
                old_cols = set(df_year.columns)
                processed_dataframes[year] = standardize_column_names(df_year, column_mapping)
                new_cols = set(processed_dataframes[year].columns)
                
                renamed = len(old_cols.intersection(column_mapping.keys()))
                mapping_results[year] = {
                    'colonnes_avant': len(old_cols),
                    'colonnes_après': len(new_cols),
                    'colonnes_renommées': renamed
                }
            
            df_mapping_results = pd.DataFrame(mapping_results).T
            safe_dataframe_display(df_mapping_results, "Résultats du mapping")
            
            st.success("✅ Standardisation des noms de colonnes terminée")

            st.markdown("---")

            # ÉTAPE 4: Classement alphabétique
            st.subheader(" Étape 4 : Classement Alphabétique des Colonnes")
            
            def ordonner_colonnes_alphabetiquement(df):
                try:
                    colonnes_ordonnees = sorted(df.columns)
                    df_ordonne = df[colonnes_ordonnees]
                    return df_ordonne
                except Exception as e:
                    st.error(f"Erreur lors du classement des colonnes : {e}")
                    return df

            for year in processed_dataframes.keys():
                processed_dataframes[year] = ordonner_colonnes_alphabetiquement(processed_dataframes[year])
            
            st.success("✅ Colonnes classées par ordre alphabétique pour tous les DataFrames")

            st.markdown("---")

            # ÉTAPE 6: Vérification de l'uniformité
            st.subheader(" Étape 5 : Vérification de l'Uniformité des Colonnes")
            
            st.markdown("#### ⚖️ Analyse des différences entre DataFrames")
            
            def afficher_differences_streamlit(ensemble1, ensemble2, nom1, nom2):
                differences = ensemble1.symmetric_difference(ensemble2)
                if differences:
                    st.warning(f"⚠️ Différences entre {nom1} et {nom2} : {differences}")
                    return False
                else:
                    st.success(f"✅ Aucune différence entre {nom1} et {nom2}")
                    return True

            sets_cols = {}
            for year, df_year in processed_dataframes.items():
                sets_cols[year] = set(df_year.columns)

            all_same = True
            years_list = sorted(list(processed_dataframes.keys()))
            
            if len(years_list) > 1:
                first_year = years_list[0]
                first_set = sets_cols[first_year]
                
                for i in range(1, len(years_list)):
                    current_year = years_list[i]
                    is_same = afficher_differences_streamlit(first_set, sets_cols[current_year],
                                                           first_year, current_year)
                    if not is_same:
                        all_same = False

            # Gestion spécifique de puiss_h dans 2015
            if '2015' in processed_dataframes and 'puiss_h' in processed_dataframes['2015'].columns:
                puiss_h_in_others = any('puiss_h' in processed_dataframes[year].columns 
                                      for year in processed_dataframes if year != '2015')
                if not puiss_h_in_others:
                    processed_dataframes['2015'] = processed_dataframes['2015'].drop(columns=["puiss_h"], errors="ignore")
                    st.info("🗑️ Colonne 'puiss_h' supprimée de 2015 (absente des autres années)")

            # Re-vérification après ajustements
            st.markdown("#### 🔄 Vérification finale après ajustements")
            sets_cols_final = {}
            for year, df_year in processed_dataframes.items():
                sets_cols_final[year] = set(df_year.columns)
            
            all_same_final = True
            
            if len(years_list) > 1:
                first_set_final = sets_cols_final[years_list[0]]
                for i in range(1, len(years_list)):
                    current_year = years_list[i]
                    is_same = afficher_differences_streamlit(first_set_final, sets_cols_final[current_year],
                                                           years_list[0], current_year)
                    if not is_same:
                        all_same_final = False

            if all_same_final:
                st.success("🎉 Tous les DataFrames ont maintenant les mêmes colonnes ! Concaténation possible.")
            else:
                st.error("❌ Des différences persistent entre les DataFrames")

            st.markdown("---")

            # ÉTAPE 7: Détection des colonnes dupliquées
            st.subheader(" Étape 6 : Détection des Colonnes Dupliquées")
            
            st.markdown("#### 🔄 Vérification et suppression des doublons de colonnes")
            
            def trouver_doublons(df, nom_df):
                colonnes_dupliquees = df.columns[df.columns.duplicated()].tolist()
                if colonnes_dupliquees:
                    st.warning(f"⚠️ Colonnes dupliquées dans {nom_df} : {colonnes_dupliquees}")
                    processed_dataframes[nom_df.split('_')[1]] = df.loc[:, ~df.columns.duplicated()]
                    st.success(f"✅ Colonnes dupliquées supprimées de {nom_df}")
                    return len(colonnes_dupliquees)
                else:
                    st.success(f"✅ Aucune colonne dupliquée dans {nom_df}")
                    return 0

            total_duplicated_cols = 0
            for year, df_year in processed_dataframes.items():
                duplicated_count = trouver_doublons(df_year, f"df_{year}")
                total_duplicated_cols += duplicated_count

            if total_duplicated_cols == 0:
                st.success("🎉 Aucune colonne dupliquée détectée dans l'ensemble des DataFrames")

            st.markdown("---")

            # ÉTAPE 8: Concaténation finale
            if all_same_final and len(processed_dataframes) > 0:
                st.subheader(" Étape 7 : Création du DataFrame final")
                
                df_final = pd.concat(processed_dataframes.values(), ignore_index=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📊 Lignes totales", f"{len(df_final):,}")
                
                with col2:
                    st.metric("📋 Colonnes", f"{len(df_final.columns):,}")
                
                with col3:
                    st.metric("📅 Années", f"{df_final['année'].nunique()}")
                
                with col4:
                    memory_usage = df_final.memory_usage(deep=True).sum() / 1024**2
                    st.metric("💾 Taille", f"{memory_usage:.1f} MB")
                
                st.success("🎉 DataFrame final concaténé créé avec succès !")

            st.markdown("---")

            # ÉTAPE 9: Analyse des valeurs manquantes
            if 'df_final' in locals():
                st.subheader(" Étape 8 : Analyse des Valeurs Manquantes")
                
                st.markdown("#### 📊 État des valeurs manquantes")
                
                nan_counts = df_final.isnull().sum()
                nan_percentage = df_final.isnull().mean() * 100
                nan_summary = pd.DataFrame({
                    'Nombre de NaN': nan_counts, 
                    'Pourcentage de NaN': nan_percentage
                })
                
                safe_dataframe_display(nan_summary, "Analyse des valeurs manquantes")
                
                # Graphique Plotly au lieu de matplotlib
                if not nan_summary.empty:
                    nan_summary_filtered = nan_summary[nan_summary['Nombre de NaN'] > 0].sort_values(
                        by='Pourcentage de NaN', ascending=False)
                    
                    if len(nan_summary_filtered) > 0:
                        fig_nan = px.bar(
                            x=nan_summary_filtered.index,
                            y=nan_summary_filtered['Pourcentage de NaN'],
                            title="Pourcentage de valeurs manquantes par colonne",
                            labels={'x': 'Colonnes', 'y': 'Pourcentage de NaN (%)'},
                            color=nan_summary_filtered['Pourcentage de NaN'],
                            color_continuous_scale='Reds'
                        )
                        fig_nan.update_layout(
                            xaxis_tickangle=-45,
                            height=500
                        )
                        st.plotly_chart(fig_nan, use_container_width=True)

            st.markdown("---")

            # ÉTAPE 10: Conversion des types de données
            if 'df_final' in locals():
                st.subheader(" Étape 9 : Conversion des Types de Données vers types numériques")
                
                cols_numeriques = ["puiss_max", "conso_urb", "conso_exurb", "conso_mixte", "co_typ_1", "nox", "hcnox", "ptcl", "co2"]

                def convert_numeric(value):
                    if isinstance(value, str):
                        value = value.replace(",", ".")
                        try:
                            return float(value)
                        except ValueError:
                            return np.nan
                    return value

                conversion_results = {}
                for col in cols_numeriques:
                    if col in df_final.columns:
                        before_type = str(df_final[col].dtype)
                        before_nulls = df_final[col].isnull().sum()
                        
                        df_final[col] = df_final[col].apply(convert_numeric)
                        
                        after_type = str(df_final[col].dtype)
                        after_nulls = df_final[col].isnull().sum()
                        
                        conversion_results[col] = {
                            'Type avant': before_type,
                            'Type après': after_type,
                            'NaN avant': before_nulls,
                            'NaN après': after_nulls,
                            'Nouveaux NaN': after_nulls - before_nulls
                        }

                if conversion_results:
                    df_conversion = pd.DataFrame(conversion_results).T
                    safe_dataframe_display(df_conversion, "Résultats des conversions")
                    st.success("✅ Conversion des colonnes numériques terminée")

            st.markdown("---")

            # ÉTAPE 11: Nettoyage des données catégorielles
            if 'df_final' in locals():
                st.subheader(" Étape 10 : Standardisation des variables catégorielles")
                
                # Nettoyer 'gamme'
                if "gamme" in df_final.columns:
                    st.markdown("##### Nettoyage de la colonne 'gamme'")
                    before_unique = df_final["gamme"].nunique()
                    
                    df_final["gamme"] = df_final["gamme"].str.upper().str.strip()

                    corrections_gamme = {
                        "MOY-INF": "MOY-INFER",
                        "MOY-INFERIEURE": "MOY-INFER"
                    }
                    df_final["gamme"] = df_final["gamme"].replace(corrections_gamme)
                    
                    after_unique = df_final["gamme"].nunique()
                    st.success(f"✅ 'gamme' nettoyée : {before_unique} → {after_unique} valeurs uniques")
                    
                    gamme_counts = df_final["gamme"].value_counts().head()
                    safe_dataframe_display(gamme_counts.to_frame(), "Top valeurs 'gamme'")

                # Nettoyer 'Carrosserie'
                if "Carrosserie" in df_final.columns:
                    st.markdown("##### Nettoyage de la colonne 'Carrosserie'")
                    before_unique = df_final["Carrosserie"].nunique()
                    
                    df_final["Carrosserie"] = df_final["Carrosserie"].str.upper().str.strip()

                    corrections_carrosserie = {
                        "COMBISPCACE": "COMBISPACE"
                    }
                    df_final["Carrosserie"] = df_final["Carrosserie"].replace(corrections_carrosserie)
                    
                    after_unique = df_final["Carrosserie"].nunique()
                    st.success(f"✅ 'Carrosserie' nettoyée : {before_unique} → {after_unique} valeurs uniques")

                # Nettoyer 'typ_crb'
                if "typ_crb" in df_final.columns:
                    st.markdown("##### Nettoyage de la colonne 'typ_crb'")
                    before_unique = df_final["typ_crb"].nunique()
                    
                    df_final["typ_crb"] = df_final["typ_crb"].str.upper().str.strip()

                    corrections_typ_crb = {
                        "GO ": "GO", "ES ": "ES", "EH ": "EH", "GH ": "GH", "GN ": "GN", "EE ": "EE",
                        "ES/GN": "GN/ES", "GP/ES": "ES/GP"
                    }
                    df_final["typ_crb"] = df_final["typ_crb"].replace(corrections_typ_crb)
                    
                    after_unique = df_final["typ_crb"].nunique()
                    st.success(f"✅ 'typ_crb' nettoyée : {before_unique} → {after_unique} valeurs uniques")
                    
                    typ_crb_counts = df_final["typ_crb"].value_counts().head()
                    safe_dataframe_display(typ_crb_counts.to_frame(), "Distribution 'typ_crb'")

            st.markdown("---")

            # ÉTAPE 12: Imputation intelligente
            if 'df_final' in locals():
                st.subheader(" Étape 11 : Imputation Intelligente des Valeurs Manquantes")
                
                # Imputation spécifique pour 2015
                st.markdown("#### 📅 Imputation spéciale pour l'année 2015")
                
                df_train = df_final[df_final["année"] < 2015]

                mapping_carrosserie = df_train.groupby("gamme")["Carrosserie"].agg(lambda x: x.value_counts().idxmax())
                mapping_gamme = df_train.groupby("Carrosserie")["gamme"].agg(lambda x: x.value_counts().idxmax())

                df_2015 = df_final[df_final["année"] == 2015]

                if not df_train.empty and not df_2015.empty:
                    gamme_sample = df_train["gamme"].dropna().sample(n=len(df_2015), replace=True, random_state=42).values

                    df_final.loc[df_final["année"] == 2015, "gamme"] = gamme_sample
                    df_final.loc[df_final["année"] == 2015, "Carrosserie"] = df_final.loc[df_final["année"] == 2015, "gamme"].map(mapping_carrosserie)
                    
                    st.success("✅ Imputation 2015 : gamme et Carrosserie basées sur la distribution historique")

                # Imputation par groupe pour variables numériques
                st.markdown("#### 🔢 Imputation des variables numériques par groupe")
                
                # hcnox
                if "hcnox" in df_final.columns:
                    before_nulls = df_final["hcnox"].isna().sum()
                    df_final["hcnox"] = df_final.groupby("typ_crb")["hcnox"].transform(lambda x: x.fillna(x.median()))
                    df_final["hcnox"].fillna(df_final["hcnox"].median(), inplace=True)
                    after_nulls = df_final["hcnox"].isna().sum()
                    st.success(f"✅ hcnox: {before_nulls} → {after_nulls} NaN (imputation par groupe typ_crb)")

                # ptcl
                if "ptcl" in df_final.columns:
                    before_nulls = df_final["ptcl"].isna().sum()
                    df_final["ptcl"] = df_final.groupby("typ_crb")["ptcl"].transform(lambda x: x.fillna(x.median()))
                    df_final["ptcl"].fillna(df_final["ptcl"].median(), inplace=True)
                    after_nulls = df_final["ptcl"].isna().sum()
                    st.success(f"✅ ptcl: {before_nulls} → {after_nulls} NaN (imputation par groupe typ_crb)")

                # Imputation finale
                st.markdown("#### 🎯 Imputation finale")
                
                cols_numeriques = ["co2","co_typ_1","conso_exurb","conso_mixte","conso_urb","nox","puiss_max"]
                cols_categorielles = ["champ_v9"] 

                numeric_imputed = 0
                for col in cols_numeriques:
                    if col in df_final.columns and df_final[col].isnull().any():
                        before_nulls = df_final[col].isnull().sum()
                        df_final[col] = df_final[col].fillna(df_final[col].mean())
                        numeric_imputed += before_nulls

                categorical_imputed = 0
                for col in cols_categorielles:
                    if col in df_final.columns and df_final[col].isnull().any():
                        before_nulls = df_final[col].isnull().sum()
                        mode_val = df_final[col].mode()
                        if not mode_val.empty:
                            df_final[col] = df_final[col].fillna(mode_val[0])
                            categorical_imputed += before_nulls

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🔢 NaN numériques imputés", numeric_imputed)
                with col2:
                    st.metric("🏷️ NaN catégoriels imputés", categorical_imputed)
                with col3:
                    remaining_nulls = df_final.isnull().sum().sum()
                    st.metric("🎯 NaN restants", remaining_nulls)

                if remaining_nulls == 0:
                    st.success("🎉 Toutes les valeurs manquantes ont été imputées avec succès !")

            st.markdown("---")

            # ÉTAPE 13: Nettoyage lib_mrq_utac
            if 'df_final' in locals():
                st.subheader(" Étape 12 : Nettoyage des Marques (lib_mrq_utac)")
                
                if "lib_mrq_utac" in df_final.columns:
                    before_unique = df_final["lib_mrq_utac"].nunique()
                    
                    df_final["lib_mrq_utac"] = df_final["lib_mrq_utac"].str.strip()
                    replace_dict = {
                        "BMW I": "BMW",
                        "BMW ": "BMW",
                        "MERCEDES AMG": "MERCEDES",
                        "MERCEDES BENZ": "MERCEDES",
                        "MERCEDES-BENZ": "MERCEDES",
                        "MERCEDES ": "MERCEDES",
                        "ALFA-ROMEO": "ALFA ROMEO",
                        "ROLLS-ROYCE" : "ROLLS ROYCE",
                        "LAND ROVER": "JAGUAR LAND ROVER LIMITED",
                        "RENAULT TECH": "RENAULT",
                        "RENAULT ": "RENAULT",
                        "FORD-CNG-TECHNIK" : "FORD",
                        "VOLKSWAGEN-TECHNIK" : "VOLKSWAGEN",
                    }

                    df_final["lib_mrq_utac"] = df_final["lib_mrq_utac"].replace(replace_dict)
                    
                    after_unique = df_final["lib_mrq_utac"].nunique()
                    st.success(f"✅ lib_mrq_utac nettoyée : {before_unique} → {after_unique} marques uniques")

            st.markdown("---")

            # ÉTAPE 14: Suppression des colonnes administratives
            if 'df_final' in locals():
                st.subheader(" Étape 13 : Suppression des Colonnes Administratives")
                
                df_final_ml = df_final.copy()
                
                to_remove = [
                    "champ_v9", "cnit", "date_maj", "dscom", "hc", 
                    "hybride", "lib_mod", "lib_mod_doss", "tvv"
                ]

                cols_before = df_final.columns.tolist()
                df_final_ml.drop(columns=to_remove, inplace=True, errors="ignore")
                cols_after = df_final_ml.columns.tolist()
                removed = [c for c in cols_before if c not in cols_after]
                
                st.success(f"✅ Colonnes administratives supprimées : {', '.join(removed)}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Colonnes avant", len(cols_before))
                with col2:
                    st.metric("🗑️ Colonnes supprimées", len(removed))
                with col3:
                    st.metric("✅ Colonnes finales", len(cols_after))

            st.markdown("---")

            # avec sauvegarde
            if 'df_final' in locals() and 'df_final_ml' in locals():
                st.subheader(" Étape 14 : Sauvegarde du dataframe post preprocessing")
                
                st.session_state['df_final'] = df_final
                st.session_state['df_final_ml'] = df_final_ml
                st.session_state['preprocessing_completed'] = True
                
                st.success("✅ Preprocessing terminé et stocké pour les autres onglets !")
                
                st.markdown("---")
                st.markdown("""
                ### 🎯 Points Clés du Preprocessing Réalisé
                
                **✅ Qualité des données :**
                - Suppression de tous les doublons
                - Harmonisation complète des colonnes entre années
                - Imputation intelligente de toutes les valeurs manquantes
                - Suppression des colonnes non pertinentes (phase 1)
                
                **✅ Standardisation :**
                - Noms de colonnes uniformisés (grâce à un mapping)
                - Variables catégorielles nettoyées et standardisées
                - Types de données convertis correctement
                """)
                
        # Informations sur les prochaines étapes
        st.markdown("---")
        st.info("""
        **🚀 Prochaines étapes :**
        
        Naviguez vers l'onglet **Visualisation** pour :
        - Explorer les corrélations entre variables
        - Analyser la relation puissance-CO2
        - Étudier l'impact du type de carburant
        - Analyser les outliers
        """)

    # ----------------------------------------------------------------
    # Onglet Visualisation des Données (index 2)
    # ----------------------------------------------------------------
    with tabs[2]:
        st.markdown("Data Visualisation pour la compréhension des relations entre les données et prise de décision quant au Feature Engineering")
        
        if 'df_final' in st.session_state:
            df_final = st.session_state['df_final']
        else:
            df_final = None
        
        viz_tabs = st.tabs(["Aperçu Rapide du Projet", "Analyse de Corrélation", "Analyse des Outliers"])

        # Onglet 1 : Aperçu Rapide du Projet
        with viz_tabs[0]:
            if df_final is None or df_final.empty:
                st.info("Exécutez d'abord l'onglet 'Preprocessing' pour construire df_final.")
            else:
                # 1) Relation puissance administrative et CO2 - MODIFICATION: Plotly
                st.markdown("### Relation entre Puissance Administrative et CO2")
                
                if "puiss_admin" in df_final.columns and "co2" in df_final.columns:
                    # Créer le graphique avec Plotly
                    fig_power = px.scatter(
                        df_final,
                        x="puiss_admin",
                        y="co2",
                        title="Relation entre la puissance administrative et le CO2",
                        labels={"puiss_admin": "Puissance Administrative", "co2": "CO2 (g/km)"},
                        opacity=0.6,
                        color_discrete_sequence=['steelblue']
                    )
                    
                    fig_power.update_layout(
                        height=500,
                        title_x=0.5
                    )
                    
                    st.plotly_chart(fig_power, use_container_width=True)
                    
                    corr, p_value = pearsonr(df_final["puiss_admin"].dropna(), 
                                        df_final["co2"].dropna())
                    
                    st.markdown(f"""
                    **📈 Analyse du Pearson :**
                    - **Corrélation de Pearson** : **{corr:.3f}** (p-value: {p_value:.2e})
                    - **Interprétation** : {'Forte' if abs(corr) > 0.7 else 'Modérée' if abs(corr) > 0.5 else 'Faible'} corrélation positive
                    - **Tendance claire** : Les émissions de CO2 augmentent avec la puissance administrative
                    - **Logique physique** : Véhicules plus puissants = moteurs plus grands = plus de consommation = plus de pollution
                    """)
                
                st.markdown("---")
                
                # 2) Impact du type de carburant sur les émissions
                st.markdown("### Impact du Type de Carburant sur les Émissions CO2")
                
                if "typ_crb" in df_final.columns:
                    fig_fuel_box = px.box(
                        df_final,
                        x="typ_crb",
                        y="co2",
                        title="Distribution des émissions CO2 par type de carburant",
                        labels={"typ_crb": "Type de carburant", "co2": "CO2 (g/km)"},
                        color="typ_crb",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig_fuel_box.update_layout(
                        xaxis_title="Type de carburant",
                        yaxis_title="CO2 (g/km)",
                        title_x=0.5,
                        showlegend=False
                    )
                    st.plotly_chart(fig_fuel_box, use_container_width=True)
                    
                                        
                    try:
                        groupes = [df_final[df_final["typ_crb"] == cat]["co2"].dropna() 
                                for cat in df_final["typ_crb"].unique() if not pd.isna(cat)]
                        anova_result = f_oneway(*groupes)
                        
                        st.markdown(f"""
                        **📊 Analyse de l'ANOVA :**
                        - **F-statistic** : **{anova_result.statistic:.2f}**
                        - **P-value** : **{anova_result.pvalue:.2e}**
                        - **Conclusion** : {'Différence très significative' if anova_result.pvalue < 0.001 else 'Différence significative' if anova_result.pvalue < 0.05 else 'Pas de différence significative'} entre les types de carburant
                        """)
                        
                    except Exception as e:
                        st.warning(f"Impossible de calculer l'ANOVA : {e}")

        # Onglet 2 : Analyse de Corrélation
        with viz_tabs[1]:
            if df_final is None or df_final.empty:
                st.info("Exécutez d'abord l'onglet 'Preprocessing' pour construire df_final avant d'analyser les corrélations.")
            else:
                # 1) Heatmap des corrélations numériques
                st.markdown("### Corrélation des Variables Numériques")
                df_num = df_final.select_dtypes(include=np.number).dropna(axis=1, how='all')

                if df_num.shape[1] > 1:
                    corr_matrix = df_num.corr()
                    fig6, ax6 = plt.subplots(figsize=(12, 10))
                    sns.heatmap(corr_matrix,
                                annot=True, fmt=".2f",
                                cmap="coolwarm", linewidths=0.5,
                                ax=ax6, center=0)
                    ax6.set_title("Heatmap des corrélations numériques")
                    plt.tight_layout()
                    st.pyplot(fig6)
                    
                    # Analyse de la multicolinéarité
                                        
                    high_corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            corr_val = corr_matrix.iloc[i, j]
                            if abs(corr_val) > 0.8:
                                high_corr_pairs.append({
                                    'Variable 1': corr_matrix.columns[i],
                                    'Variable 2': corr_matrix.columns[j],
                                    'Corrélation': corr_val
                                })
                    
                    if high_corr_pairs:
                        df_high_corr = pd.DataFrame(high_corr_pairs)
                        df_high_corr = df_high_corr.sort_values('Corrélation', key=abs, ascending=False)
                        
                        st.warning("**⚠️Impact sur la modélisation: risque de multicolinéarité détecté !**")
                        safe_dataframe_display(df_high_corr, "Corrélations élevées (|r| > 0.8)")
                        
                        st.markdown("""
                        **📋 Interprétation :**
                        - **Variables de consommation** (`conso_urb`, `conso_exurb`, `conso_mixte`) : très corrélées entre elles et avec CO2
                        - **Variables de masse** (`masse_ordma_min`, `masse_ordma_max`) : corrélation très élevée (redondance)
                        - **Variables de puissance** (`puiss_admin`, `puiss_max`) : corrélation élevée
                        
                        ➡️ **Solution** : supprimer ces variables redondantes dans l'étape Feature Engineering
                        """)
                        
                        multicollinear_vars = ["conso_exurb", "conso_mixte", "conso_urb", 
                                             "masse_ordma_max", "masse_ordma_min", 
                                             "puiss_admin", "puiss_max"]
                        st.session_state['multicollinear_vars'] = multicollinear_vars
                        
                    else:
                        st.success("✅ Aucune corrélation problématique détectée (seuil |r| > 0.8)")
                    
                else:
                    st.info("Pas assez de colonnes numériques (au moins 2) disponibles après traitement pour générer une heatmap de corrélation.")

                # 2) Analyse des corrélations entre variables catégorielles avec Cramer's V
                st.markdown('---')
                st.markdown("### Corrélations entre Variables Catégorielles (Cramer's V)")
                categories = ['Carrosserie', 'gamme', 'lib_mod_doss', 'lib_mrq_utac', 'typ_boite_nb_rapp', 'typ_crb']
                categories_available = [col for col in categories if col in df_final.columns]
                if len(categories_available) >= 2:
                    cramer_matrix = pd.DataFrame(np.zeros((len(categories_available), len(categories_available))), index=categories_available, columns=categories_available)
                    for col1 in categories_available:
                        for col2 in categories_available:
                            if col1 != col2:
                                try:
                                    cramer_matrix.loc[col1, col2] = cramers_v(df_final[col1], df_final[col2])
                                except:
                                    cramer_matrix.loc[col1, col2] = 0
                            else:
                                cramer_matrix.loc[col1, col2] = 1
                    cramer_matrix = cramer_matrix.astype(float)
                    (fig_cramer, ax_cramer) = plt.subplots(figsize=(10, 8))
                    sns.heatmap(cramer_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax_cramer)
                    ax_cramer.set_title("Heatmap de Cramer's V (corrélation entre variables catégorielles)")
                    plt.tight_layout()
                    st.pyplot(fig_cramer)
                    
                    st.warning("⚠️ **Impact sur la modélisation : associations fortes révélées par le Cramer's V qui s'expliquent par la logique métier du secteur automobile.**")
                    
                    strong_associations = []
                    for i in range(len(cramer_matrix.columns)):
                        for j in range(i + 1, len(cramer_matrix.columns)):
                            cramer_val = cramer_matrix.iloc[i, j]
                            if cramer_val > 0.3:
                                strong_associations.append({'Variable 1': cramer_matrix.columns[i], 'Variable 2': cramer_matrix.columns[j], "Cramer's V": cramer_val})
                    if strong_associations:
                        df_strong_assoc = pd.DataFrame(strong_associations)
                        df_strong_assoc = df_strong_assoc.sort_values("Cramer's V", ascending=False)
                        safe_dataframe_display(df_strong_assoc, 'Associations catégorielles fortes')
                        
                        st.markdown('**📋 Interprétation :**')
                        st.markdown("""
                                                
                        - **Marque ↔ Gamme** : chaque constructeur a sa propre segmentation (premium, généraliste, etc.)
                        - **Carrosserie ↔ Gamme** : les berlines sont souvent en gamme supérieure, les citadines en gamme inférieure  
                        - **Type de carburant ↔ Gamme** : les véhicules haut de gamme adoptent plus facilement l'hybride/électrique
                        - **Marque ↔ Type de boîte** : certains constructeurs privilégient l'automatique (premium) ou le manuel (généraliste)
                        """)
                        
                        st.markdown("**➡️ Solution :** mise en place d'une stratégie d'encodage différenciée par variable dans l'étape Feature Engineering :")
                        st.markdown("""
                                                
                        - `typ_crb` : regroupement en 3 catégories principales (`GO`/`ES`/`Autres`) puis OneHot
                        - `lib_mrq_utac` : encodage fréquentiel (trop de modalités pour OneHot)  
                        - `Carrosserie`/`gamme` : OneHot avec drop='first' pour éviter la redondance
                        - `typ_boite_nb_rapp` : encodage fréquentiel (modalités nombreuses et variées)
                      
                                    
                        """)

                        st.markdown('---')
                        st.markdown('### Conclusions générales')
                        st.success("""
                        
                        - **Variables numériques :** la multicolinéarité détectée est problématique et sera traitée dans le Feature Engineering pour ne pas fausser les résultats de nos modèles de régression.

                        - **Variables catégorielles :** les associations détectées sont normales et attendues dans le contexte automobile. Notre stratégie d'encodage différencié permet de les exploiter efficacement sans créer de problèmes de dimensionnalité.
                        """)

                else:
                    st.info(f"Pas assez de variables catégorielles disponibles pour l'analyse Cramer's V. Variables trouvées : {categories_available}")

        # Onglet 3 : Analyse des Outliers
        with viz_tabs[2]:
                            
            if df_final is None or df_final.empty:
                st.info("Exécutez d'abord l'onglet 'Preprocessing' pour construire df_final avant d'analyser les outliers.")
            else:
                
                st.markdown("#### Identification des valeurs aberrantes")
                
                numerical_cols = df_final.select_dtypes(include=['float64', 'int64']).columns

                Q1 = df_final[numerical_cols].quantile(0.25)
                Q3 = df_final[numerical_cols].quantile(0.75)
                IQR = Q3 - Q1

                outliers = ((df_final[numerical_cols] < (Q1 - 1.5 * IQR)) | (df_final[numerical_cols] > (Q3 + 1.5 * IQR)))

                outliers_count = outliers.sum()
                
                df_outliers = pd.DataFrame({
                    'Colonne': outliers_count.index,
                    'Nombre d\'outliers': outliers_count.values,
                    'Pourcentage': (outliers_count.values / len(df_final) * 100).round(2)
                })
                
                safe_dataframe_display(df_outliers, "Analyse des outliers")
                
                fig_outliers = px.bar(
                    df_outliers.sort_values('Pourcentage', ascending=False).head(10),
                    x='Pourcentage',
                    y='Colonne',
                    orientation='h',
                    title="Top 10 des variables avec le plus d'outliers",
                    labels={'Pourcentage': 'Pourcentage d\'outliers (%)', 'Colonne': 'Variables'},
                    color='Pourcentage',
                    color_continuous_scale='Reds'
                )
                fig_outliers.update_layout(height=500)
                st.plotly_chart(fig_outliers, use_container_width=True)
                
                if "puiss_admin" in df_final.columns:
                    high_power_cars = df_final[df_final["puiss_admin"] > 60]
                    if len(high_power_cars) > 0 and "lib_mrq_utac" in df_final.columns:
                        brand_counts = high_power_cars["lib_mrq_utac"].value_counts().head(10)
                                
                        fig_brands = px.bar(
                            x=brand_counts.values, 
                            y=brand_counts.index, 
                            orientation='h',
                            title="Top 10 des marques avec puiss_admin > 60",
                            labels={'x': 'Nombre de véhicules', 'y': 'Marque (lib_mrq_utac)'},
                            color=brand_counts.values,
                            color_continuous_scale='viridis'
                            )
                        fig_brands.update_layout(
                            height=500,
                            showlegend=False,
                            coloraxis_showscale=False
                            )
                        st.plotly_chart(fig_brands, use_container_width=True)
                
                # Boxplots pour visualiser les outliers
                st.markdown("#### Boxplots des variables numériques")
                
                numerical_cols_for_boxplot = df_final.select_dtypes(include=["int64", "float64"]).columns
                
                if len(numerical_cols_for_boxplot) > 0:
                                        
                    n_cols = min(3, len(numerical_cols_for_boxplot))
                    n_rows = (len(numerical_cols_for_boxplot) + n_cols - 1) // n_cols
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
                    if n_rows == 1:
                        axes = [axes] if n_cols == 1 else axes
                    else:
                        axes = axes.flatten()
                    
                    for i, col in enumerate(numerical_cols_for_boxplot):
                        if i < len(axes):
                            sns.boxplot(x=df_final[col], ax=axes[i])
                            axes[i].set_title(f"Boxplot de {col}")
                    
                    for i in range(len(numerical_cols_for_boxplot), len(axes)):
                        axes[i].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Aucune colonne numérique trouvée pour l'analyse en boxplot.")

                # Conclusions
                st.markdown("#### Conclusions générales")
                
                st.markdown("""
                **Observations principales :**
                - Certaines variables présentent des outliers significatifs
                - Ce outliers s'expliquent (exemple : véhicules de forte puissance sont souvent des véhicules de luxe ou sportifs)
                - Les outliers peuvent représenter des segments spécifiques du marché automobile
                
                **Stratégies de traitement :**
                - Certaines colonnes seront supprimées car déjà identifiées dans l'analyse des corrélations (`conso_urb`,`conso_mixte`,`conso_exurb`, etc...)
                - Pas d'actions spécifiques prévues pour les autres hormis des opérations d'encodage
                
                **Impact sur la modélisation :**
                - Cela nous a aider à identifier les modèles à utiliser qui gèrent mieux les outliers (Random Forest, XGBoost)
                """)


            st.markdown("---")
            st.info("""
            **🚀 Prochaines étapes :**
            
            Naviguez vers l'onglet **Feature Engineering** pour :
            - Supprimer les variables multicolinéaires identifiées
            - Séparer les données d'entraînement et de test
            - Encoder les variables catégorielles
            - Standardiser les variables numériques
            - Préparer les données finales pour la modélisation
            """)

    # ----------------------------------------------------------------
    # Onglet Feature Engineering (index 3)
    # ----------------------------------------------------------------
    with tabs[3]:
        st.markdown("Amélioration et réalisation de la phase 2 de notre dataframe cible")
        
        if 'df_final_ml' in st.session_state:
            df_final_ml = st.session_state['df_final_ml']
        else:
            df_final_ml = None

        if df_final_ml is None or df_final_ml.empty:
            st.info("Veuillez d'abord exécuter l'onglet 'Preprocessing' pour obtenir df_final_ml.")
        else:
            # ÉTAPE 1: Suppression des variables multicolinéaires
            st.subheader(" Étape 1 : Suppression des Variables Multicolinéaires")
            
            multicollinear_vars = ["conso_exurb", "conso_mixte", "conso_urb", 
                                 "masse_ordma_max", "masse_ordma_min", 
                                 "puiss_admin", "puiss_max"]
            
            existing_multicol_vars = [var for var in multicollinear_vars if var in df_final_ml.columns]
            
            if existing_multicol_vars:
                if len(existing_multicol_vars) > 1:
                    corr_before = df_final_ml[existing_multicol_vars + ['co2']].corr()['co2'].drop('co2')
                    
                    df_corr_display = pd.DataFrame({
                        'Variable': corr_before.index,
                        'Corrélation avec CO2': corr_before.values.round(3),
                        'Statut': ['🗑️ À supprimer'] * len(corr_before)
                    })
                    
                    safe_dataframe_display(df_corr_display, "Variables multicolinéaires")
                
                cols_before_multicol = df_final_ml.columns.tolist()
                df_final_ml = df_final_ml.drop(columns=existing_multicol_vars, errors='ignore')
                cols_after_multicol = df_final_ml.columns.tolist()
                
                removed_multicol = [c for c in cols_before_multicol if c not in cols_after_multicol]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Colonnes avant", len(cols_before_multicol))
                with col2:
                    st.metric("🗑️ Variables supprimées", len(removed_multicol))
                with col3:
                    st.metric("✅ Colonnes restantes", len(cols_after_multicol))
                
                st.success(f"✅ Variables multicolinéaires supprimées : {', '.join(removed_multicol)}")
                
            else:
                st.info("ℹ️ Aucune variable multicolinéaire trouvée dans df_final_ml")

            st.markdown("---")

            # ÉTAPE 2: État Initial des données après suppression multicolinéarité
            st.subheader(" Étape 2 : État des Données après Nettoyage")
            
            st.markdown("#### 📋 Récapitulatif des données nettoyées")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Nombre de lignes", f"{df_final_ml.shape[0]:,}")
            
            with col2:
                st.metric("📋 Nombre de colonnes", f"{df_final_ml.shape[1]:,}")
            
            with col3:
                st.metric("🎯 Variable cible", "co2")
            
            colonnes_numeriques = df_final_ml.select_dtypes(include=np.number).columns.tolist()
            colonnes_categorielles = df_final_ml.select_dtypes(include='object').columns.tolist()
            
            toutes_colonnes = df_final_ml.columns.tolist()
            types_colonnes = []
            for col in toutes_colonnes:
                if col in colonnes_numeriques:
                    types_colonnes.append('Numérique')
                elif col in colonnes_categorielles:
                    types_colonnes.append('Catégorielle')
                else:
                    types_colonnes.append('Autre')
            
            df_recap = pd.DataFrame({
                'Colonne': toutes_colonnes,
                'Type': types_colonnes,
                'Valeurs Uniques': [df_final_ml[col].nunique() for col in toutes_colonnes],
                'Valeurs Manquantes': [df_final_ml[col].isnull().sum() for col in toutes_colonnes],
                '% Manquantes': [round(df_final_ml[col].isnull().sum() / len(df_final_ml) * 100, 2) for col in toutes_colonnes]
            })
            
            safe_dataframe_display(df_recap, "État final des données pour ML")

            st.markdown("---")
            
            # ÉTAPE 3: Séparation du jeu d'entraînement et de test
            st.subheader(" Étape 3 : Séparation Train/Test AVANT Encodage")
            
            if 'co2' not in df_final_ml.columns:
                st.error("Colonne 'co2' non trouvée dans df_final_ml. Impossible de procéder au Feature Engineering.")
            else:
                X = df_final_ml.drop("co2", axis=1)
                y = df_final_ml["co2"]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🏋️ X_train", f"{X_train.shape[0]:,} × {X_train.shape[1]:,}")
                
                with col2:
                    st.metric("🧪 X_test", f"{X_test.shape[0]:,} × {X_test.shape[1]:,}")
                
                with col3:
                    st.metric("🎯 y_train", f"{y_train.shape[0]:,}")
                
                with col4:
                    st.metric("🎯 y_test", f"{y_test.shape[0]:,}")
                
                st.success("✅ Division train/test effectuée (80/20) avec random_state=42")

            st.markdown("---")
            
            # ÉTAPE 4: Encodage des variables
            st.subheader(" Étape 4 : Encodage des Variables Catégorielles")
            
            X_train = X_train.copy()
            X_test = X_test.copy()
            
            st.markdown("#### 🔄 OneHotEncoder pour Carrosserie et gamme")
            
            ohe_cols = ["Carrosserie", "gamme"]
            ohe_cols = [col for col in ohe_cols if col in X_train.columns]
            
            if ohe_cols:
                ohe = OneHotEncoder(drop="first", sparse_output=False)
                
                X_train_ohe = ohe.fit_transform(X_train[ohe_cols])
                X_test_ohe = ohe.transform(X_test[ohe_cols])
                
                X_train_ohe = pd.DataFrame(X_train_ohe, columns=ohe.get_feature_names_out(ohe_cols))
                X_test_ohe = pd.DataFrame(X_test_ohe, columns=ohe.get_feature_names_out(ohe_cols))
                
                X_train = X_train.drop(columns=ohe_cols).reset_index(drop=True)
                X_test = X_test.drop(columns=ohe_cols).reset_index(drop=True)
                X_train = pd.concat([X_train, X_train_ohe], axis=1)
                X_test = pd.concat([X_test, X_test_ohe], axis=1)
                
                st.success(f"✅ OneHotEncoder appliqué à : {', '.join(ohe_cols)}")
                
            else:
                st.warning("⚠️ Aucune colonne trouvée pour OneHotEncoder (Carrosserie, gamme)")

            st.markdown("#### ⛽ Encodage spécial pour typ_crb")
            
            if 'typ_crb' in X_train.columns:
                def regrouper_carburant(carb):
                    if carb == "GO":
                        return "GO"
                    elif carb == "ES":
                        return "ES"
                    else:
                        return "Autres"

                X_train["typ_crb_grp"] = X_train["typ_crb"].apply(regrouper_carburant)
                X_test["typ_crb_grp"] = X_test["typ_crb"].apply(regrouper_carburant)

                distrib_carb = X_train["typ_crb_grp"].value_counts()
                
                ohe_typ_crb = OneHotEncoder(drop=None, sparse_output=False)

                X_train_crb = ohe_typ_crb.fit_transform(X_train[["typ_crb_grp"]])
                X_test_crb = ohe_typ_crb.transform(X_test[["typ_crb_grp"]])

                X_train_crb = pd.DataFrame(X_train_crb, columns=ohe_typ_crb.get_feature_names_out(["typ_crb_grp"]))
                X_test_crb = pd.DataFrame(X_test_crb, columns=ohe_typ_crb.get_feature_names_out(["typ_crb_grp"]))

                X_train = X_train.drop(columns=["typ_crb", "typ_crb_grp"]).reset_index(drop=True)
                X_test = X_test.drop(columns=["typ_crb", "typ_crb_grp"]).reset_index(drop=True)

                X_train = pd.concat([X_train, X_train_crb], axis=1)
                X_test = pd.concat([X_test, X_test_crb], axis=1)
                
                st.success("✅ typ_crb regroupé (GO/ES/Autres) et encodé avec succès")
                
            else:
                st.warning("⚠️ Colonne 'typ_crb' non trouvée pour l'encodage")

            st.markdown("#### 📊 Encodage fréquentiel")
            
            freq_cols = ["lib_mrq_utac", "typ_boite_nb_rapp"]
            freq_cols = [col for col in freq_cols if col in X_train.columns]
            
            if freq_cols:
                freq_maps = {col: X_train[col].value_counts(normalize=True) for col in freq_cols}

                for col in freq_cols:
                    before_unique = X_train[col].nunique()
                    X_train[col] = X_train[col].map(freq_maps[col])
                    X_test[col] = X_test[col].map(freq_maps[col]).fillna(0)
                    st.info(f"📋 {col}: {before_unique} modalités → fréquences [0-1]")
                
                st.success(f"✅ Encodage fréquentiel appliqué à : {', '.join(freq_cols)}")
            else:
                st.warning("⚠️ Aucune colonne trouvée pour l'encodage fréquentiel")

            st.markdown("---")
            
            # ÉTAPE 5: Standardisation
            st.subheader(" Étape 5 : Standardisation des Variables Numériques")
            
            one_hot_generated_cols = []
            if 'ohe' in locals():
                one_hot_generated_cols.extend(ohe.get_feature_names_out(ohe_cols).tolist())
            if 'ohe_typ_crb' in locals():
                one_hot_generated_cols.extend(ohe_typ_crb.get_feature_names_out(["typ_crb_grp"]).tolist())

            cols_to_exclude = one_hot_generated_cols + freq_cols
            numerical_cols = [col for col in X_train.select_dtypes(include=["int64", "float64"]).columns if col not in cols_to_exclude]

            if numerical_cols:
                st.info(f"📊 Application du StandardScaler : {', '.join(numerical_cols)}")
                st.info(f"🚫 Variables exclues : {', '.join(cols_to_exclude)}")
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train[numerical_cols])
                X_test_scaled = scaler.transform(X_test[numerical_cols])
                
                for i, col in enumerate(numerical_cols):
                    X_train[col] = X_train_scaled[:, i].astype('float64')
                    X_test[col] = X_test_scaled[:, i].astype('float64')
                
                st.success(f"✅ Standardisation appliquée à {len(numerical_cols)} variables numériques")

            st.markdown("---")
            
            # ÉTAPE 6: Résumé final du df_final_ml propre
            st.subheader("📊 **DataFrame final prêt pour la modélisation**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 X_train final", f"{X_train.shape[0]:,} × {X_train.shape[1]:,}")
            
            with col2:
                st.metric("📊 X_test final", f"{X_test.shape[0]:,} × {X_test.shape[1]:,}")
            
            with col3:
                st.metric("🎯 Variables finales", X_train.shape[1])
            
            with col4:
                memory_usage = (X_train.memory_usage(deep=True).sum() + X_test.memory_usage(deep=True).sum()) / 1024**2
                st.metric("💾 Taille totale", f"{memory_usage:.1f} MB")
            
            
            colonnes_finales = pd.DataFrame({
                'Variable': X_train.columns.tolist(),
                'Type': [str(X_train[col].dtype) for col in X_train.columns],
                'Valeurs Uniques': [X_train[col].nunique() for col in X_train.columns],
                'Min': [round(X_train[col].min(), 4) if X_train[col].dtype in ['int64', 'float64'] else 'N/A' for col in X_train.columns],
                'Max': [round(X_train[col].max(), 4) if X_train[col].dtype in ['int64', 'float64'] else 'N/A' for col in X_train.columns]
            })
            
            safe_dataframe_display(colonnes_finales, "Variables finales")
            
            st.session_state['X_train_fe'] = X_train
            st.session_state['X_test_fe'] = X_test
            st.session_state['y_train_fe'] = y_train
            st.session_state['y_test_fe'] = y_test
            st.session_state['feature_columns_fe'] = X_train.columns.tolist()
            st.session_state['df_final_ml_clean'] = df_final_ml
            
            st.success("🎉 **Feature Engineering terminé !**")

            st.markdown("---")

            # MODIFICATION: Nouvelle section récapitulative
                        
            st.markdown("""
            ### 🎯 Points Clés du Feature Engineering Réalisé
            
            **Nettoyage et Préparation :**
            - Suppression des variables multicolinéaires identifiées lors de l'analyse de corrélation
            - Division train/test (80/20) avant tout encodage pour éviter le data leakage
            - Préservation de l'intégrité des données de test
            
            **Encodage Intelligent :**
            - **OneHotEncoder** : Variables catégorielles à faible cardinalité (Carrosserie, gamme)
            - **Regroupement + OneHot** : typ_crb regroupé en 3 catégories principales (GO/ES/Autres)
            - **Encodage fréquentiel** : Variables à haute cardinalité (marques, boîtes de vitesse)
            
            **Standardisation :**
            - StandardScaler appliqué uniquement aux variables numériques continues
            - Exclusion des variables encodées (binaires) de la standardisation
                        
            """)

        st.markdown("---")
        st.info("""
        **🚀 Prochaines étapes :**
        
        Naviguez vers l'onglet **Modélisation ML** pour :
        - Charger des modèles pré-entraînés
        - Comparer les performances
        - Évaluer différents algorithmes de ML
        - Sélectionner le meilleur modèle
        """)

    
    # ----------------------------------------------------------------
    # Onglet Modélisation ML (index 4) - MODIFIÉ
    # ----------------------------------------------------------------
    with tabs[4]:
        st.markdown("Récupération du dataframe post Feature Engineering et sélection des modèles adaptés")
        

        if 'X_train_fe' in st.session_state and st.session_state['X_train_fe'] is not None:
            X_train = st.session_state['X_train_fe']
            X_test = st.session_state['X_test_fe']
            y_train = st.session_state['y_train_fe']
            y_test = st.session_state['y_test_fe']
            feature_columns = st.session_state['feature_columns_fe']
            
            st.subheader("📊 Résumé des Données pour la Modélisation")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🏋️ X_train", f"{X_train.shape[0]:,} × {X_train.shape[1]:,}")
            
            with col2:
                st.metric("🧪 X_test", f"{X_test.shape[0]:,} × {X_test.shape[1]:,}")
            
            with col3:
                st.metric("🎯 y_train", f"{y_train.shape[0]:,}")
            
            with col4:
                st.metric("🎯 y_test", f"{y_test.shape[0]:,}")
            
            st.success("✅ Données prêtes pour la modélisation - Issues du Feature Engineering avec suppression de la multicolinéarité")
            
        else:
            st.info("Veuillez d'abord exécuter l'onglet 'Feature Engineering' pour préparer les données.")
            return

        st.markdown("---")
        
        # SECTION 1: Modèles Conventionnels - MODIFIÉ
        st.subheader('Partie 1 : Modèles Conventionnels')
        
        run_conventional = st.button('🚀 Charger les Modèles Conventionnels', key='run_conventional_button')
        
        if run_conventional or 'conventional_models_loaded' in st.session_state:
            if run_conventional:
                st.markdown('#### Chargement en cours...')
                
                with st.spinner('Chargement des modèles...'):
                    # Utilisation de la version mise en cache
                    loaded_models, loaded_scalers = load_models_and_scalers_cached('saved_models_regression')
                
                if not loaded_models:
                    st.error("❌ Aucun modèle de ML trouvé dans le dossier 'saved_models_regression'")
                    return
                
                conventional_models = {name: model for name, model in loaded_models.items() 
                                    if 'GridSearch' not in name}
                
                if not conventional_models:
                    st.warning("⚠️ Aucun modèle conventionnel trouvé (tous semblent être des modèles GridSearch)")
                    conventional_models = loaded_models
                
                st.markdown("#### 📋 Modèles conventionnels détectés :")
                for i, model_name in enumerate(conventional_models.keys(), 1):
                    st.write(f"{i}. **{model_name}**")
                
                st.success(f'✅ {len(conventional_models)} modèles conventionnels chargés avec succès !')
                
                # Évaluation parallèle pour plus de rapidité
                with st.spinner('Évaluation des modèles en cours...'):
                    results = evaluate_models_parallel(conventional_models, X_test, y_test)
                
                st.success('✅ Évaluation terminée !')
                
                st.session_state['conventional_results'] = results
                st.session_state['loaded_models'] = conventional_models
                st.session_state['conventional_models_loaded'] = True
            
            else:
                results = st.session_state.get('conventional_results', {})
                conventional_models = st.session_state.get('loaded_models', {})
            
            if results:
                st.markdown('#### Résultats des Modèles Conventionnels')
                
                results_data = []
                for name, metrics in results.items():
                    results_data.append({
                        'Modèle': name,
                        'MAE': round(metrics['MAE'], 4) if not np.isnan(metrics['MAE']) else 'Erreur',
                        'R²': round(metrics['R2'], 4) if not np.isnan(metrics['R2']) else 'Erreur'
                    })
                
                df_results_conv = pd.DataFrame(results_data)
                
                # Tri par R² décroissant
                df_valid = df_results_conv[df_results_conv['R²'] != 'Erreur'].copy()
                df_error = df_results_conv[df_results_conv['R²'] == 'Erreur'].copy()
                
                if not df_valid.empty:
                    df_valid['R²'] = df_valid['R²'].astype(float)
                    df_valid = df_valid.sort_values('R²', ascending=False)
                    df_valid['R²'] = df_valid['R²'].round(4)
                    df_final_conv = pd.concat([df_valid, df_error], ignore_index=True)
                    
                    # Identifier le meilleur modèle
                    best_model_conv = df_valid.iloc[0]['Modèle'] if not df_valid.empty else None
                else:
                    df_final_conv = df_results_conv
                    best_model_conv = None
                
                # Nouveau tableau Plotly amélioré
                fig_table_conv = create_enhanced_plotly_table(
                    df_final_conv, 
                    "Performances des Modèles Conventionnels",
                    best_model_conv
                )
                
                st.plotly_chart(fig_table_conv, use_container_width=True)
                
                # Graphique de comparaison - NOUVEAU
                if not df_valid.empty:
                    valid_results = {k: v for k, v in results.items() if not np.isnan(v['R2'])}
                    
                    if valid_results:
                        models_names = list(valid_results.keys())
                        r2_scores = [valid_results[name]['R2'] for name in models_names]
                        mae_scores = [valid_results[name]['MAE'] for name in models_names]
                        
                        fig_comp = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        # Barres R²
                        fig_comp.add_trace(
                            go.Bar(
                                name='R² Score',
                                x=models_names,
                                y=r2_scores,
                                marker_color='lightblue',
                                yaxis='y'
                            ),
                            secondary_y=False
                        )
                        
                        # Ligne MAE
                        fig_comp.add_trace(
                            go.Scatter(
                                name='MAE',
                                x=models_names,
                                y=mae_scores,
                                mode='lines+markers',
                                marker_color='red',
                                line=dict(width=3),
                                yaxis='y2'
                            ),
                            secondary_y=True
                        )
                        
                        fig_comp.update_layout(
                            title='Comparaison des Performances - Modèles Conventionnels',
                            xaxis=dict(title='Modèles'),
                            height=500
                        )
                        
                        fig_comp.update_yaxes(title_text="R² Score", secondary_y=False)
                        fig_comp.update_yaxes(title_text="MAE", secondary_y=True)
                        
                        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("---")

        # SECTION 2: Modèles GridSearchCV - MODIFIÉ
        st.subheader('Partie 2 : Modèles GridSearchCV avec meilleurs hyperparamètres')

        @st.cache_data
        def load_hyperparameters_from_csv():
            csv_path = os.path.join('saved_models_regression', 'best_hyperparameters_with_metrics.csv')
            if os.path.exists(csv_path):
                try:
                    df_hyperparams = pd.read_csv(csv_path)
                    return df_hyperparams
                except Exception as e:
                    st.warning(f"Erreur lors du chargement du fichier CSV : {e}")
                    return None
            else:
                st.warning(f"Fichier {csv_path} non trouvé")
                return None

        def parse_hyperparameters_from_csv(df_hyperparams, model_name):
            if df_hyperparams is None:
                return {}
            
            clean_model_name = model_name.replace("(GridSearch)", "").replace("(Grid)", "").strip()
            
            model_rows = df_hyperparams[df_hyperparams['Model'].str.contains(clean_model_name, case=False, na=False)]
            
            if model_rows.empty:
                return {}
            
            hyperparams = {}
            
            for _, row in model_rows.iterrows():
                parameter = row['Parameter']
                best_value = row['Best_Value']
                
                if parameter == 'No hyperparameters':
                    continue
                    
                if pd.isna(best_value) or best_value == '':
                    best_value = 'None (défaut)'
                
                hyperparams[parameter] = best_value
            
            return hyperparams
        
        run_gridsearch = st.button('🚀 Charger les Modèles GridSearchCV', key='run_gridsearch_button')
        
        if run_gridsearch or 'gridsearch_models_loaded' in st.session_state:
            if run_gridsearch:
                df_hyperparams = load_hyperparameters_from_csv()
                
                with st.spinner('Chargement des modèles GridSearchCV...'):
                    # Utilisation de la version mise en cache
                    all_models, _ = load_models_and_scalers_cached('saved_models_regression')
                
                gridsearch_models = {name: model for name, model in all_models.items() 
                                if 'GridSearch' in name}
                
                if not gridsearch_models:
                    st.warning('⚠️ Aucun modèle GridSearchCV trouvé dans les noms de fichiers')
                    st.info('🔄 Recherche de modèles avec "gridsearch" dans le nom...')
                    
                    gridsearch_models = {name: model for name, model in all_models.items() 
                                    if 'gridsearch' in name.lower()}
                
                if gridsearch_models:
                    for i, (model_name, model) in enumerate(gridsearch_models.items(), 1):
                        st.markdown(f"**{i}. {model_name}**")
                        
                        # Cas spécial pour Linear Regression (GridSearch)
                        if "Linear Regression" in model_name and "GridSearch" in model_name:
                            st.markdown("   ⚠️ **Hyperparamètres non trouvés**")
                        else:
                            hyperparams_csv = parse_hyperparameters_from_csv(df_hyperparams, model_name)
                            
                            if hyperparams_csv:
                                st.markdown("   📊 **Hyperparamètres optimisés :**")
                                for param, value in hyperparams_csv.items():
                                    st.markdown(f"   - **{param}** : {value}")
                            else:
                                if hasattr(model, 'best_params_') and model.best_params_:
                                    st.markdown("   📊 **Hyperparamètres optimisés (depuis modèle) :**")
                                    for param, value in model.best_params_.items():
                                        st.markdown(f"   - **{param}** : {value}")
                                else:
                                    clean_name = model_name.replace("(GridSearch)", "").replace("(Grid)", "").strip()
                                    st.markdown(f"   ⚠️ Hyperparamètres non trouvés (recherché: '{clean_name}')")
                                    
                                    if df_hyperparams is not None:
                                        available_models = df_hyperparams['Model'].unique()
                                        st.markdown(f"   🔍 Modèles disponibles dans CSV: {list(available_models)}")
                        
                        st.markdown("")
                    
                    st.success(f'✅ {len(gridsearch_models)} modèles GridSearchCV chargés avec succès !')
                    
                    # Évaluation parallèle
                    with st.spinner('Évaluation des modèles GridSearchCV...'):
                        gs_results_raw = evaluate_models_parallel(gridsearch_models, X_test, y_test)
                    
                    # Enrichir avec les hyperparamètres
                    gs_results = {}
                    for name, metrics in gs_results_raw.items():
                        hyperparams_csv = parse_hyperparameters_from_csv(df_hyperparams, name)
                        best_params = hyperparams_csv if hyperparams_csv else getattr(gridsearch_models[name], 'best_params_', {})
                        
                        gs_results[name] = {
                            'MAE': metrics['MAE'],
                            'R2': metrics['R2'],
                            'best_params': best_params,
                            'cv_score': getattr(gridsearch_models[name], 'best_score_', 'N/A')
                        }
                    
                    st.success('✅ Évaluation GridSearchCV terminée !')
                    
                    st.session_state['grid_search_results'] = gs_results
                    st.session_state['gridsearch_models'] = gridsearch_models
                    st.session_state['gridsearch_models_loaded'] = True
                    st.session_state['df_hyperparams'] = df_hyperparams
                
                else:
                    st.info('ℹ️ Aucun modèle GridSearchCV spécifique trouvé.')
            
            else:
                gs_results = st.session_state.get('grid_search_results', {})
                gridsearch_models = st.session_state.get('gridsearch_models', {})
                df_hyperparams = st.session_state.get('df_hyperparams', None)
            
            if gs_results:
                st.markdown('#### Résultats des Modèles GridSearchCV')
                
                gs_results_data = []
                for name, metrics in gs_results.items():
                    gs_results_data.append({
                        'Modèle': name,
                        'MAE': round(metrics['MAE'], 4) if not np.isnan(metrics['MAE']) else 'Erreur',
                        'R²': round(metrics['R2'], 4) if not np.isnan(metrics['R2']) else 'Erreur'
                    })
                
                df_results_gs = pd.DataFrame(gs_results_data)
                
                # Tri par R² décroissant
                df_valid_gs = df_results_gs[df_results_gs['R²'] != 'Erreur'].copy()
                df_error_gs = df_results_gs[df_results_gs['R²'] == 'Erreur'].copy()
                
                if not df_valid_gs.empty:
                    df_valid_gs['R²'] = df_valid_gs['R²'].astype(float)
                    df_valid_gs = df_valid_gs.sort_values('R²', ascending=False)
                    df_valid_gs['R²'] = df_valid_gs['R²'].round(4)
                    df_final_gs = pd.concat([df_valid_gs, df_error_gs], ignore_index=True)
                    
                    # Identifier le meilleur modèle
                    best_model_gs = df_valid_gs.iloc[0]['Modèle'] if not df_valid_gs.empty else None
                else:
                    df_final_gs = df_results_gs
                    best_model_gs = None
                
                # Nouveau tableau Plotly amélioré
                fig_table_gs = create_enhanced_plotly_table(
                    df_final_gs, 
                    "Performances des Modèles GridSearchCV",
                    best_model_gs
                )
                
                st.plotly_chart(fig_table_gs, use_container_width=True)
                
                # NOUVEAU: Graphique de comparaison pour GridSearchCV
                if not df_valid_gs.empty:
                    valid_results_gs = {k: v for k, v in gs_results.items() if not np.isnan(v['R2'])}
                    
                    if valid_results_gs:
                        models_names_gs = list(valid_results_gs.keys())
                        r2_scores_gs = [valid_results_gs[name]['R2'] for name in models_names_gs]
                        mae_scores_gs = [valid_results_gs[name]['MAE'] for name in models_names_gs]
                        
                        fig_comp_gs = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        # Barres R²
                        fig_comp_gs.add_trace(
                            go.Bar(
                                name='R² Score',
                                x=models_names_gs,
                                y=r2_scores_gs,
                                marker_color='lightgreen',
                                yaxis='y'
                            ),
                            secondary_y=False
                        )
                        
                        # Ligne MAE
                        fig_comp_gs.add_trace(
                            go.Scatter(
                                name='MAE',
                                x=models_names_gs,
                                y=mae_scores_gs,
                                mode='lines+markers',
                                marker_color='orange',
                                line=dict(width=3),
                                yaxis='y2'
                            ),
                            secondary_y=True
                        )
                        
                        fig_comp_gs.update_layout(
                            title='Comparaison des Performances - Modèles GridSearchCV',
                            xaxis=dict(title='Modèles'),
                            height=500
                        )
                        
                        fig_comp_gs.update_yaxes(title_text="R² Score", secondary_y=False)
                        fig_comp_gs.update_yaxes(title_text="MAE", secondary_y=True)
                        
                        st.plotly_chart(fig_comp_gs, use_container_width=True)

            st.markdown("---")

        # SECTION 3: Comparaison Globale - MODIFIÉ
        if ('conventional_models_loaded' in st.session_state and 'gridsearch_models_loaded' in st.session_state):
            st.subheader("Partie 3 : Comparaison Globale des Modèles")
                                    
            # Afficher les tableaux de performance avant la sélection du meilleur modèle
            col1, col2 = st.columns(2)
            
            # Tableau des modèles conventionnels
            with col1:
                if 'conventional_results' in st.session_state:
                    results = st.session_state['conventional_results']
                    results_data = []
                    for name, metrics in results.items():
                        results_data.append({
                            'Modèle': name,
                            'MAE': round(metrics['MAE'], 4) if not np.isnan(metrics['MAE']) else 'Erreur',
                            'R²': round(metrics['R2'], 4) if not np.isnan(metrics['R2']) else 'Erreur'
                        })
                    
                    df_results_conv = pd.DataFrame(results_data)
                    
                    df_valid = df_results_conv[df_results_conv['R²'] != 'Erreur'].copy()
                    df_error = df_results_conv[df_results_conv['R²'] == 'Erreur'].copy()
                    
                    if not df_valid.empty:
                        df_valid['R²'] = df_valid['R²'].astype(float)
                        df_valid = df_valid.sort_values('R²', ascending=False)
                        df_valid['R²'] = df_valid['R²'].round(4)
                        df_final_conv = pd.concat([df_valid, df_error], ignore_index=True)
                        
                        best_model_conv_comp = df_valid.iloc[0]['Modèle'] if not df_valid.empty else None
                    else:
                        df_final_conv = df_results_conv
                        best_model_conv_comp = None
                    
                    fig_table_conv_comp = create_enhanced_plotly_table(
                        df_final_conv, 
                        "Modèles Conventionnels",
                        best_model_conv_comp
                    )
                    fig_table_conv_comp.update_layout(height=250)
                    
                    st.plotly_chart(fig_table_conv_comp, use_container_width=True)
                else:
                    st.info("Aucun modèle conventionnel chargé")
            
            # Tableau des modèles GridSearchCV
            with col2:
                if 'grid_search_results' in st.session_state:
                    gs_results = st.session_state['grid_search_results']
                    gs_results_data = []
                    for name, metrics in gs_results.items():
                        gs_results_data.append({
                            'Modèle': name,
                            'MAE': round(metrics['MAE'], 4) if not np.isnan(metrics['MAE']) else 'Erreur',
                            'R²': round(metrics['R2'], 4) if not np.isnan(metrics['R2']) else 'Erreur'
                        })
                    
                    df_results_gs = pd.DataFrame(gs_results_data)
                    
                    df_valid_gs = df_results_gs[df_results_gs['R²'] != 'Erreur'].copy()
                    df_error_gs = df_results_gs[df_results_gs['R²'] == 'Erreur'].copy()
                    
                    if not df_valid_gs.empty:
                        df_valid_gs['R²'] = df_valid_gs['R²'].astype(float)
                        df_valid_gs = df_valid_gs.sort_values('R²', ascending=False)
                        df_valid_gs['R²'] = df_valid_gs['R²'].round(4)
                        df_final_gs_comp = pd.concat([df_valid_gs, df_error_gs], ignore_index=True)
                        
                        best_model_gs_comp = df_valid_gs.iloc[0]['Modèle'] if not df_valid_gs.empty else None
                    else:
                        df_final_gs_comp = df_results_gs
                        best_model_gs_comp = None
                    
                    fig_table_gs_comp = create_enhanced_plotly_table(
                        df_final_gs_comp, 
                        "Modèles GridSearchCV",
                        best_model_gs_comp
                    )
                    fig_table_gs_comp.update_layout(height=250)
                    
                    st.plotly_chart(fig_table_gs_comp, use_container_width=True)
                else:
                    st.info("Aucun modèle GridSearchCV chargé")
            
            st.markdown("---")
            
            # Sélection du meilleur modèle
            st.markdown("#### 🏆 Sélection du meilleur modèle")
            
            best_r2 = -np.inf
            best_model_info = None
            
            if 'conventional_results' in st.session_state:
                for model_name, metrics in st.session_state['conventional_results'].items():
                    if not np.isnan(metrics['R2']):
                        if metrics['R2'] > best_r2:
                            best_r2 = metrics['R2']
                            best_model_info = {
                                'name': model_name,
                                'type': 'Conventionnel',
                                'metrics': metrics
                            }
            
            if 'grid_search_results' in st.session_state:
                for model_name, metrics in st.session_state['grid_search_results'].items():
                    if not np.isnan(metrics['R2']):
                        if metrics['R2'] > best_r2:
                            best_r2 = metrics['R2']
                            best_model_info = {
                                'name': model_name,
                                'type': 'GridSearchCV',
                                'metrics': metrics
                            }
            
            if best_model_info:
                st.session_state['best_model_info'] = best_model_info
                
                st.success(f"🏆 Le meilleur modèle est **{best_model_info['name']} ({best_model_info['type']})** avec un R² de **{best_model_info['metrics']['R2']:.4f}**")
            
            else:
                st.info("Aucun modèle n'a encore été chargé. Veuillez exécuter au moins une section ci-dessus.")

        else:
            st.info("Chargez d'abord des modèles dans les sections précédentes pour voir la comparaison globale.")

        st.markdown("---")
        st.info("""
        **🚀 Prochaines étapes :**

        Naviguez vers l'onglet **Feature Importance** pour :
        - Analyser l'importance des variables
        - Interpréter les coefficients des modèles
        - Comprendre les facteurs d'influence avec SHAP
        """)


    # ----------------------------------------------------------------
    # Onglet Feature Importance (index 5) - MODIFIÉ
    # ----------------------------------------------------------------
    with tabs[5]:
        st.markdown("Analyse et interprétation de l'importance des caractéristiques des modèles entraînés.")
        
        if 'X_train_fe' not in st.session_state:
            st.info("Veuillez d'abord exécuter l'onglet 'Feature Engineering' pour préparer les données.")
            return

        X_train = st.session_state['X_train_fe']
        X_test = st.session_state['X_test_fe']
        y_train = st.session_state['y_train_fe']
        y_test = st.session_state['y_test_fe']
        feature_names = st.session_state['feature_columns_fe']

        # SECTION 1: Comparaison Feature Importance vs SHAP
        st.subheader("Etape 1 : Comparaison des Méthodes d'Interprétation")
        
        comparison_table = go.Figure(data=[go.Table(
            header=dict(
                values=["<b>Aspect</b>", "<b>Importance des Caractéristiques</b>", "<b>Valeurs SHAP</b>"],
                fill_color='#4a7dc4',
                font=dict(color='white', size=12),
                align='left',
                height=40
            ),
            cells=dict(
                values=[
                    ["Portée", "Direction", "Types de Modèles", "Calcul", "Interprétation"],
                    ["Globale (modèle entier)", "Magnitude uniquement", "Principalement basée sur les arbres", "Rapide", "\"Importance globale\""],
                    ["Globale + Locale (par prédiction)", "Magnitude + Direction", "Fonctionne pour tous les modèles", 
                    "Plus lent (surtout pour les grands ensembles de données)", "\"Comment/pourquoi cette prédiction ?\""]
                ],
                fill_color=['#e6e6e6', '#f9f9f9', '#ffffff'],
                font=dict(size=12),
                align='left',
                height=30
            )
        )])

        comparison_table.update_layout(
            title="<b>Comparison: Feature Importance vs SHAP Values</b>",
            title_x=0.5,
            margin=dict(l=20, r=20, t=60, b=20),
            width=900,
            height=350
        )

        st.plotly_chart(comparison_table, use_container_width=True)

        
        # SECTION 2: Feature Importance des Modèles
        st.subheader("Etape 2 : Feature Importance des Modèles")
        
        has_conventional = 'loaded_models' in st.session_state
        has_gridsearch = 'gridsearch_models' in st.session_state
        
        if not has_conventional and not has_gridsearch:
            st.warning("⚠️ Aucun modèle chargé trouvé. Veuillez d'abord charger des modèles dans l'onglet 'Modélisation ML'.")
            return
        
        def extract_importance(model, model_name):
            if hasattr(model, 'feature_importances_'):
                return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
            elif hasattr(model, 'coef_'):
                return pd.Series(np.abs(model.coef_), index=feature_names).sort_values(ascending=False)
            else:
                return None
        
        importances_conv = {}
        if has_conventional:
            for name, model in st.session_state['loaded_models'].items():
                importance = extract_importance(model, name)
                if importance is not None:
                    importances_conv[f"{name} (Conv)"] = importance
        
        importances_grid = {}
        if has_gridsearch:
            for name, model in st.session_state['gridsearch_models'].items():
                importance = extract_importance(model, name)
                if importance is not None:
                    importances_grid[f"{name} (Grid)"] = importance
        
        if importances_conv or importances_grid:
            
            if importances_conv:
                st.markdown("##### 🔧 Modèles Conventionnels")
                
                n_models_conv = len([k for k in importances_conv.keys() if 'Linear' not in k])
                if n_models_conv > 0:
                    fig_conv = make_subplots(
                        rows=1, cols=min(3, n_models_conv),
                        subplot_titles=[k.replace(' (Conv)', '') for k in importances_conv.keys() if 'Linear' not in k]
                    )
                    
                    col_idx = 1
                    colors = ['royalblue', 'seagreen', 'mediumpurple']
                    
                    for i, (name, importance) in enumerate(importances_conv.items()):
                        if 'Linear' not in name and col_idx <= 3:
                            top_features = importance.head(10)
                            fig_conv.add_trace(
                                go.Bar(
                                    x=top_features.values,
                                    y=top_features.index,
                                    orientation='h',
                                    marker_color=colors[i % len(colors)],
                                    name=name.replace(' (Conv)', ''),
                                    showlegend=False
                                ),
                                row=1, col=col_idx
                            )
                            col_idx += 1
                    
                    fig_conv.update_layout(
                        title_text='Feature Importance - Modèles Conventionnels',
                        height=500,
                        showlegend=False
                    )
                    
                    for i in range(1, min(4, n_models_conv + 1)):
                        fig_conv.update_xaxes(title_text="Importance", row=1, col=i)
                        fig_conv.update_yaxes(title_text="Features", row=1, col=i)
                    
                    st.plotly_chart(fig_conv, use_container_width=True)
                
                linear_importance = None
                for name, importance in importances_conv.items():
                    if 'linear' in name.lower():
                        linear_importance = importance
                        break
                
                if linear_importance is not None:
                    fig_linear = go.Figure(go.Bar(
                        x=linear_importance.head(10).values,
                        y=linear_importance.head(10).index,
                        orientation='h',
                        marker_color='darkorange',
                        name='Linear Regression'
                    ))
                    
                    fig_linear.update_layout(
                        title='Coefficients - Linear Regression (Conventionnel)',
                        xaxis_title="Importance (|Coefficient|)",
                        yaxis_title="Features",
                        height=400
                    )
                    
                    st.plotly_chart(fig_linear, use_container_width=True)
            
            if importances_grid:
                st.markdown("##### 🎯 Modèles GridSearchCV")
                
                fig_grid = make_subplots(
                    rows=1, cols=min(3, len(importances_grid)),
                    subplot_titles=[k.replace(' (Grid)', '') for k in importances_grid.keys()]
                )
                
                colors = ['royalblue', 'tomato', 'seagreen']
                
                for i, (name, importance) in enumerate(importances_grid.items()):
                    if i < 3:
                        top_features = importance.head(10)
                        fig_grid.add_trace(
                            go.Bar(
                                x=top_features.values,
                                y=top_features.index,
                                orientation='h',
                                marker_color=colors[i % len(colors)],
                                name=name.replace(' (Grid)', ''),
                                showlegend=False
                            ),
                            row=1, col=i+1
                        )
                
                fig_grid.update_layout(
                    title_text='Feature Importance - Modèles GridSearchCV',
                    height=500,
                    showlegend=False
                )
                
                for i in range(1, min(4, len(importances_grid) + 1)):
                    fig_grid.update_xaxes(title_text="Importance", row=1, col=i)
                    fig_grid.update_yaxes(title_text="Features", row=1, col=i)
                
                st.plotly_chart(fig_grid, use_container_width=True)
            
            st.markdown("#### 📋 Tableau Comparatif des Importances")
            
            all_importances = {**importances_conv, **importances_grid}
            
            if all_importances:
                df_importances = pd.DataFrame(all_importances).fillna(0)
                
                n_rows = len(df_importances)
                n_cols = len(df_importances.columns)
                
                row_colors = ['white' if i % 2 == 0 else 'whitesmoke' for i in range(n_rows)]
                
                fill_colors = [row_colors]
                
                for col in df_importances.columns:
                    top3 = df_importances[col].nlargest(3).index.tolist()
                    col_colors = [
                        'lightgreen' if feat in top3 else base
                        for feat, base in zip(df_importances.index, row_colors)
                    ]
                    fill_colors.append(col_colors)
                
                fig_table = go.Figure(data=[
                    go.Table(
                        columnwidth=[200] + [100] * n_cols,
                        header=dict(
                            values=["<b>Feature</b>"] + [f"<b>{c}</b>" for c in df_importances.columns],
                            fill_color='darkslategray',
                            font=dict(color='white', size=14),
                            align='left'
                        ),
                        cells=dict(
                            values=[df_importances.index.tolist()] +
                                [df_importances[c].round(3).tolist() for c in df_importances.columns],
                            fill_color=fill_colors,
                            align='left'
                        )
                    )
                ])
                
                fig_table.update_layout(
                    title="Importances par modèle (Top 3 en vert)",
                    width=1200,
                    height=120 + 25 * n_rows,
                    margin=dict(l=20, r=20, t=60, b=20)
                )
                
                st.plotly_chart(fig_table, use_container_width=True)

        st.markdown("---")
        
        # SECTION 3: Analyse SHAP - MODIFIÉ
        st.subheader("Etape 3 : Analyse SHAP pour le meilleur modèle")
        
        best_model_info = st.session_state.get('best_model_info')
        
        if best_model_info:
            
            best_model = None
            if best_model_info['type'] == 'GridSearchCV' and 'gridsearch_models' in st.session_state:
                best_model = st.session_state['gridsearch_models'].get(best_model_info['name'])
            elif best_model_info['type'] == 'Conventionnel' and 'loaded_models' in st.session_state:
                best_model = st.session_state['loaded_models'].get(best_model_info['name'])
            
            if best_model is None:
                st.error("Impossible de récupérer le meilleur modèle")
                return
            
            # MODIFICATION: Suppression du slider et hardcoding de 500
            sample_size = 500
            st.info(f"Échantillon fixé à {sample_size} observations pour l'analyse SHAP")
            
            run_shap = st.button("🚀 Exécuter l'analyse SHAP", key='run_shap_button')
            
            if run_shap:
                try:
                                        
                    with st.spinner("Calcul des valeurs SHAP..."):
                        X_test_sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)
                        
                        explainer = shap.Explainer(
                            best_model,
                            X_train,
                            feature_perturbation="interventional"
                        )
                        
                        shap_values = explainer(X_test_sample, check_additivity=False)
                    
                    st.success("✅ Valeurs SHAP calculées avec succès !")
                    
                    # Visualisations SHAP
                    
                    # 1. Beeswarm plot
                    fig_beeswarm, ax_beeswarm = plt.subplots(figsize=(10, 8))
                    shap.plots.beeswarm(shap_values, show=False)
                    plt.title(f"SHAP Beeswarm Plot - {best_model_info['name']}")
                    plt.tight_layout()
                    st.pyplot(fig_beeswarm)
                    
                    # 2. Bar plot
                    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
                    shap.plots.bar(shap_values, show=False)
                    plt.title(f"SHAP Bar Plot - {best_model_info['name']}")
                    plt.tight_layout()
                    st.pyplot(fig_bar)
                    
                    # 3. Analyse quantitative
                                        
                    abs_shap = np.abs(shap_values.values).mean(axis=0)
                    
                    df_shap = pd.DataFrame({
                        'feature': X_test_sample.columns,
                        'mean_abs_shap': abs_shap
                    })
                    
                    df_shap['pct'] = df_shap['mean_abs_shap'] / df_shap['mean_abs_shap'].sum() * 100
                    df_shap = df_shap.sort_values('pct', ascending=False)
                    
                    st.markdown("**Top 5 des features par importance SHAP :**")
                    top5_display = df_shap.head(5).copy()
                    top5_display['pct_formatted'] = top5_display['pct'].apply(lambda x: f"{x:.1f}%")
                    
                    safe_dataframe_display(
                        top5_display[['feature', 'pct_formatted']].rename(columns={
                            'feature': 'Feature',
                            'pct_formatted': 'Contribution (%)'
                        }),
                        "Top 5 Features SHAP"
                    )
                    
                    top3 = df_shap.head(3)
                    
                    st.markdown("#### 🎯 Interprétation des Résultats")
                    
                    st.markdown(f"""
                    **Résultats clés de l'analyse SHAP :**
                    
                    1. **Caractéristique à l'impact le plus significatif sur les prédictions de CO₂ :**
                    - **'{top3.iloc[0].feature}'** (contribution : **{top3.iloc[0].pct:.1f}%**)
                    
                    
                    2. **Autres caractéristiques influentes :**
                    - **{top3.iloc[1].feature}** ({top3.iloc[1].pct:.1f}%)
                    - **{top3.iloc[2].feature}** ({top3.iloc[2].pct:.1f}%)
                    
                    
                    Les valeurs SHAP nous permettent de voir non seulement **quelles** variables sont importantes, 
                    mais aussi **comment** elles influencent chaque prédiction individuelle dans le contexte des émissions de CO₂ des véhicules.
                    """)
                    
                    # MODIFICATION: Suppression du graphique de contribution
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors du calcul SHAP : {e}")
                    st.info("💡 Conseil : Vérifiez la compatibilité du modèle avec SHAP")
        
        else:
            st.info("Aucun meilleur modèle identifié. Veuillez d'abord charger des modèles dans l'onglet 'Modélisation ML'.")
          

        st.markdown("---")

    # ----------------------------------------------------------------
    # MODIFICATION: Nouvel onglet Conclusions / Recommandations (index 6)
    # ----------------------------------------------------------------
    with tabs[6]:       
        conclusions_tabs = st.tabs(["📊 Conclusions", "🎯 Recommandations"])
        
        # Sous-onglet 1: Conclusions
        with conclusions_tabs[0]:
            st.markdown("""         
            Voici une synthèse de notre projet de régression concernant les facteurs favorisant les émissions de CO2 
            """)        
            # Conclusions sur les données
            st.markdown("#### Qualité et Traitement des Données")
            
            st.markdown("""
            **✅ Points forts identifiés :**
            - **Volume conséquent** : plusieurs années de données avec plus de 150 000 lignes d'observations
            - **Richesse des variables** : caractéristiques techniques, constructeurs, types de carburant
            - **Cohérence temporelle** : évolution des émissions observable sur la période d'étude
            - **Complétude après traitement** : 100% de données exploitables après imputation intelligente
            
            **⚠️ Défis relevés :**
            - **Hétérogénéité initiale** : différences de structure entre années nécessitant harmonisation
            - **Multicolinéarité** : variables redondantes (consommations, masses, puissances)
            - **Valeurs manquantes** : travail conséquent de gestion des NaN
            """)
            
            # Conclusions sur les modèles
            st.markdown("")
            st.markdown("#### Performance des Modèles de Machine Learning")
            
            if 'best_model_info' in st.session_state:
                best_info = st.session_state['best_model_info']
                st.markdown(f"""
                **🏆 Meilleur modèle identifié :**
                - **{best_info['name']} ({best_info['type']})**
                - **R² = {best_info['metrics']['R2']:.4f}** (explique {best_info['metrics']['R2']*100:.1f}% de la variance)
                - **MAE = {best_info['metrics']['MAE']:.2f} g/km** (erreur moyenne absolue)
                """)
            
                st.markdown(f"""
                **📈 Enseignements sur les algorithmes :**
                - **Modèles d'ensemble** (Random Forest, Gradient Boosting) : excellentes performances grâce à leur robustesses
                - **Régression linéaire** : performance honorable malgré la simplicité, bonne interprétabilité
                - **Gestion des non-linéarités** : les modèles basés sur les arbres capturent mieux les interactions complexes
                """)
            
            # Conclusions sur les variables
            st.markdown("")
            st.markdown("#### Variables les Plus Influentes")
                       
            
            st.markdown("""
            **🎯 Facteurs techniques dominants :**
            
            L'analyse d'importance des variables (Feature Importance + SHAP) révèle que les émissions de CO₂ 
            sont principalement déterminées par :
            
            1. **Caractéristiques du moteur** : puissance, cylindrée, technologie
            2. **Type de carburant** : impact majeur (Diesel vs Essence vs Hybride)
            3. **Masse du véhicule** : corrélation directe avec la consommation
            4. **Évolution temporelle** : amélioration technologique progressive
            5. **Segment de marché** : différences entre constructeurs et gammes
            
            """)
            
            # Conclusions sur la méthodologie
            st.markdown("")
            st.markdown("#### Robustesse de la Méthodologie")
            
            st.markdown("""
            **✅ Approche rigoureuse :**
            
            1. **Preprocessing exhaustif** : nettoyage, harmonisation, imputation intelligente
            2. **Feature Engineering avancé** : gestion multicolinéarité, encodage adaptatif
            3. **Validation croisée** : division train/test respectée, pas de data leakage
            4. **Comparaison multi-modèles** : évaluation objective sur métriques standards
            5. **Interprétabilité** : analyse SHAP pour comprendre les prédictions
            
            **📊 Métriques de validation :**
            - **R² élevé** : Modèles expliquent une part significative de la variance
            - **MAE faible** : Erreurs de prédiction dans des ordres de grandeur acceptables
            - **Cohérence** : Convergence entre Feature Importance et SHAP
            """)
            
        # Sous-onglet 2: Recommandations
        with conclusions_tabs[1]:
                        
            st.markdown("""         
            Voici nos recommandations stratégiques 
            pour les constructeurs automobiles souhaitant réduire l'empreinte carbone de leurs véhicules.
            """)
            
            # Recommandations techniques
            st.markdown("#### Optimisations Techniques Prioritaires")
            
            st.markdown("""            
            - **Downsizing moteur** : réduire la cylindrée tout en maintenant les performances
            - **Hybridation progressive** : privilégier des motorisations hybride sur les modèles existants
            - **Optimisation combustion** : technologies d'injection directe et gestion électronique avancée                      
            - **Réduction masse à vide** : privilégier les matériaux légers type aluminium, composites, aciers haute résistance
            """)
            
            # Recommandations stratégiques
            st.markdown("""#### Miser sur l'innovation et la R&D""")

            st.markdown("""
            - **Motorisations alternatives** : développer l'électrique & l'hydrogène pour les véhicules lourds.
            - **Récupération d'énergie** : développer les systèmes KERS (Kinetic Energy Recovery System).
            - **Intelligence artificielle** : intégrer des systèmes d'optimisation en temps réel de la combustion.
            """)
                        
           
            # Conclusion finale
            st.markdown("""#### Message Final""")
            
            st.markdown("""
            Les modèles prédictifs développés dans cette analyse fournissent une base scientifique 
            solide pour orienter les décisions d'investissement et les priorités de développement produit.      
            Les constructeurs qui sauront anticiper et intégrer rapidement les technologies de réduction des émissions 
            prendront un avantage concurrentiel décisif.                  
            """)
            
            # cc: 
            st.markdown("---")
            st.success(f"""
            🎉 **Analyse Complète du problème de régression terminée !**           
            """)

    # Bouton de retour au début
    st.markdown("---")

    if st.button("🔝 Retour au début", key="back_to_top_button"):
        st.markdown("""
        <script>
        window.scrollTo(0, 0);
        </script>
        """, unsafe_allow_html=True)

# Exécuter l'application
if __name__ == "__main__":
    show_linear_regression()