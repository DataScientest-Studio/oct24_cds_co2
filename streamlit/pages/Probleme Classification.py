import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Classification CO2",
    page_icon="🌱",
    layout="wide"
)

# ==========================================
# DÉFINITION DE LA CLASSE FUELGROUPER AU NIVEAU MODULE
# ==========================================
class FuelGrouper(BaseEstimator, TransformerMixin):
    """Transformer pour grouper les types de carburant"""
    
    def __init__(self):
        self.categories_ = ["GO", "ES", "Autres"]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            transformed_col = X["typ_crb"].map(
                lambda x: x if x in ["GO","ES"] else "Autres"
            ).to_frame(name="typ_crb_grouped")
        else:
            transformed_col = pd.Series(X.flatten()).map(
                lambda x: x if x in ["GO","ES"] else "Autres"
            ).to_frame(name="typ_crb_grouped")
        return transformed_col
    
    def get_feature_names_out(self, input_features=None):
        return np.array(["typ_crb_grouped"])

# Fonction utilitaire pour afficher les DataFrames
def safe_dataframe_display(df, title="DataFrame"):
    """Affiche un DataFrame de manière sécurisée dans Streamlit"""
    try:
        if df is not None and not df.empty:
            st.markdown(f"**{title}**")
            st.dataframe(df, use_container_width=True)
        else:
            st.warning(f"⚠️ {title} est vide ou non défini")
    except Exception as e:
        st.error(f"Erreur lors de l'affichage de {title}: {str(e)}")

# Fonction pour charger les modèles avec gestion d'erreur améliorée
def load_models_and_scalers(models_dir):
    """Charge les modèles et scalers depuis un dossier"""
    models = {}
    scalers = {}
    
    if not os.path.exists(models_dir):
        return models, scalers
    
    # S'assurer que FuelGrouper est disponible dans le namespace global
    import sys
    current_module = sys.modules[__name__]
    if not hasattr(current_module, 'FuelGrouper'):
        setattr(current_module, 'FuelGrouper', FuelGrouper)
    
    for filename in os.listdir(models_dir):
        if filename.endswith('.pkl'):
            filepath = os.path.join(models_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    obj = pickle.load(f)
                
                if 'scaler' in filename.lower():
                    scalers[filename] = obj
                else:
                    models[filename] = obj
                    
            except Exception as e:
                st.warning(f"⚠️ Impossible de charger {filename}: {str(e)}")
    
    return models, scalers

def create_plotly_confusion_matrix(cm, labels, title, color_scale='Blues'):
    """Crée une matrice de confusion avec Plotly"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=color_scale,
        showscale=True,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Prédictions",
        yaxis_title="Vraies valeurs",
        width=450,
        height=400,
        font=dict(size=11)
    )
    
    return fig

def create_enhanced_metrics_table(results_df):
    """Crée un tableau de métriques amélioré avec Plotly"""
    
    # Extraire les valeurs numériques pour le tri
    results_df['Accuracy_num'] = results_df['Accuracy'].str.rstrip('%').astype(float) / 100
    results_df['F1_num'] = results_df['F1-Score'].str.rstrip('%').astype(float) / 100
    
    # Trier par accuracy décroissante
    results_df = results_df.sort_values('Accuracy_num', ascending=False).reset_index(drop=True)
    
    # Identifier le meilleur modèle
    best_model = results_df.iloc[0]['Modèle']
    
    # Couleurs pour les cellules
    def get_color(value, metric_type):
        if metric_type == 'accuracy':
            if value >= 0.95: return 'lightgreen'
            elif value >= 0.90: return 'lightblue'
            elif value >= 0.85: return 'lightyellow'
            else: return 'lightcoral'
        else:  # f1-score
            if value >= 0.95: return 'lightgreen'
            elif value >= 0.90: return 'lightblue'
            elif value >= 0.85: return 'lightyellow'
            else: return 'lightcoral'
    
    # Couleurs pour chaque cellule
    model_colors = ['gold' if model == best_model else 'white' for model in results_df['Modèle']]
    accuracy_colors = [get_color(acc, 'accuracy') for acc in results_df['Accuracy_num']]
    f1_colors = [get_color(f1, 'f1') for f1 in results_df['F1_num']]
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Modèle</b>', '<b>Accuracy</b>', '<b>F1-Score</b>'],
            fill_color='darkslategray',
            font=dict(color='white', size=14),
            align='center',
            height=40
        ),
        cells=dict(
            values=[
                results_df['Modèle'],
                results_df['Accuracy'],
                results_df['F1-Score']
            ],
            fill_color=[model_colors, accuracy_colors, f1_colors],
            align='center',
            font=dict(size=12),
            height=35
        )
    )])
    
    fig.update_layout(
        title="<b>Performance des Modèles de Classification</b>",
        title_x=0.5,
        margin=dict(l=20, r=20, t=60, b=20),
        height=300
    )
    
    return fig, best_model

def show_classification():
    """Fonction principale pour afficher la page de classification"""
    
    st.title("Classification de l'efficacité des émissions de CO2")
    st.markdown("Classer les véhicules par label d'émission ACRISS (A+...F)")
    st.markdown("---")
    
    # Vérification des données
    if 'df_final_ml' not in st.session_state:
        st.error("❌ Aucune donnée disponible. Veuillez d'abord charger et préprocesser les données dans l'onglet précédent.")
        st.info("💡 Retournez à l'onglet 'Problème Régression' pour charger les données.")
        return
    
    # Création des sous-onglets
    sub_tabs = st.tabs([
        "Chargement des données", 
        "Feature Engineering", 
        "Modélisation ML", 
        "Feature Importance",
        "Conclusions/Recommandations"
    ])
    
    # ==========================================
    # SOUS-ONGLET 1: CHARGEMENT DES DONNÉES
    # ==========================================
    with sub_tabs[0]:
        st.markdown ('On récupère une copie du dataframe traité lors du problème de régression')
        # Récupération des données depuis la session (avant split)
        df_final_ml = st.session_state['df_final_ml'].copy()
        
        # Section 1: Récupération du dataframe
        st.markdown("### Etape 1 : Chargement et information du dataframe")
        
        # Informations sur le dataset
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Nombre de lignes", f"{df_final_ml.shape[0]:,}")
        with col2:
            st.metric("📋 Nombre de colonnes", f"{df_final_ml.shape[1]:,}")
        with col3:
            if 'co2' in df_final_ml.columns:
                st.metric("🎯 Valeurs CO2 valides", f"{df_final_ml['co2'].notna().sum():,}")
            else:
                st.metric("❌ Colonne CO2", "Manquante")
        
        # Affichage des colonnes disponibles
        cols_df = pd.DataFrame({
            'Colonne': df_final_ml.columns,
            'Type': df_final_ml.dtypes,
            'Non-null': [df_final_ml[col].notna().sum() for col in df_final_ml.columns],
            'Null': [df_final_ml[col].isna().sum() for col in df_final_ml.columns]
        })
        safe_dataframe_display(cols_df, f'Structure du dataset ({df_final_ml.shape[1]} colonnes)')
        
        st.success("✅ Dataframe chargé avec succès")
        
        # Espace entre les sections
        st.markdown("")
        
        # Section 2: Définition des classes ACRISS
        st.markdown("### Etape 2 : Définition des émissions de CO2 basées sur les catégories ACRISS")
        
        # Définition des classes ACRISS
        co2_bins = [0, 0.001, 50, 75, 95, 130, 225, np.inf]
        co2_labels = ['A+', 'A', 'B', 'C', 'D', 'E', 'F']
        
        class_definition = pd.DataFrame({
            'Classe': co2_labels,
            'Seuil min (g/km)': [0, 0.001, 50, 75, 95, 130, 225],
            'Seuil max (g/km)': [0.001, 50, 75, 95, 130, 225, '∞'],
            'Description': [
                'Véhicules électriques purs',
                'Très faibles émissions',
                'Faibles émissions', 
                'Émissions modérées',
                'Émissions moyennes',
                'Émissions élevées',
                'Très fortes émissions'
            ]
        })
        safe_dataframe_display(class_definition, 'Définition des classes ACRISS')
        
        # Application des classes
        if 'co2' in df_final_ml.columns:
            df_final_ml['co2_efficiency_class'] = pd.cut(
                df_final_ml['co2'],
                bins=co2_bins,
                labels=co2_labels,
                right=True,
                include_lowest=True
            )
            st.success("✅ Classes d'efficacité CO2 créées selon le barème ACRISS")
            
            # Visualisation de la distribution
            st.markdown("#### 📊 Distribution des classes")
            
            # Vérification que les classes ont été créées
            if 'co2_efficiency_class' in df_final_ml.columns:
                class_dist = df_final_ml['co2_efficiency_class'].value_counts().sort_index()
                class_pct = df_final_ml['co2_efficiency_class'].value_counts(normalize=True).sort_index() * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    # Graphique en barres
                    color_map = {
                        'A+': 'rgb(0, 150, 64)', 'A': 'rgb(100, 190, 73)', 
                        'B': 'rgb(170, 205, 57)', 'C': 'rgb(255, 230, 0)',
                        'D': 'rgb(255, 150, 0)', 'E': 'rgb(255, 80, 0)', 
                        'F': 'rgb(220, 0, 0)'
                    }
                    
                    fig_bar = px.bar(
                        x=class_dist.index, 
                        y=class_dist.values,
                        title='Distribution par classe',
                        labels={'x': 'Classe CO2', 'y': 'Nombre de véhicules'},
                        color=class_dist.index,
                        color_discrete_map=color_map
                    )
                    fig_bar.update_layout(showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    # Graphique en secteurs
                    fig_pie = px.pie(
                        values=class_dist.values,
                        names=class_dist.index,
                        title='Répartition en %',
                        color=class_dist.index,
                        color_discrete_map=color_map
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Tableau récapitulatif
                summary_df = pd.DataFrame({
                    'Classe': class_dist.index,
                    'Nombre': class_dist.values,
                    'Pourcentage': [f"{pct:.1f}%" for pct in class_pct.values]
                })
                safe_dataframe_display(summary_df, 'Répartition des classes')
                
            else:
                st.error("❌ Erreur lors de la création des classes")
        else:
            st.error("❌ Colonne 'co2' manquante dans le dataset")
        
        # Sauvegarde pour les autres sous-onglets
        st.session_state['df_final_ml_with_classes'] = df_final_ml
        st.session_state['co2_bins'] = co2_bins
        st.session_state['co2_labels'] = co2_labels
    
    # ==========================================
    # SOUS-ONGLET 2: FEATURE ENGINEERING
    # ==========================================
    with sub_tabs[1]:
        if 'df_final_ml_with_classes' not in st.session_state:
            st.warning("⚠️ Veuillez d'abord exécuter le sous-onglet 'Chargement des données'.")
            return
        
        df_final_ml = st.session_state['df_final_ml_with_classes'].copy()
        
        # Section 1: Création de la nouvelle variable cible
        st.markdown ('Préparation du dataframe final de notre problème de classification et création de la nouvelle variable cible')
        st.markdown("### Etape 1 : Création de la nouvelle variable cible")
        
        # Affichage de la colonne cible créée
        st.markdown("🎯 Variable cible créée: **`co2_efficiency_class`**")
        if 'co2_efficiency_class' in df_final_ml.columns:
            target_info = df_final_ml['co2_efficiency_class'].describe()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total observations", f"{len(df_final_ml):,}")
            with col2:
                st.metric("✅ Valeurs valides", f"{df_final_ml['co2_efficiency_class'].notna().sum():,}")
            with col3:
                st.metric("❌ Valeurs manquantes", f"{df_final_ml['co2_efficiency_class'].isna().sum():,}")
            with col4:
                st.metric("🏷️ Classes uniques", f"{df_final_ml['co2_efficiency_class'].nunique()}")
        else:
            st.error("❌ Variable cible 'co2_efficiency_class' non trouvée")
        
        # Construction de X et y, suppression des NaN
        X = df_final_ml.drop(['co2','co2_efficiency_class'], axis=1)
        y = df_final_ml['co2_efficiency_class']
        
        # Suppression des NaN
        mask = y.notna()
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)
        
        st.success("✅ Variables X et y définies avec succès")
        
        # Espace entre les sections
        st.markdown("")
        
        # Section 2: Séparation train/test
        st.markdown("### Etape 2 : Séparation du jeu d'entraînement et de test")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏋️ Train X", f"{X_train.shape[0]:,}")
        with col2:
            st.metric("🧪 Test X", f"{X_test.shape[0]:,}")
        with col3:
            st.metric("🏋️ Train y", f"{len(y_train):,}")
        with col4:
            st.metric("🧪 Test y", f"{len(y_test):,}")
        
        st.success("✅ Split train/test effectué avec stratification")
        
        # Espace entre les sections
        st.markdown("")
        
        # Section 3: Encodage des labels
        st.markdown("### Etape 3 : Encodage des labels")
        
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        
        # Affichage du mapping
        mapping_df = pd.DataFrame({
            'Classe originale': le.classes_,
            'Valeur encodée': range(len(le.classes_))
        })
        safe_dataframe_display(mapping_df, 'Mapping des classes')
        
        st.success("✅ Labels encodés avec LabelEncoder")
        
        # Espace entre les sections
        st.markdown("")
        
        # Section 4: Pipeline spécifiquement pour le carburant
        st.markdown("### Etape 4 : Pipeline spécifiquement pour le carburant")
        
        crb_pipe = Pipeline([
            ("grouper", FuelGrouper()),
            ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
        ])
        
        st.success("✅ Pipeline carburant créé (GO, ES, Autres)")
        
        # Espace entre les sections
        st.markdown("")
        
        # Section 5: Application des Transformation (ColumnTransformer)
        st.markdown("### Etape 5 : Application des Transformation (ColumnTransformer)")
        
        # Définition des colonnes à transformer
        all_cols = X_train.columns.tolist()
        ohe_basic = [c for c in ["Carrosserie", "gamme"] if c in all_cols]
        freq_cols = [c for c in ["lib_mrq_utac","typ_boite_nb_rapp"] if c in all_cols]
        fuel_col = [c for c in ["typ_crb"] if c in all_cols]
        exclude = ohe_basic + freq_cols + fuel_col
        num_cols = [c for c in all_cols
                   if c not in exclude
                   and X_train[c].dtype in ("int64","float64")]
        
        transformers = []
        if ohe_basic:
            transformers.append((
                "ohe_basic",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                ohe_basic
            ))
        if freq_cols:
            transformers.append((
                "ordinal_freq",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                freq_cols
            ))
        if fuel_col:
            transformers.append(("ohe_crb", crb_pipe, fuel_col))
        if num_cols:
            transformers.append(("scaler", StandardScaler(), num_cols))
        
        preprocessor = ColumnTransformer(
            transformers, 
            remainder="drop", 
            verbose_feature_names_out=True
        )
        
        # Affichage des transformers
        transform_df = pd.DataFrame({
            'Transformer': [t[0] for t in transformers],
            'Type': [str(type(t[1]).__name__) for t in transformers],
            'Colonnes': [str(t[2]) for t in transformers]
        })
        safe_dataframe_display(transform_df, 'Transformers configurés')
        
        st.success("✅ ColumnTransformer configuré")
        
        # Espace entre les sections
        st.markdown("")
        
        # Informations sur les prochaines étapes
        st.markdown("---")
        st.info("""
        **🚀 Prochaines étapes :**
        
        Naviguez vers l'onglet **Modélisation ML** pour :
        -  **Charger les modèles pré-entraînés** : Decision Tree, Random Forest, Gradient Boosting
        -  **Évaluer les performances** : accuracy, F1-Score, matrices de confusion
        -  **Comparer les modèles** : Standard vs GridSearchCV
        -  **Sélectionner le meilleur modèle** : basé sur les métriques de performance
        -  **Visualiser les résultats** : distribution des classes prédites
        """)
        
        # Sauvegarde pour le sous-onglet suivant
        st.session_state['X_train_clf'] = X_train
        st.session_state['X_test_clf'] = X_test
        st.session_state['y_train_clf'] = y_train
        st.session_state['y_test_clf'] = y_test
        st.session_state['y_train_enc_clf'] = y_train_enc
        st.session_state['y_test_enc_clf'] = y_test_enc
        st.session_state['le_clf'] = le
        st.session_state['preprocessor_clf'] = preprocessor
        st.session_state['FuelGrouper'] = FuelGrouper
    
    # ==========================================
    # SOUS-ONGLET 3: MODÉLISATION ML (MODIFIÉ)
    # ==========================================
    with sub_tabs[2]:
        # Section 1: Dictionnaire des modèles
        st.markdown ('Récupération du dataframe post Feature Engineering et sélection des modèles adaptés')
        st.markdown("### Etape 1 : Définition du dictionnaire des modèles & grilles")
        
        models_and_grids = {
            "Decision Tree": (
                DecisionTreeClassifier(random_state=42),
                {
                    "clf__max_depth": [None, 3, 5, 10],
                    "clf__min_samples_split": [2, 5, 10],
                    "clf__criterion": ["gini","entropy"]
                }
            ),
            "Random Forest": (
                RandomForestClassifier(random_state=42),
                {
                    "clf__n_estimators": [50, 100],
                    "clf__max_depth": [None, 5, 10],
                    "clf__min_samples_leaf": [1, 2]
                }
            ),
            "Gradient Boosting": (
                GradientBoostingClassifier(random_state=42),
                {
                    "clf__n_estimators": [100, 200],
                    "clf__learning_rate": [0.05, 0.1],
                    "clf__max_depth": [3, 5]
                }
            )
        }
        
        # Affichage des modèles
        models_df = pd.DataFrame({
            'Modèle': list(models_and_grids.keys()),
            'Paramètres à optimiser': [len(grid) for _, (_, grid) in models_and_grids.items()],
            'Combinaisons': [
                np.prod([len(v) for v in grid.values()]) 
                for _, (_, grid) in models_and_grids.items()
            ]
        })
        safe_dataframe_display(models_df, 'Modèles et grilles de recherche')
        
        st.success("✅ 3 modèles configurés avec GridSearchCV")
        
        # Section 2: Chargement des modèles (MODIFIÉ)
        st.markdown("### Etape 2 : Chargement et évaluation des modèles")
        
        if st.button('🚀 Charger tous les modèles'):
            models_dir = "saved_models_classification"
            if os.path.exists(models_dir):
                try:
                    # Vérification des données de test
                    if not all(key in st.session_state for key in ['X_test_clf', 'y_test_enc_clf', 'le_clf']):
                        st.error("❌ Données de test manquantes. Veuillez d'abord exécuter le preprocessing.")
                        return
                    
                    X_test = st.session_state['X_test_clf']
                    y_test_enc = st.session_state['y_test_enc_clf']
                    le = st.session_state['le_clf']
                    
                    # Chargement des modèles avec la fonction corrigée
                    loaded_models, _ = load_models_and_scalers(models_dir)
                    
                    if loaded_models:
                        st.success(f"✅ Modèles chargés depuis {models_dir}")
                        
                        # Dictionnaire pour stocker les résultats
                        all_results = []
                        confusion_matrices = {}  # Pour stocker les matrices de confusion
                        classification_reports = {}  # Pour stocker les rapports de classification
                        
                        # Évaluation de chaque modèle
                        model_types = ["Decision Tree", "Random Forest", "Gradient Boosting"]
                        
                        for model_type in model_types:
                            # === MODÈLE STANDARD ===
                            standard_filename = f"clf_conventionnel_{model_type.lower().replace(' ', '_')}.pkl"
                            if standard_filename in loaded_models:
                                model = loaded_models[standard_filename]
                                
                                # Prédictions
                                y_pred = model.predict(X_test)
                                accuracy = accuracy_score(y_test_enc, y_pred)
                                f1 = f1_score(y_test_enc, y_pred, average='weighted')
                                
                                # Sauvegarde des résultats
                                all_results.append({
                                    'Modèle': f"{model_type} (Standard)",
                                    'Accuracy': f"{accuracy:.2%}",
                                    'F1-Score': f"{f1:.2%}"
                                })
                                
                                # Sauvegarde pour matrices de confusion
                                y_pred_labels = le.inverse_transform(y_pred)
                                y_test_labels = le.inverse_transform(y_test_enc)
                                
                                cm = confusion_matrix(y_test_labels, y_pred_labels, labels=le.classes_)
                                confusion_matrices[f"{model_type} (Standard)"] = {
                                    'matrix': cm,
                                    'labels': le.classes_,
                                    'color': 'Blues'
                                }
                                
                                # Rapport de classification
                                report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
                                classification_reports[f"{model_type} (Standard)"] = report
                            
                            # === MODÈLE GRIDSEARCH ===
                            gridsearch_filename = f"clf_gridsearch_{model_type.lower().replace(' ', '_')}.pkl"
                            if gridsearch_filename in loaded_models:
                                model = loaded_models[gridsearch_filename]
                                
                                # Prédictions
                                y_pred = model.predict(X_test)
                                accuracy = accuracy_score(y_test_enc, y_pred)
                                f1 = f1_score(y_test_enc, y_pred, average='weighted')
                                
                                # Sauvegarde des résultats
                                all_results.append({
                                    'Modèle': f"{model_type} (GridSearch)",
                                    'Accuracy': f"{accuracy:.2%}",
                                    'F1-Score': f"{f1:.2%}"
                                })
                                
                                # Sauvegarde pour matrices de confusion
                                y_pred_labels = le.inverse_transform(y_pred)
                                y_test_labels = le.inverse_transform(y_test_enc)
                                
                                cm = confusion_matrix(y_test_labels, y_pred_labels, labels=le.classes_)
                                confusion_matrices[f"{model_type} (GridSearch)"] = {
                                    'matrix': cm,
                                    'labels': le.classes_,
                                    'color': 'Greens'
                                }
                                
                                # Rapport de classification
                                report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
                                classification_reports[f"{model_type} (GridSearch)"] = report
                        
                        # === AFFICHAGE DES MÉTRIQUES (NOUVEAU DESIGN) ===
                        st.markdown("---")
                        st.markdown("### Performance des Modèles")
                        
                        results_df = pd.DataFrame(all_results)
                        
                        # Tableau amélioré avec Plotly
                        fig_metrics, best_model_name = create_enhanced_metrics_table(results_df)
                        st.plotly_chart(fig_metrics, use_container_width=True)
                        
                        
                        # Analyse des résultats
                        st.markdown("#### Analyse des Performances")
                        
                        # Extraire les valeurs numériques pour l'analyse
                        results_df['Accuracy_num'] = results_df['Accuracy'].str.rstrip('%').astype(float) / 100
                        results_df['F1_num'] = results_df['F1-Score'].str.rstrip('%').astype(float) / 100
                        
                        best_accuracy = results_df['Accuracy_num'].max()
                        worst_accuracy = results_df['Accuracy_num'].min()
                        avg_accuracy = results_df['Accuracy_num'].mean()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🏆 Meilleure Accuracy", f"{best_accuracy:.1%}")
                        with col2:
                            st.metric("📊 Accuracy Moyenne", f"{avg_accuracy:.1%}")
                        with col3:
                            st.metric("📉 Écart Min-Max", f"{(best_accuracy - worst_accuracy):.1%}")
                        
                        # Interprétations
                        st.markdown("**💡 Interprétations :**")
                        
                        if best_accuracy >= 0.95:
                            st.success("✅ **Excellente performance** : Le meilleur modèle atteint une accuracy supérieure à 95%, indiquant une très bonne capacité de classification des émissions CO2.")
                        elif best_accuracy >= 0.90:
                            st.info("ℹ️ **Bonne performance** : Le meilleur modèle atteint une accuracy entre 90-95%, ce qui est satisfaisant pour ce type de classification.")
                        else:
                            st.warning("⚠️ **Performance modérée** : Le meilleur modèle a une accuracy inférieure à 90%, il pourrait être nécessaire d'optimiser davantage.")
                        
                        # Comparaison Standard vs GridSearch
                        standard_models = results_df[results_df['Modèle'].str.contains('Standard')]
                        gridsearch_models = results_df[results_df['Modèle'].str.contains('GridSearch')]
                        
                        if not standard_models.empty and not gridsearch_models.empty:
                            avg_standard = standard_models['Accuracy_num'].mean()
                            avg_gridsearch = gridsearch_models['Accuracy_num'].mean()
                            improvement = avg_gridsearch - avg_standard
                            
                            if improvement > 0.01:
                                st.success(f"📈 **Impact positif de GridSearchCV** : Amélioration moyenne de {improvement:.1%} par rapport aux modèles standards.")
                            elif improvement > 0:
                                st.info(f"📊 **Légère amélioration avec GridSearchCV** : Gain de {improvement:.1%}.")
                            else:
                                st.warning("⚠️ **GridSearchCV sans impact significatif** : Les hyperparamètres par défaut semblent déjà optimaux.")
                        
                        # Chargement et sauvegarde du meilleur modèle
                        best_model_file = os.path.join(models_dir, "clf_best_overall_pipeline.pkl")
                        if os.path.exists(best_model_file) and "clf_best_overall_pipeline.pkl" in loaded_models:
                            best_model = loaded_models["clf_best_overall_pipeline.pkl"]
                            st.session_state['best_overall_pipeline_clf'] = best_model
                            st.session_state['clf_results'] = results_df
                            st.session_state['confusion_matrices_clf'] = confusion_matrices
                            st.session_state['classification_reports_clf'] = classification_reports
                            
                            st.success(f"🏆 **Meilleur modèle identifié** : {best_model_name}")
                        
                    else:
                        st.warning("⚠️ Aucun modèle trouvé dans le dossier")
                        
                except Exception as e:
                    st.error(f"❌ Erreur lors du chargement: {str(e)}")
                    st.exception(e)
            else:
                st.error(f"❌ Dossier {models_dir} introuvable")

        st.markdown("---")
        st.info("""
        **🚀 Prochaines étapes :**

        Naviguez vers l'onglet **Feature Importance** pour :
        - Analyser les matrices de confusion détaillées
        - Visualiser la distribution des classes prédites
        - Comprendre les erreurs de classification
        """)

    
    # ==========================================
    # SOUS-ONGLET 4: FEATURE IMPORTANCE (MODIFIÉ)
    # ==========================================
    with sub_tabs[3]:
        if 'best_overall_pipeline_clf' not in st.session_state:
            st.warning("⚠️ Veuillez d'abord charger les modèles dans l'onglet 'Modélisation ML'.")
            return
        
        st.markdown("Analyse détaillée des performances et visualisation des résultats de classification")
        
        # Section 1: Matrices de confusion
        st.markdown("### Matrices de Confusion Détaillées")
        
        if 'confusion_matrices_clf' in st.session_state:
            confusion_matrices = st.session_state['confusion_matrices_clf']
            
            # Affichage en grille 2x3
            models_list = list(confusion_matrices.keys())
            
            # Première ligne : Modèles Standard
            st.markdown("#### 🔧 Modèles Standard")
            standard_models = [m for m in models_list if 'Standard' in m]
            
            if standard_models:
                cols = st.columns(len(standard_models))
                for i, model_name in enumerate(standard_models):
                    with cols[i]:
                        cm_data = confusion_matrices[model_name]
                        fig = create_plotly_confusion_matrix(
                            cm_data['matrix'],
                            cm_data['labels'],
                            model_name,
                            cm_data['color']
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Deuxième ligne : Modèles GridSearch
            st.markdown("#### 🎯 Modèles GridSearch")
            gridsearch_models = [m for m in models_list if 'GridSearch' in m]
            
            if gridsearch_models:
                cols = st.columns(len(gridsearch_models))
                for i, model_name in enumerate(gridsearch_models):
                    with cols[i]:
                        cm_data = confusion_matrices[model_name]
                        fig = create_plotly_confusion_matrix(
                            cm_data['matrix'],
                            cm_data['labels'],
                            model_name,
                            cm_data['color']
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Analyse des matrices de confusion
            st.markdown("#### Analyse des Matrices de Confusion")
            
            st.markdown("""
                        
            - **Diagonale principale** : prédictions correctes (plus les valeurs sont élevées, mieux c'est)
            - **Hors diagonale** : erreurs de classification (plus les valeurs sont faibles, mieux c'est)
            - **Classes adjacentes** : erreurs entre classes proches (A↔B, B↔C) sont plus acceptables
            - **Classes distantes** : erreurs entre classes éloignées (A↔F) sont plus problématiques
            """)
            
            # Analyse quantitative des erreurs
            if confusion_matrices:
                # Prendre le meilleur modèle pour l'analyse
                best_model_name = list(confusion_matrices.keys())[0]  # Ou identifier le meilleur
                best_cm = confusion_matrices[best_model_name]['matrix']
                labels = confusion_matrices[best_model_name]['labels']
                
                # Calcul des métriques par classe
                total_per_class = best_cm.sum(axis=1)
                correct_per_class = np.diag(best_cm)
                accuracy_per_class = correct_per_class / total_per_class
                
                class_analysis = pd.DataFrame({
                    'Classe': labels,
                    'Total échantillons': total_per_class,
                    'Prédictions correctes': correct_per_class,
                    'Accuracy par classe': [f"{acc:.1%}" for acc in accuracy_per_class]
                })
                
                safe_dataframe_display(class_analysis, f"Analyse par classe du meilleur modèle - {best_model_name}")
                
                # Identification des classes problématiques
                worst_classes = class_analysis.loc[accuracy_per_class < 0.8, 'Classe'].tolist()
                if worst_classes:
                    st.warning(f"⚠️ **Classes avec accuracy < 80%** : {', '.join(worst_classes)}")
                    st.markdown("Ces classes nécessitent une attention particulière et pourraient bénéficier de plus de données d'entraînement ou de features supplémentaires.")
                else:
                    st.success("✅ **Toutes les classes ont une accuracy ≥ 80%** - Performance équilibrée")
        
        # Section 2: Distribution des prédictions
        st.markdown("---")
        st.markdown("### Visualisation de la distribution des véhicules par classe d'efficacité CO2")
        
        best_overall_pipeline = st.session_state['best_overall_pipeline_clf']
        
        if 'df_final_ml_with_classes' not in st.session_state:
            st.warning("⚠️ Dataset avec classes manquant.")
            return
        
        df_final_ml = st.session_state['df_final_ml_with_classes']
        le = st.session_state.get('le_clf')
        
        # Prédiction sur l'ensemble des données
        X_full = df_final_ml.drop(['co2', 'co2_efficiency_class'], axis=1)
        
        try:
            preds = best_overall_pipeline.predict(X_full)
            
            if le is not None:
                pred_labels = le.inverse_transform(preds)
            else:
                pred_labels = preds
            
            # Ajout des prédictions
            df_final_ml['predicted_co2_class'] = pred_labels
            
            # Calcul des pourcentages
            pred_class_percentages = df_final_ml['predicted_co2_class'].value_counts(normalize=True).reset_index()
            pred_class_percentages.columns = ['CO2_Class', 'Percentage']
            
            # Ordre des classes et couleurs
            class_order = ['A+', 'A', 'B', 'C', 'D', 'E', 'F']
            color_map = {
                'A+': 'rgb(0, 150, 64)', 'A': 'rgb(100, 190, 73)', 
                'B': 'rgb(170, 205, 57)', 'C': 'rgb(255, 230, 0)',
                'D': 'rgb(255, 150, 0)', 'E': 'rgb(255, 80, 0)', 
                'F': 'rgb(220, 0, 0)'
            }
            
            # Reformatage des données
            percentage_data = pred_class_percentages.set_index('CO2_Class')['Percentage'].reindex(class_order, fill_value=0)
            
            # Création de la visualisation ACRISS
            shapes = [
                # Band F (y=0 à 1)
                go.layout.Shape(type="path", path=" M 0,0 L 4,0 L 3.5,1 L 0,1 Z", 
                               fillcolor=color_map['F'], line=dict(color='white', width=1)),
                # Band E (y=1 à 2)
                go.layout.Shape(type="path", path=" M 0,1 L 3.5,1 L 3,2 L 0,2 Z", 
                               fillcolor=color_map['E'], line=dict(color='white', width=1)),
                # Band D (y=2 à 3)
                go.layout.Shape(type="path", path=" M 0,2 L 3,2 L 2.5,3 L 0,3 Z", 
                               fillcolor=color_map['D'], line=dict(color='white', width=1)),
                # Band C (y=3 à 4)
                go.layout.Shape(type="path", path=" M 0,3 L 2.5,3 L 2,4 L 0,4 Z", 
                               fillcolor=color_map['C'], line=dict(color='white', width=1)),
                # Band B (y=4 à 5)
                go.layout.Shape(type="path", path=" M 0,4 L 2,4 L 1.5,5 L 0,5 Z", 
                               fillcolor=color_map['B'], line=dict(color='white', width=1)),
                # Band A (y=5 à 6)
                go.layout.Shape(type="path", path=" M 0,5 L 1.5,5 L 1,6 L 0,6 Z", 
                               fillcolor=color_map['A'], line=dict(color='white', width=1)),
                # Band A+ (y=6 à 7)
                go.layout.Shape(type="path", path=" M 0,6 L 1,6 L 1,7 L 0,7 Z", 
                               fillcolor=color_map['A+'], line=dict(color='white', width=1)),
            ]
            
            annotations = []
            # Annotations des lettres
            letter_annotation_data = {
                'A+': (0.5, 6.5), 'A': (0.75, 5.5), 'B': (1, 4.5), 'C': (1.25, 3.5),
                'D': (1.5, 2.5), 'E': (1.75, 1.5), 'F': (2, 0.5),
            }
            for cls, pos in letter_annotation_data.items():
                annotations.append(
                    go.layout.Annotation(
                        x=pos[0], y=pos[1], text=cls, showarrow=False,
                        font=dict(color='black', size=20, weight='bold'),
                        xanchor='center', yanchor='middle'
                    )
                )
            
            # Annotations des pourcentages
            percentage_annotation_data = {
                'A+': (4.5, 6.5), 'A': (4.5, 5.5), 'B': (4.5, 4.5), 'C': (4.5, 3.5),
                'D': (4.5, 2.5), 'E': (4.5, 1.5), 'F': (4.5, 0.5),
            }
            for cls in class_order:
                percentage = percentage_data.get(cls, 0)
                pos = percentage_annotation_data[cls]
                annotations.append(
                    go.layout.Annotation(
                        x=pos[0], y=pos[1], text=f'{percentage:.1%}', showarrow=False,
                        font=dict(color='black', size=12),
                        xanchor='left', yanchor='middle'
                    )
                )
            
            # Création de la figure avec titre repositionné
            fig = go.Figure()
            fig.update_layout(
                shapes=shapes, 
                annotations=annotations,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 6]),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 7.2]),
                title="Distribution des classes CO2 (Prédictions du meilleur modèle)",
                title_x=0.02,  # Positionnement à gauche
                title_y=0.98,  # Positionnement en haut
                showlegend=False,
                height=450,
                width=650,
                margin=dict(l=10, r=10, t=40, b=10),
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau récapitulatif
            distribution_df = pd.DataFrame({
                'Classe': class_order,
                'Nombre de véhicules': [df_final_ml['predicted_co2_class'].value_counts().get(cls, 0) for cls in class_order],
                'Pourcentage': [f"{percentage_data.get(cls, 0):.1%}" for cls in class_order]
            })
            safe_dataframe_display(distribution_df)
            
            # Analyse de la distribution
            st.markdown("#### 📈 Analyse de la Distribution")
            
            # Classes les plus représentées
            top_classes = distribution_df.nlargest(3, 'Nombre de véhicules')['Classe'].tolist()
            st.info(f"**Classes les plus représentées** : {', '.join(top_classes)}")
            
            # Répartition écologique
            eco_classes = ['A+', 'A', 'B']
            eco_percentage = sum(percentage_data.get(cls, 0) for cls in eco_classes)
            
            if eco_percentage > 0.5:
                st.success(f"✅ **Bonne répartition écologique** : {eco_percentage:.1%} des véhicules sont classés A+, A ou B")
            elif eco_percentage > 0.3:
                st.warning(f"⚠️ **Répartition écologique modérée** : {eco_percentage:.1%} des véhicules sont classés A+, A ou B")
            else:
                st.error(f"**Répartition écologique faible** : Seulement {eco_percentage:.1%} des véhicules sont classés A+, A ou B")
            
            st.success("✅ Notre meilleur modèle prédit la classe CO2 de tout le dataset et affiche la distribution (%) de ces classes prédites avec une visualisation façon barème ACRISS")
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction: {str(e)}")
            st.exception(e)

    # ==========================================
    # SOUS-ONGLET 5: CONCLUSIONS/RECOMMANDATIONS (MODIFIÉ)
    # ==========================================
    with sub_tabs[4]:
        st.markdown("Synthèse des résultats et recommandations stratégiques pour l'optimisation des émissions CO2")
        
        # Création de sous-sections
        conclusion_sections = st.tabs(["📊 Conclusions", "🎯 Recommandations"])
        
        # Section 1: Synthèse des Résultats
        with conclusion_sections[0]:
                       
            # Performance du modèle (MODIFIÉ SELON LE MODÈLE DE RÉGRESSION)
            st.markdown("#### Performance des Modèles de Machine Learning")
            
            if 'clf_results' in st.session_state:
                results_df = st.session_state['clf_results']
                
                # Extraire les métriques du meilleur modèle
                if not results_df.empty:
                    results_df['Accuracy_num'] = results_df['Accuracy'].str.rstrip('%').astype(float) / 100
                    results_df['F1_num'] = results_df['F1-Score'].str.rstrip('%').astype(float) / 100
                    best_model = results_df.loc[results_df['Accuracy_num'].idxmax()]
                    
                    st.markdown("🏆 **Meilleur modèle identifié :**")
                    st.markdown("")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🥇 Modèle", best_model['Modèle'])
                    with col2:
                        st.metric("📈 Accuracy", best_model['Accuracy'])
                    with col3:
                        st.metric("🎯 F1-Score", best_model['F1-Score'])
                    
                    st.markdown("")
                    st.markdown("📈 **Enseignements sur les algorithmes :**")
                    st.markdown("")
                    
                    # Analyse de performance
                    accuracy_val = best_model['Accuracy_num']
                    f1_val = best_model['F1_num']
                    st.markdown("""
                    - **Modèles d'ensemble** (Random Forest, Gradient Boosting) : excellentes performances grâce à leur robustesse et capacité à gérer les données complexes
                    - **Decision Tree** : performance correcte avec une excellente interprétabilité des règles de classification
                    - **Gestion des classes multiples** : les modèles basés sur les arbres capturent efficacement les seuils ACRISS et les transitions entre classes
                    - **Impact de l'optimisation** : GridSearchCV apporte une amélioration moyenne de 1.1% par rapport aux modèles standards
                    """)

            
            # Analyse des classes ACRISS
            st.markdown("#### Analyse des Classes d'Efficacité ACRISS")
            
            st.markdown("""
            **✅ Points forts identifiés :**
            
            - **Classification cohérente** : le modèle respecte la logique des seuils ACRISS
            - **Discrimination efficace** : bonne séparation entre les classes d'émissions
            - **Robustesse** : performance stable sur différents types de véhicules
            - **Interprétabilité** : résultats alignés avec les standards européens
            
            **📊 Distribution observée :**
            - Les classes intermédiaires (C, D, E) sont les plus représentées
            - Faible proportion de véhicules A+ (électriques purs) pour notre dataset
            """)
            
            # Facteurs d'influence
            st.markdown("#### Facteurs d'Influence Identifiés")
            
            st.markdown("""
            **🎯 Variables les plus discriminantes :**
            
            - **Type de carburant** : impact majeur sur la classification (Essence vs Diesel vs Hybride)
            - **Puissance du moteur** : confirmation de la corrélation directe avec les émissions
            - **Masse du véhicule** : confirmation de l'influence significative sur la consommation

            
            **💡 Indicateurs métier :**
            - Les véhicules hybrides sont systématiquement mieux classés
            - La puissance administrative reste un prédicteur fiable
            - L'évolution temporelle montre une amélioration progressive
            """)
                       
        
        # Section 2: Recommandations Stratégiques (MODIFIÉ)
        with conclusion_sections[1]:
            st.markdown("#### Optimisations Techniques Prioritaires")
            
            # Recommandations pour les constructeurs
                        
            st.markdown("""            
            - **Électrification accélérée** : augmenter la part de véhicules A+ et A dans la gamme
            - **Hybridation systématique** : intégrer des technologies d'hybridation sur les modèles classés C et D
            - **Downsizing moteur** : réduire la cylindrée tout en maintenant les performances
            - **Réduction masse à vide** : privilégier les matériaux légers type aluminium, composites, aciers haute résistance
            
            """)
            
            # Recommandations réglementaires
            st.markdown('')
            st.markdown("#### Implications Réglementaires")
            
            st.markdown("""

            - Prioriser les véhicules classes A, B, C dans les volumes
            - Développer des offres spécifiques pour les marchés sensibles au CO2
            - Anticiper les évolutions réglementaires par la R&D
            """)
            
            # Message de conclusion
            st.markdown("---")
            st.markdown("#### Message Final")
            
            st.markdown("""           
            La classification  des émissions CO2 représente un outil stratégique majeur pour l'industrie automobile. 
            Au-delà de la simple conformité réglementaire, elle permet :
            
            - **L'optimisation produit** basée sur des données objectives
            - **L'anticipation des évolutions** technologiques et réglementaires  
            - **La différenciation concurrentielle** par l'innovation
            - **L'aide à la décision** pour consommateurs et professionnels
            
            """)
            
            # Prochaines étapes
            st.markdown("---")
            st.markdown("#### Capitalisation de ce projet")
            
            st.info("""
                       
            - Déploiement du modèle en production comme outil d'aide à la conception de véhicule
            - Enrichissement des modèles avec de nouvelles données ou bases venant de l'UE
            - Extension à d'autres types de véhicules et marchés internationaux
            """)
            
            # Message Final
            st.markdown("---")
            st.success("🎉 **Analyse Complète du problème de classification terminée !**")

# Point d'entrée principal
if __name__ == "__main__":
    show_classification()