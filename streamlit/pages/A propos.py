import streamlit as st

# --------------------------------------------------------------------
# 3) PAGE À PROPOS
# --------------------------------------------------------------------
def show_about():
    st.title("Projet DataScientest - Émissions de CO2")
    st.markdown("---")
    st.write("""
    **Auteurs** : Jonathan LANGNER / Jonathan CHICHEPORTICHE
    

    _Cette application a été développée avec Streamlit pour présenter la modélisation des émissions de CO2._
    """)
    st.markdown("---")
    st.subheader("Détails du projet :")
    st.markdown("""
    - **Objectif** : Fournir une plateforme interactive pour comprendre et modéliser les émissions de CO2.
    - **Technologies utilisées** :
        - Streamlit : Pour la création de l'application web.
        - Pandas : Pour la manipulation et l'analyse des données.
        - Scikit-learn : Pour les algorithmes d'apprentissage automatique (Régression Linéaire, Régression Logistique) et les métriques d'évaluation.
        - Matplotlib et Seaborn : Pour la visualisation des données.
    - **Données** : Les utilisateurs peuvent télécharger leurs propres fichiers CSV pour analyse.
    """)

    st.markdown("---")
    st.subheader("Nous contacter :")
    st.markdown("[Lien vers votre dépôt GitHub (https://github.com/DataScientest-Studio/oct24_cds_co2)")
    st.markdown("N'hésitez pas à nous contacter pour toute question ou commentaire !")

if __name__ == "__main__":
    show_about()