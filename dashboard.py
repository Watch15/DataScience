import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# --- Chargement des données avec mise en cache ---
@st.cache_data
def load_data():
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv")
    predictions = pd.read_csv("predictions.csv").round(2)
    feature_importance = pd.read_csv("feature_importance.csv")
    return X_test, y_test, predictions, feature_importance

X_test, y_test, predictions, feature_importance = load_data()

# --- Barre latérale ---
with st.sidebar:
    st.title("⚙️ Options")
    page = st.radio("📂 Choisir une page :", ["Accueil", "Performances", "Exploration"])
    top_n = st.slider("🎯 Variables importantes (top N)", 5, 30, 10)
    selected_index = st.number_input("🔍 Index d'observation à explorer", 0, len(X_test)-1, 0)

# --- Page Accueil ---
if page == "Accueil":
    st.title("📊 Dashboard de Modèle Prédictif")
    st.markdown("""
    Bienvenue !  
    Ce dashboard vous permet de :
    - Visualiser la performance du modèle (R², MAPE, etc.)
    - Comparer valeurs réelles vs prédictions
    - Explorer l’importance des variables
    - Inspecter des individus spécifiques
    - Télécharger les résultats
    """)

# --- Page Performances ---
elif page == "Performances":
    st.title("📈 Évaluation du modèle")

    # --- Scores ---
    r2 = np.corrcoef(predictions["target"], y_test.squeeze())[0, 1]**2
    mae = np.mean(np.abs(predictions["target"] - y_test.squeeze()))
    mape = np.mean(np.abs((y_test.squeeze() - predictions["target"]) / y_test.squeeze())) * 100
    rmse = np.sqrt(np.mean((y_test.squeeze() - predictions["target"]) ** 2))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{r2:.3f}")
    col2.metric("MAE", f"{mae:.2f}")
    col3.metric("MAPE (%)", f"{mape:.2f}")
    col4.metric("RMSE", f"{rmse:.2f}")

    with st.expander("ℹ️ À propos des métriques d'évaluation"):
        st.write("""
        - **R²** : proportion de variance expliquée par le modèle.
        - **MAE** : moyenne des erreurs absolues.
        - **MAPE** : pourcentage moyen d'erreur absolue.
        - **RMSE** : racine de l’erreur quadratique moyenne.
        """)

    # --- Scatter réel vs prédictions avec coloration par erreur ---
    st.subheader("Valeurs réelles vs prédictions")
    errors = np.abs(y_test.squeeze() - predictions["target"])
    fig, ax = plt.subplots()
    scatter = ax.scatter(y_test, predictions["target"], c=errors, cmap="coolwarm", alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Valeurs réelles")
    ax.set_ylabel("Prédictions")
    ax.set_title("Comparaison Réel / Prédit")
    fig.colorbar(scatter, ax=ax, label="Erreur absolue")
    st.pyplot(fig)

    # --- Importances des variables (Plotly) ---
    st.subheader("Importance des variables")
    top_features = feature_importance.sort_values(by="Importance", ascending=False).head(top_n)
    fig = px.bar(top_features, x="Importance", y="Feature", orientation="h", title="Top variables")
    st.plotly_chart(fig)

    # --- Téléchargement des prédictions ---
    st.download_button(
        label="📥 Télécharger les prédictions (CSV)",
        data=predictions.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv"
    )

# --- Page Exploration ---
elif page == "Exploration":
    st.title("🔍 Exploration des données")

    st.subheader("Features de l’individu sélectionné")
    st.write(X_test.iloc[selected_index])
    st.write(f"🎯 **Prédiction du modèle : {predictions['target'].iloc[selected_index]:.2f}**")

    # Exploration libre d'une variable
    st.subheader("Distribution d'une variable")
    selected_col = st.selectbox("Choisir une variable", X_test.columns)
    st.write(X_test[selected_col].value_counts().head(10))
    st.bar_chart(X_test[selected_col].value_counts().head(10))

    # Erreurs extrêmes
    st.subheader("📉 Plus grandes erreurs du modèle")
    X_test_errors = X_test.copy()
    X_test_errors["true"] = y_test.values
    X_test_errors["pred"] = predictions["target"]
    X_test_errors["abs_error"] = np.abs(X_test_errors["true"] - X_test_errors["pred"])
    st.dataframe(X_test_errors.sort_values("abs_error", ascending=False).head(10))

    # Carte interactive si coordonnées disponibles
    if "LAT" in X_test.columns and "LONG" in X_test.columns:
        st.subheader("🗺️ Carte des prédictions")
        map_data = X_test.copy()
        map_data["target"] = predictions["target"]
        st.map(map_data[["LAT", "LONG"]])
