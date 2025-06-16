import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Charger les fichiers
@st.cache_data
def load_data():
    X_test = pd.read_csv("X_test.csv")  # ou charger via pickle
    y_test = pd.read_csv("y_test.csv")
    predictions = pd.read_csv("predictions.csv")
    feature_importance = pd.read_csv("feature_importance.csv")  # deux colonnes : Feature, Importance
    return X_test, y_test, predictions, feature_importance

X_test, y_test, predictions, feature_importance = load_data()

# --- Titre du dashboard ---
st.title("📊 Dashboard de Modèle Prédictif")

# --- 1. Résumé des performances ---
st.header("Évaluation du modèle")
r2 = np.corrcoef(predictions["target"], y_test.squeeze())[0, 1]**2
mae = np.mean(np.abs(predictions["target"] - y_test.squeeze()))
mape = np.mean(np.abs((y_test.squeeze() - predictions["target"]) / y_test.squeeze())) * 100
rmse = np.sqrt(np.mean((y_test.squeeze() - predictions["target"]) ** 2))

st.markdown(f"""
- **R²** : `{r2:.4f}`
- **MAE** : `{mae:.2f}`
- **MAPE (%)** : `{mape:.2f}%`
- **RMSE** : `{rmse:.2f}`
""")

# --- 2. Scatter réel vs prédit ---
st.subheader("Valeurs réelles vs prédictions")
fig, ax = plt.subplots()
ax.scatter(y_test, predictions["target"], alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel("Valeurs réelles")
ax.set_ylabel("Prédictions")
ax.set_title("Comparaison Réel / Prédit")
st.pyplot(fig)

# --- 3. Importance des variables ---
st.subheader("Importance des variables")
top_n = st.slider("Nombre de variables à afficher", 5, 30, 10)
top_features = feature_importance.sort_values(by="Importance", ascending=False).head(top_n)
st.bar_chart(top_features.set_index("Feature"))

# --- 4. Prévisualisation des prédictions ---
st.subheader("Explorer les prédictions")
index = st.number_input("Sélectionner un index (ligne test)", 0, len(X_test)-1, 0)
st.write("🔍 **Features de l'individu sélectionné :**")
st.write(X_test.iloc[index])
st.write(f"🎯 **Prédiction du modèle : {predictions['target'].iloc[index]:.2f}**")

# --- 5. Filtrage par région ou ville (si dispo) ---
if "Nom de la région" in X_test.columns:
    st.subheader("Filtrage par région")
    region = st.selectbox("Choisir une région", X_test["Nom de la région"].unique())
    filtered = X_test[X_test["Nom de la région"] == region]
    st.write(f"{len(filtered)} individus trouvés dans cette région.")
    st.write(filtered.head())

st.markdown("---")
st.markdown("🧠 *Dashboard généré avec Streamlit — prêt pour l’analyse interactive !*")
