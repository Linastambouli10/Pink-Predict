import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# CONFIG
st.set_page_config(page_title="Pink-Predict", layout="wide", page_icon="🎗️")

# MODELE
@st.cache_resource
def charger_modele():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    model = LogisticRegression(max_iter=5000)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    model.fit(X_scaled, y)
    return model, scaler, data.feature_names, df.mean(), df.std()

model, scaler, feature_names, mean_values, std_values = charger_modele()

# SIDEBAR
st.sidebar.header("Paramètres Patient")
seuil = st.sidebar.slider("Seuil d'alerte (Sensibilité)", 0.0, 1.0, 0.5)
st.sidebar.markdown("---")
def input_slider(label, key):
    avg = float(mean_values[key])
    std = float(std_values[key])
    return st.sidebar.slider(label, max(0.0, avg - 3*std), avg + 4*std, avg)

radius = input_slider("Rayon (Radius)", 'mean radius')
texture = input_slider("Texture", 'mean texture')
perimeter = input_slider("Périmètre", 'mean perimeter')
area = input_slider("Aire", 'mean area')
concavity = input_slider("Concavité", 'mean concavity')

# DATA PREP
input_dict = mean_values.to_dict()
input_dict['mean radius'] = radius
input_dict['mean texture'] = texture
input_dict['mean perimeter'] = perimeter
input_dict['mean area'] = area
input_dict['mean concavity'] = concavity
input_df = pd.DataFrame([input_dict], columns=feature_names)
input_scaled = scaler.transform(input_df)

# PREDICTION
proba = model.predict_proba(input_scaled)
malin_prob = proba[0][0]
is_cancer = malin_prob > seuil

# AFFICHAGE
st.title("💗🎗️ Pink-Predict : Analyseur Tumeur IA")
col_diag, col_radar = st.columns([1, 2])

with col_diag:
    st.subheader("1. Diagnostic")
    if is_cancer:
        st.error("🔴 POSITIF (Risque Élevé)")
        st.metric("Probabilité Malignité", f"{malin_prob*100:.1f}%")
        if malin_prob < 0.5: st.caption(f"⚠️ Seuil strict ({seuil})")
    else:
        st.success("🟢 NÉGATIF (Risque Faible)")
        st.metric("Probabilité Malignité", f"{malin_prob*100:.1f}%")
    st.progress(int(malin_prob*100))

with col_radar:
    st.subheader("2. Profil Visuel Comparatif")
    categories = ['Rayon', 'Texture', 'Périmètre', 'Aire', 'Concavity']
    keys = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean concavity']
    patient_vals = [radius, texture, perimeter, area, concavity]
    avg_vals = [float(mean_values[k]) for k in keys]
    patient_rel = [p/a for p, a in zip(patient_vals, avg_vals)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[1]*5, theta=categories, fill='toself', name='Moyenne', line_color='green', opacity=0.3))
    fig.add_trace(go.Scatterpolar(r=patient_rel, theta=categories, fill='toself', name='Patient', line_color='red' if is_cancer else 'blue', opacity=0.7))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, max(max(patient_rel), 2)])), height=450, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("3. Cerveau du Modèle")
coefs = model.coef_[0]
importance_df = pd.DataFrame({'Variable': feature_names, 'Poids': coefs})
importance_df['Abs_Poids'] = importance_df['Poids'].abs()
importance_df = importance_df.sort_values(by='Abs_Poids', ascending=False).head(10)
fig_bar = px.bar(importance_df, x='Poids', y='Variable', orientation='h', color='Poids', color_continuous_scale='RdBu_r', text_auto='.2f')
fig_bar.update_layout(height=400)
st.plotly_chart(fig_bar, use_container_width=True)