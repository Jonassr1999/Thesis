import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import plotly.express as px
from sklearn.inspection import partial_dependence
import plotly.graph_objects as go
import plotly.io as pio
import os
def show_hr_dashboard(model_output):
    st.title("HR Management Dashboard")

    # --- Modell-Ergebnisse laden ---
    y_test = model_output["y_test"]
    y_pred = model_output["y_pred"]
    y_pred_proba = model_output.get("y_pred_proba")  # Wahrscheinlichkeiten
    model = model_output["model"]
    X_test = model_output["X_test"]

    # --- Schwellenwert für Risiko-Definition ---
    risk_threshold = 0.7  # kann bei Bedarf angepasst werden
    cost_per_employee = 50000  # Kosten pro gefährdetem Mitarbeiter in Euro

    # Achtung: 0 = ausgetreten → Fluktuation = Anteil 0
    actual_rate = (y_test == 0).mean() * 100
    predicted_rate = (y_pred == 0).mean() * 100

    # --- Anzahl gefährdeter Mitarbeiter ---
    at_risk_count = 0
    if y_pred_proba is not None:
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
            # Klasse 0 = Austritt → Wahrscheinlichkeit für Austritt >= Schwelle
            at_risk_mask = (y_pred_proba[:, 0] >= risk_threshold) & (y_test == 1)
            at_risk_count = np.sum(at_risk_mask)
        else:
            # Ein-dimensionales Wahrscheinlichkeitsarray (für Klasse 0 angenommen)
            at_risk_mask = (y_pred_proba >= risk_threshold) & (y_test == 1)
            at_risk_count = np.sum(at_risk_mask)
    else:
        # Wenn keine Wahrscheinlichkeiten vorliegen: Nutze binäre Vorhersage (0 = Austritt)
        at_risk_mask = (y_pred == 0) & (y_test == 1)
        at_risk_count = np.sum(at_risk_mask)

    # --- Berechnung der potentiellen Kosten ---
    total_potential_cost = at_risk_count * cost_per_employee

    # --- Hauptkennzahlen in schönen Boxen ---
    st.markdown("""
    <div style='background-color: #4682b4; 
                padding: 8px; border-radius: 5px; margin: 10px 0;'>
        <h5 style='color: white; text-align: center; margin: 0; font-weight: 400; opacity: 0.9;'>
            Fluktuation & Risiko-Kennzahlen
        </h5>
    </div>
    """, unsafe_allow_html=True)

    kpi_col1, kpi_col2 = st.columns(2)

    with kpi_col1:
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; 
                    border-left: 4px solid #d62728; text-align: center;'>
            <h4 style='color: #d62728; margin-bottom: 5px;'>Tatsächliche Fluktuation</h4>
            <h2 style='color: #2c3e50; margin: 0; font-size: 36px;'>{actual_rate:.1f}%</h2>
            <p style='color: #7f8c8d; font-size: 12px; margin: 5px 0 0 0;'>
                Prozentsatz inaktiver Mitarbeiter
            </p>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; 
                    border-left: 4px solid #1f77b4; text-align: center;'>
            <h4 style='color: #1f77b4; margin-bottom: 5px;'>Vorhergesagte Fluktuation</h4>
            <h2 style='color: #2c3e50; margin: 0; font-size: 36px;'>{predicted_rate:.1f}%</h2>
            <p style='color: #7f8c8d; font-size: 12px; margin: 5px 0 0 0;'>
                KI-basierte Prognose
            </p>
        </div>
        """, unsafe_allow_html=True)

    # --- Zusätzliche Kosten-Details in einer schönen Box ---
    st.markdown("---")
    
    # Kosten-KPI Box
    st.markdown("""
    <div style='background-color: #4682b4; 
                padding: 8px; border-radius: 5px; margin: 10px 0;'>
        <h5 style='color: white; text-align: center; margin: 0; font-weight: 400; opacity: 0.9;'>
            Finanzielle Auswirkungen der Fluktuation
        </h5>
    </div>
    """, unsafe_allow_html=True)
    
    cost_col1, cost_col2, cost_col3, cost_col4 = st.columns(4)

    with cost_col1:
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; 
                    border-left: 4px solid #ff7f0e; text-align: center;'>
            <h4 style='color: #ff7f0e; margin-bottom: 5px;'>Gefährdete Mitarbeiter</h4>
            <h2 style='color: #2c3e50; margin: 0;'>{at_risk_count}</h2>
            <p style='color: #7f8c8d; font-size: 12px; margin: 5px 0 0 0;'>
                Hohe Austrittswahrscheinlichkeit
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with cost_col2:
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; 
                    border-left: 4px solid #e74c3c; text-align: center;'>
            <h4 style='color: #e74c3c; margin-bottom: 5px;'>Kosten pro MA</h4>
            <h2 style='color: #2c3e50; margin: 0;'>50.000 €</h2>
            <p style='color: #7f8c8d; font-size: 12px; margin: 5px 0 0 0;'>
                Rekrutierung, Einarbeitung, Produktivitätsverlust
            </p>
        </div>
        """, unsafe_allow_html=True)

    with cost_col3:
        formatted_total = f"{total_potential_cost:,.0f}".replace(",", ".")
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; 
                    border-left: 4px solid #f39c12; text-align: center;'>
            <h4 style='color: #f39c12; margin-bottom: 5px;'>Potentielle Kosten</h4>
            <h2 style='color: #2c3e50; margin: 0;'>{formatted_total} €</h2>
            <p style='color: #7f8c8d; font-size: 12px; margin: 5px 0 0 0;'>
                Gesamtrisiko aller gefährdeten MA
            </p>
        </div>
        """, unsafe_allow_html=True)

    with cost_col4:
        savings_potential = total_potential_cost * 0.2  # Annahme: 20% der Kosten vermeidbar
        formatted_savings = f"{savings_potential:,.0f}".replace(",", ".")
        st.markdown(f"""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; 
                    border-left: 4px solid #27ae60; text-align: center;'>
            <h4 style='color: #27ae60; margin-bottom: 5px;'>Einsparpotential</h4>
            <h2 style='color: #2c3e50; margin: 0;'>{formatted_savings} €</h2>
            <p style='color: #7f8c8d; font-size: 12px; margin: 5px 0 0 0;'>
                Durch präventive Maßnahmen - Annahme 20% der Kosten vermeidbar
            </p>
        </div>
        """, unsafe_allow_html=True)

    # --- SHAP Summary Plot ---
    st.markdown("---")
    st.subheader("Wichtige Einflussfaktoren (SHAP Summary Plot)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    fig_shap, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
    st.pyplot(fig_shap)

    # --- Partielle Abhängigkeitsdiagramme (PDPs) ---
    #st.markdown("---")

    top_n=5
    """
    Erstellt Partial Dependence Plots (PDPs) mit SHAP für die Top-n Features
    und speichert jeden Plot als einzelne PDF-Datei im aktuellen Arbeitsverzeichnis.
    """
    # Falls du den Speicherort ändern möchtest:
    save_path = os.getcwd()

    # SHAP-Werte berechnen
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Feature-Importance bestimmen (für Klasse 0 = Austritt)
    if isinstance(shap_values, list):  # Bei Klassifikationsmodellen mit mehreren Klassen
        shap_importance = np.abs(shap_values[0]).mean(axis=0)
    else:
        shap_importance = np.abs(shap_values).mean(axis=0)

    feature_names = X_test.columns
    top_indices = np.argsort(shap_importance)[-top_n:][::-1]
    top_features = feature_names[top_indices]

    # PDPs erzeugen und als PDFs speichern
    for feature in top_features:
        fig, ax = plt.subplots(figsize=(6, 4))
        shap.dependence_plot(feature, shap_values[0] if isinstance(shap_values, list) else shap_values,
                             X_test, ax=ax, show=False)
        plt.tight_layout()

        pdf_filename = os.path.join(save_path, f"PDP_{feature}.pdf")
        plt.savefig(pdf_filename)
        plt.close(fig)

    print(f"{len(top_features)} PDP-PDFs gespeichert in: {save_path}")

