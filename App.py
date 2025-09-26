# Imports
import streamlit as st
import pandas as pd
import plotly.express as px
from preprocessing_ordinal import apply_ordinal_encoding
from ml_forest import train_random_forest
from ml_tree import train_decision_tree
from ml_lightgbm import run_lightgbm_classification
from ml_xgboost import run_xgboost_classification
from ml_logreg import run_logistic_regression_classification
from ml_neuralnetwork import run_neural_network_classification
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from preprocessing_ordinal_SMOTE_PSM import apply_smote
from preprocessing_ordinal_SMOTE_PSM import propensity_score_matching
from shap_analysis import run_shap_analysis
import plotly.graph_objects as go
from causal_econML_causalforest import run_causal_forest_analysis
from hr_management import show_hr_dashboard

# ---------------------------------------------------------
# 1. Generelle Einstellungen
# ---------------------------------------------------------
st.set_page_config(page_title="KPI Dashboard", layout="wide")

# Initialisiert Session State
if "filtered_data" not in st.session_state:
    st.session_state["filtered_data"] = None
if "uploaded_data" not in st.session_state:
    st.session_state["uploaded_data"] = None
if "predictions" not in st.session_state:
    st.session_state["predictions"] = None
if "models_trained" not in st.session_state:
    st.session_state["models_trained"] = False
if "training_complete" not in st.session_state:
    st.session_state["training_complete"] = False

target_col = "Aktiv"

# ---------------------------------------------------------
# 2. Sidebar-Konfiguration
# ---------------------------------------------------------
st.sidebar.title('Sidebar')
upload_file = st.sidebar.file_uploader('Hochladen einer Datei mit Mitarbeiterdaten', type=["csv"])
st.sidebar.title('Navigation')
page = st.sidebar.selectbox('Wählen Sie aus, was Sie anzeigen möchten:', ['Analyse', 'Vorhersage','Management-Dashboard'])

# ---------------------------------------------------------
# 3. Hilfsfunktionen
# ---------------------------------------------------------

def initialize_models(df, target_col="Aktiv"):
    """
    Trainiert alle ML-Modelle einmal OHNE Streamlit-Ausgaben zu erzeugen
    und speichert die Ergebnisse im Session State
    """
    if df is None:
        return

    # Vorverarbeitung
    df_encoded_tree = apply_ordinal_encoding(df)
    df_encoded_nn = apply_ordinal_encoding(df)

    # -------------------- Decision Tree --------------------
    if "Decision Tree" not in st.session_state:
        st.session_state["Decision Tree"] = train_decision_tree(df_encoded_tree)

    # -------------------- Random Forest --------------------
    if "Random Forest" not in st.session_state:
        st.session_state["Random Forest"] = train_random_forest(df_encoded_tree)

    # -------------------- Logistic Regression --------------------
    if "Logistic Regression" not in st.session_state:
        st.session_state["Logistic Regression"] = run_logistic_regression_classification(df_encoded_tree)

    # -------------------- XGBoost --------------------
    if "XGBoost" not in st.session_state:
        st.session_state["XGBoost"] = run_xgboost_classification(df_encoded_tree)

    # -------------------- LightGBM --------------------
    if "LightGBM" not in st.session_state:
        st.session_state["LightGBM"] = run_lightgbm_classification(df_encoded_tree)

    # -------------------- Neural Network --------------------
    if "Neuronales Netzwerk" not in st.session_state:
        st.session_state["Neuronales Netzwerk"] = run_neural_network_classification(df_encoded_nn)

    # Markiere Training als abgeschlossen
    st.session_state["models_trained"] = True
    st.session_state["training_complete"] = True

def show_training_status():
    """
    Zeigt den Status des Modelltrainings an
    """
    if st.session_state.get("models_trained", False):
        st.sidebar.success("✅ Alle Modelle trainiert")
    elif st.session_state.get("uploaded_data") is not None:
        st.sidebar.info("🔄 Modelle werden im Hintergrund trainiert...")
    else:
        st.sidebar.info("📁 Warten auf Datei-Upload")

# Confusion Matrix mit Erklärung
def plot_confusion_matrix_interactive(cm, df_test, target_col="Aktiv"):
    labels = ["Inaktiv", "Aktiv"]
    explanations = [
        ["Richtig Inaktiv: Mitarbeiter ist inaktiv und wurde korrekt als inaktiv vorhergesagt.",
         "Falsch Positiv: Mitarbeiter ist inaktiv, wurde aber als aktiv vorhergesagt."],
        ["Falsch Negativ: Mitarbeiter ist aktiv, wurde aber als inaktiv vorhergesagt → potentiell Austrittsgefährdet.",
         "Richtig Aktiv: Mitarbeiter ist aktiv und wurde korrekt als aktiv vorhergesagt."]
    ]
    fig = go.Figure()
    for i in range(2):
        for j in range(2):
            fig.add_trace(go.Scatter(
                x=[j],
                y=[-i],
                mode="markers+text",
                marker=dict(
                    size=150,
                    color=(
                        "#ff6961" if (i == 1 and j == 0)        # False Negative: rot
                        else "#90ee90" if i == j                # True Positive/Negative: grün
                        else "#d3d3d3"                          # Rest: grau
                    )
                ),
                text=[f"{cm[i][j]}"],
                textposition="middle center",
                textfont=dict(size=24),
                hovertext=explanations[i][j],
                hoverinfo="text",
                name=f"{labels[i]} → {labels[j]}",
                customdata=[[i, j]],
                hoverlabel=dict(bgcolor="white", font_size=20)
            ))
    fig.update_layout(
        xaxis=dict(tickvals=[0, 1], ticktext=["Vorhergesagt Inaktiv", "Vorhergesagt Aktiv"], range=[-0.5, 1.5]),
        yaxis=dict(tickvals=[-0, -1], ticktext=["Tatsächlich Inaktiv", "Tatsächlich Aktiv"], range=[-1.5, 0.5]),
        title="Interaktive Confusion Matrix mit Erklärungen",
        height=500,
        showlegend=False
    )
    st.plotly_chart(fig)
    
    # Auswahl ermöglichen
    st.markdown("#### Details je Kategorie anzeigen")
    selected_option = st.selectbox("Kategorie auswählen:", [
        "Richtig Inaktiv", "Falsch Positiv", "Falsch Negativ", "Richtig Aktiv"])
    if selected_option == "Richtig Aktiv":
        st.dataframe(df_test[(df_test[target_col] == 1) & (df_test["Vorhersage"] == 1)])
    elif selected_option == "Falsch Positiv":
        st.dataframe(df_test[(df_test[target_col] == 0) & (df_test["Vorhersage"] == 1)])
    elif selected_option == "Falsch Negativ":
        st.dataframe(df_test[(df_test[target_col] == 1) & (df_test["Vorhersage"] == 0)])
    elif selected_option == "Richtig Inaktiv":
        st.dataframe(df_test[(df_test[target_col] == 0) & (df_test["Vorhersage"] == 0)])

def show_model_results(model_name, results, df, target_col="Aktiv"):
    """
    Einheitliche Funktion zur Anzeige der Modellergebnisse
    """
    st.info(f"{model_name} – Klassifikation mit Modellbewertung")
    
    # Metriken anzeigen - vereinheitlicht für alle Modelle
    if 'metrics' in results:
        metrics = results['metrics']
    else:
        # Fallback für Modelle mit direkten Metriken (wie Random Forest)
        metrics = {
            'accuracy': results.get('accuracy', 0),
            'precision': results.get('precision', 0),
            'recall': results.get('recall', 0),
            'f1': results.get('f1', 0)
        }
    
    st.markdown(f"**Genauigkeit:** {metrics['accuracy']:.2f}")
    st.markdown(f"**Präzision:** {metrics['precision']:.2f}")
    st.markdown(f"**Recall:** {metrics['recall']:.2f}")
    st.markdown(f"**F1-Score:** {metrics['f1']:.2f}")
    
    # Testdaten des letzten Folds 
    if 'last_test_idx' in results:
        df_last_test = df.iloc[results["last_test_idx"]].copy()
    else:
        df_last_test = df.loc[results["last_test_idx"]].copy()
    
    # Vorhersagen hinzufügen 
    if 'last_test_pred' in results:
        df_last_test["Vorhersage"] = results["last_test_pred"]
    elif 'y_pred' in results:
        df_last_test["Vorhersage"] = results["y_pred"]
    
    # Neuen Index hinzufügen
    df_last_test.insert(0, "Neuer_Index", range(len(df_last_test)))
    
    st.markdown("### Testdaten des letzten Folds")
    st.dataframe(df_last_test)
    
    # Confusion Matrix 
    st.markdown("### Confusion Matrix mit Erklärungen")
    
    if 'conf_matrix' in results:
        cm = results["conf_matrix"]
    elif 'conf_matrix_df' in results:
        cm = results["conf_matrix_df"].values
    else:
        # Confusion Matrix aus Vorhersagen berechnen
        if 'last_test_pred' in results and 'last_test_true' in results:
            cm = confusion_matrix(results["last_test_true"], results["last_test_pred"])
        else:
            cm = confusion_matrix(df_last_test[target_col], df_last_test["Vorhersage"])
    
    plot_confusion_matrix_interactive(cm, df_last_test, target_col)
    
    # SHAP Analyse für alle Modelle (wenn verfügbar)
    if model_name != "Neuronales Netzwerk":  # Neural Networks können problematisch für SHAP sein
        with st.expander("SHAP Analyse anzeigen", expanded=False):
            try:
                # Model-Objekt finden - verschiedene Schlüssel möglich
                model_obj = None
                if 'model' in results:
                    model_obj = results["model"]
                elif 'clf' in results:
                    model_obj = results["clf"]
                
                if model_obj and 'X_test' in results and 'X_train' in results:
                    run_shap_analysis(
                        model=model_obj,
                        X_test=results["X_test"],
                        X_train=results["X_train"],
                        model_name=model_name
                    )
                else:
                    st.warning("SHAP-Analyse nicht verfügbar - benötigte Daten fehlen")
            except Exception as e:
                st.warning(f"SHAP-Analyse für {model_name} nicht möglich: {str(e)}")

# ---------------------------------------------------------
# 4. Datei einlesen und automatisches Training
# ---------------------------------------------------------

if upload_file is not None:
    if st.session_state["uploaded_data"] is None:
        try:
            df = pd.read_csv(upload_file, sep=";", encoding="ansi")
            st.session_state["uploaded_data"] = df
            
            # Direkt nach Upload: Training im Hintergrund starten
            # OHNE UI-Ausgaben zu erzeugen
            initialize_models(df, target_col=target_col)
            
        except Exception as e:
            st.error(f"Fehler beim Laden der Datei: {e}")
else:
    if "uploaded_data" not in st.session_state:
        st.session_state["uploaded_data"] = None

# Training-Status in der Sidebar anzeigen
show_training_status()

# ---------------------------------------------------------
# 5. Korrelationsmatrix mit Plotly
# ---------------------------------------------------------
def plot_ordinal_correlation_matrix(df, title="Korrelationsmatrix"):
    # Wendet die ordinale Codierung an (nur definierte Spalten + Zielvariable)
    df_encoded = apply_ordinal_encoding(df)
    # Nur die Spalten ohne "_vorhanden" + "Aktiv"
    cols_to_plot = [col for col in df_encoded.columns if not col.endswith("_vorhanden")]
    # Korrelation berechnen
    corr = df_encoded[cols_to_plot].corr(numeric_only=True)
    # Heatmap mit fester Skala
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title=title,
    )
    fig.update_layout(
        width=700,
        height=700,
        xaxis_title="",
        yaxis_title="",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        autosize=True,
        margin=dict(l=120, r=40, t=100, b=120), 
    )

    # automatische Achsenmargins aktivieren
    fig.update_xaxes(automargin=True)
    return fig

# ---------------------------------------------------------
# 6. Seiten-Funktionen
# ---------------------------------------------------------

def home(uploaded_file):
    if uploaded_file is not None:
        try:
            # Datei wird geladen
            file_ext = uploaded_file.name.split(".")[-1].lower()
            if file_ext == "csv":  # Überprüft ob es sich um eine CSV-Datei handelt
                new_data = pd.read_csv(uploaded_file, sep=None, engine='python')  # erkennt Trennzeichen automatisch
            else:
                st.error("Ungültiger Dateityp. Bitte eine CSV-Datei hochladen.")
                return
            # Zielvariable definieren
            if target_col in new_data.columns:
                X = new_data.drop(columns=[target_col])
                y = new_data[target_col]
            st.session_state["uploaded_data"] = new_data
        except Exception as e:
            st.error(f"Fehler beim Laden der Datei: {e}")
 
# ---------------------------------------------------------
# 6a. Analyse-Funktionen
# ---------------------------------------------------------
def interactive_plot():
    df = st.session_state["uploaded_data"]
    if df is None:
        st.warning("Bitte lade zuerst eine CSV-Datei hoch.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    st.subheader("Übersicht über die Daten")
    st.dataframe(df)
    st.subheader("statistische Beschreibung der Daten")
    df_encoded = apply_ordinal_encoding(df)
    st.write(df_encoded.describe())
    if df is not None:
        st.subheader("Visualisierungen")
        col1, col2 = st.columns(2)
        # Diagramm 1 – fest vorgegeben (z. B. Bereich und Aktiv)
        with col1:
            st.markdown("### Vordefinierte Visualisierung: Bereich vs. Aktiv")
            st.markdown("&nbsp;")
            st.markdown("&nbsp;")
            st.markdown("&nbsp;")
            st.markdown("&nbsp;")
            if 'Bereich' in df.columns and 'Aktiv' in df.columns:
                grouped_static = df.groupby(['Bereich', 'Aktiv']).size().reset_index(name="Anzahl")
                grouped_static['Aktiv'] = grouped_static['Aktiv'].astype(str)
                fig_static = px.bar(
                    grouped_static,
                    x='Bereich',
                    y='Anzahl',
                    color='Aktiv',
                    barmode='stack',
                    color_discrete_map={"0": "lightgray", "1": "steelblue"},
                    title="Aktiv nach Bereich"
                )
                st.plotly_chart(fig_static, use_container_width=True)
            else:
                st.warning("Die Spalten 'Bereich' und/oder 'Aktiv' sind im Datensatz nicht vorhanden.")
        # Diagramm 2 – vom Benutzer auswählbar
        with col2:
            st.markdown("### Benutzerdefinierte Visualisierung")
            x_axis_val = st.selectbox('X-Achse auswählen', options=df.columns, key="xaxis")
            y_axis_val = st.selectbox(
                'Y-Achse (binär) auswählen',
                options=df.columns,
                index=list(df.columns).index("Aktiv") if "Aktiv" in df.columns else 0,
                key="yaxis"
    )
            grouped_dynamic = df.groupby([x_axis_val, y_axis_val]).size().reset_index(name="Anzahl")
            grouped_dynamic[y_axis_val] = grouped_dynamic[y_axis_val].astype(str)
            fig_dynamic = px.bar(
                grouped_dynamic,
                x=x_axis_val,
                y="Anzahl",
                color=y_axis_val,
                barmode="stack",
                color_discrete_map={"0": "lightgray", "1": "steelblue"},
                title=f"Verteilung von {y_axis_val} nach {x_axis_val}"
            )
            st.plotly_chart(fig_dynamic, use_container_width=True)
        # Diagramm 3 – Korrelationsmatrix
        st.markdown("---")  # Trennlinie
        st.subheader("Korrelationsmatrix")
        fig = plot_ordinal_correlation_matrix(df)
        st.plotly_chart(fig, use_container_width=True, height=800) 
    else:
        st.warning("Bitte lade zuerst eine CSV-Datei hoch.")
        # Diagramm 4 - Scatterplot
    st.markdown("---")
    st.subheader("Scatterplot zweier Merkmale")

    df_encoded = apply_ordinal_encoding(df)
    numeric_cols = [col for col in df_encoded.columns if 
                    pd.api.types.is_numeric_dtype(df_encoded[col]) and not col.endswith("_vorhanden")]

    if len(numeric_cols) < 2:
        st.warning("Nicht genug numerische Merkmale für einen Scatterplot.")
        return

    x_feature = st.selectbox("Wähle X-Achse", numeric_cols, index=0)
    y_feature = st.selectbox("Wähle Y-Achse", numeric_cols, index=1)

    df_plot = df_encoded[[x_feature, y_feature]].dropna()

    # Scatterplot mit optionaler Farbe nach Aktiv
    # Anzahl der Mitarbeiter pro Kombination zählen
    df_count = df_encoded.groupby([x_feature, y_feature]).size().reset_index(name='Anzahl')

    # Optional: Aktiv als Farbe darstellen
    if "Aktiv" in df_encoded.columns:
        # Mittelwert der Aktiv-Variable pro Kombination
        df_count["Aktiv"] = df_encoded.groupby([x_feature, y_feature])["Aktiv"].mean().values

    fig = px.scatter(
        df_count,
        x=x_feature,
        y=y_feature,
        size='Anzahl',          # Punktgröße nach Anzahl der Mitarbeiter
        color='Aktiv' if "Aktiv" in df_count.columns else None,
        color_continuous_scale="Blues",
        hover_data=['Anzahl'],  # Zeigt die Anzahl beim Hovern
        title=f"Scatterplot: {x_feature} vs. {y_feature} (Größe = Anzahl Mitarbeiter)"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 6b. Vorhersage-Funktionen
# ---------------------------------------------------------
def prediction_model():
    if st.session_state["uploaded_data"] is None:
        st.warning("Bitte laden Sie zunächst eine Datei hoch.")
        return
    
    # Prüfen ob Modelle fertig trainiert sind
    if not st.session_state.get("models_trained", False):
        st.info("🔄 Modelle werden noch trainiert. Bitte warten Sie einen Moment...")
        st.stop()
    
    df = st.session_state["uploaded_data"]
    # Dropdown zur Auswahl des Modells
    model_choice = st.selectbox("Wähle einen ML-Algorithmus", ["Decision Tree", "Random Forest", "Logistic Regression","XGBoost","LightGBM","Neuronales Netzwerk","Szenario Simulation"])
    st.markdown(f"## Modell: {model_choice}")
    # Ergebnisse je nach Modell anzeigen    
    if model_choice in ["Decision Tree", "Random Forest", "Logistic Regression", "XGBoost", "LightGBM"]:
        results = st.session_state[model_choice]
        show_model_results(model_choice, results, df, target_col)
    
    # Spezialbehandlung für Neural Network (falls SHAP problematisch ist)
    elif model_choice == "Neuronales Netzwerk":
        results = st.session_state["Neuronales Netzwerk"]
        st.info("Neuronales Netzwerk – Klassifikation mit Modellbewertung")
        
        # Metriken anzeigen
        if 'metrics' in results:
            st.markdown(f"**Genauigkeit:** {results['metrics']['accuracy']:.2f}")
            st.markdown(f"**Präzision:** {results['metrics']['precision']:.2f}")
            st.markdown(f"**Recall:** {results['metrics']['recall']:.2f}")
            st.markdown(f"**F1-Score:** {results['metrics']['f1']:.2f}")
        
        # Testdaten anzeigen
        df_last_test = df.loc[results["last_test_idx"]].copy()
        df_last_test["Vorhersage"] = results["last_test_pred"]
        st.dataframe(df_last_test)
        
        # Confusion Matrix
        st.markdown("### Confusion Matrix mit Erklärungen")
        if 'conf_matrix_df' in results:
            cm = results["conf_matrix_df"].values
        else:
            cm = confusion_matrix(df_last_test[target_col], df_last_test["Vorhersage"])
        
        plot_confusion_matrix_interactive(cm, df_last_test, target_col)
        
        # Hinweis zu SHAP für Neural Networks
        st.info("SHAP-Analyse für Neuronale Netzwerke erfordert spezielle Implementierung und wurde hier ausgelassen.")

    elif model_choice == "Szenario Simulation":
        # Preprocessing
        df_encoded = apply_ordinal_encoding(df)
        
        # Treatment-Schwellenwert
        treatment_threshold = st.slider(
            "Schwellenwert für hohe Gehaltserhöhung",
            min_value=0, max_value=40, value=20, step=2
        )
        
        # Treatment definieren
        df_encoded["Gehaltserhöhung"] = (df_encoded["Gehaltsentwicklung"] >= treatment_threshold).astype(int)
        
        # Feature-Spalten für PSM
        feature_cols_psm = [col for col in df_encoded.columns if col not in ["Gehaltserhöhung", "Gehaltsentwicklung", "Aktiv"]]
        
        # PSM für Treatment vs Control
        df_matched = propensity_score_matching(
            df_encoded, 
            treatment_col="Gehaltserhöhung",
            outcome_col="Aktiv", 
            feature_cols=feature_cols_psm
        )
        
        st.write(f"Gesamtdaten: {len(df_encoded)}")
        st.write(f"Daten nach PSM: {len(df_matched)}")
        st.write(f"Treatment-Balance: {df_matched['Gehaltserhöhung'].mean():.2%}")
        
        # Kausalanalyse
        run_causal_forest_analysis(df_matched)

# ---------------------------------------------------------
# 7. Management Dashboard
# ---------------------------------------------------------
def management_dashboard():
    if st.session_state["uploaded_data"] is None:
        st.warning("Bitte laden Sie zunächst eine Datei hoch.")
        return
    
    # Prüfen ob Modelle fertig trainiert sind
    if not st.session_state.get("models_trained", False):
        st.info("🔄 Modelle werden noch trainiert. Bitte warten Sie einen Moment...")
        st.stop()
    
    df = st.session_state["uploaded_data"]
    df_encoded = apply_ordinal_encoding(df)
    
    # LightGBM Modell ist bereits trainiert - aus Session State laden
    model_output = st.session_state["LightGBM"]
    
    # Speichere die Vorhersagen im session_state für andere Seiten
    st.session_state["prediction_results"] = {
        "y_test": model_output["y_test"],
        "y_pred": model_output["y_pred"],
    }
    show_hr_dashboard(model_output)
    # Letzten Testfold anzeigen
    df_last_test = df.iloc[model_output["last_test_idx"]].copy()
    df_last_test["Vorhersage"] = model_output["last_test_pred"]
    df_last_test.insert(0, "Neuer_Index", range(len(df_last_test)))
    # Konfusionsmatrix visualisieren
    st.markdown("### Confusion Matrix mit Erklärungen")
    plot_confusion_matrix_interactive(model_output["conf_matrix_df"].values, df_last_test)

# ---------------------------------------------------------
# 8. Navigation über Sidebar
# ---------------------------------------------------------
# Navigation über Sidebar
if page == 'Analyse':
    interactive_plot()
elif page == "Vorhersage":
    prediction_model()
elif page == "Management-Dashboard":
    management_dashboard()