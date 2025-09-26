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
import seaborn as sns
from causal_dowhy_analysis import run_causal_analysis
from preprocessing_ordinal_causalml import apply_ordinal_encoding_causal
from shap_analysis import run_shap_analysis
import plotly.graph_objects as go
from causal_econML_causalforest import run_causal_forest_analysis
from hr_management import show_hr_dashboard
import hashlib
import numpy as np
import shap
import pickle
import joblib
import json
import os
from datetime import datetime

# ---------------------------------------------------------
# 1. Generelle Einstellungen
# ---------------------------------------------------------
st.set_page_config(page_title="KPI Dashboard", layout="wide")
# Erweiterte Session State Initialisierung
def init_session_state():
    """Initialisiert alle notwendigen Session State Variablen mit Persistierung"""
    defaults = {
        "filtered_data": None,
        "uploaded_data": None,
        "predictions": None,
        "data_hash": None,
        "models_trained": False,
        "model_results": {},
        "all_train_losses": {},
        "all_val_losses": {},
        "training_status": {},
        "use_model_persistence": True,
        "model_cache_info": {},
        "prediction_results": None  
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialisiere Session State
init_session_state()
target_col = "Aktiv"

# ---------------------------------------------------------
# Modell-Persistierung Funktionen
# ---------------------------------------------------------

def create_model_directory():
    """Erstellt das Verzeichnis für gespeicherte Modelle"""
    model_dir = "saved_models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir

def generate_model_filename(model_name, data_hash):
    """Generiert einen eindeutigen Dateinamen für das Modell"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{data_hash[:8]}_{timestamp}"

def save_model_results(model_name, model_results, data_hash):
    """Speichert Modellergebnisse auf der Festplatte"""
    try:
        model_dir = create_model_directory()
        filename = generate_model_filename(model_name, data_hash)
        
        # Speichere verschiedene Komponenten
        model_data = {
            'model_name': model_name,
            'data_hash': data_hash,
            'timestamp': datetime.now().isoformat(),
            'metrics': model_results.get('metrics', {}),
            'training_status': ' Erfolgreich'
        }
        
        # Speichere das eigentliche Modell (falls vorhanden)
        if 'model' in model_results:
            model_path = os.path.join(model_dir, f"{filename}_model.pkl")
            joblib.dump(model_results['model'], model_path)
            model_data['model_path'] = model_path
        
        # Speichere Loss-Kurven (falls vorhanden)
        if 'train_loss' in model_results and 'val_loss' in model_results:
            loss_data = {
                'train_loss': model_results['train_loss'],
                'val_loss': model_results['val_loss']
            }
            loss_path = os.path.join(model_dir, f"{filename}_losses.pkl")
            with open(loss_path, 'wb') as f:
                pickle.dump(loss_data, f)
            model_data['loss_path'] = loss_path
        
        # Speichere andere wichtige Daten
        other_data = {}
        for key in ['conf_matrix', 'conf_matrix_df', 'last_test_idx', 'last_test_pred', 
                   'feature_importance', 'X_test', 'y_test', 'y_pred']:
            if key in model_results:
                other_data[key] = model_results[key]
        
        if other_data:
            other_path = os.path.join(model_dir, f"{filename}_data.pkl")
            with open(other_path, 'wb') as f:
                pickle.dump(other_data, f)
            model_data['data_path'] = other_path
        
        # Speichere Metadaten als JSON
        metadata_path = os.path.join(model_dir, f"{filename}_metadata.json")
        with open(metadata_path, 'w') as f:
            # Konvertiere numpy arrays zu Listen für JSON-Serialisierung
            json_safe_data = {}
            for key, value in model_data.items():
                if isinstance(value, (list, dict, str, int, float, bool)) or value is None:
                    json_safe_data[key] = value
                else:
                    json_safe_data[key] = str(value)
            json.dump(json_safe_data, f, indent=2)
        
        model_data['metadata_path'] = metadata_path
        return model_data
        
    except Exception as e:
        st.error(f"Fehler beim Speichern von {model_name}: {e}")
        return None

def load_model_results(model_name, data_hash):
    """Lädt gespeicherte Modellergebnisse von der Festplatte"""
    try:
        model_dir = "saved_models"
        if not os.path.exists(model_dir):
            return None
        
        # Suche nach passenden Dateien
        matching_files = []
        for filename in os.listdir(model_dir):
            if filename.startswith(f"{model_name}_{data_hash[:8]}") and filename.endswith("_metadata.json"):
                matching_files.append(filename)
        
        if not matching_files:
            return None
        
        # Lade die neueste Datei
        latest_file = sorted(matching_files)[-1]
        metadata_path = os.path.join(model_dir, latest_file)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Rekonstruiere die Modellergebnisse
        model_results = {
            'metrics': metadata.get('metrics', {}),
            'training_status': metadata.get('training_status', ' Geladen')
        }
        
        # Lade das Modell
        if 'model_path' in metadata and os.path.exists(metadata['model_path']):
            model_results['model'] = joblib.load(metadata['model_path'])
        
        # Lade Loss-Kurven
        if 'loss_path' in metadata and os.path.exists(metadata['loss_path']):
            with open(metadata['loss_path'], 'rb') as f:
                loss_data = pickle.load(f)
                model_results.update(loss_data)
        
        # Lade andere Daten
        if 'data_path' in metadata and os.path.exists(metadata['data_path']):
            with open(metadata['data_path'], 'rb') as f:
                other_data = pickle.load(f)
                model_results.update(other_data)
        
        return model_results
        
    except Exception as e:
        st.error(f"Fehler beim Laden von {model_name}: {e}")
        return None

def get_saved_models_info():
    """Gibt Informationen über alle gespeicherten Modelle zurück"""
    model_dir = "saved_models"
    if not os.path.exists(model_dir):
        return []
    
    models_info = []
    for filename in os.listdir(model_dir):
        if filename.endswith("_metadata.json"):
            try:
                with open(os.path.join(model_dir, filename), 'r') as f:
                    metadata = json.load(f)
                    models_info.append({
                        'filename': filename,
                        'model_name': metadata.get('model_name', 'Unknown'),
                        'data_hash': metadata.get('data_hash', 'Unknown')[:8],
                        'timestamp': metadata.get('timestamp', 'Unknown'),
                        'metrics': metadata.get('metrics', {})
                    })
            except Exception as e:
                continue
    
    return sorted(models_info, key=lambda x: x.get('timestamp', ''), reverse=True)

def clean_old_models(keep_per_model=3):
    """Löscht alte Modellversionen, behält nur die neuesten"""
    model_dir = "saved_models"
    if not os.path.exists(model_dir):
        return
    
    models_info = get_saved_models_info()
    models_by_name = {}
    
    # Gruppiere nach Modellname
    for info in models_info:
        model_name = info['model_name']
        if model_name not in models_by_name:
            models_by_name[model_name] = []
        models_by_name[model_name].append(info)
    
    # Lösche alte Versionen
    for model_name, model_list in models_by_name.items():
        if len(model_list) > keep_per_model:
            to_delete = model_list[keep_per_model:]
            for model_info in to_delete:
                base_name = model_info['filename'].replace('_metadata.json', '')
                files_to_delete = [
                    f"{base_name}_metadata.json",
                    f"{base_name}_model.pkl",
                    f"{base_name}_losses.pkl",
                    f"{base_name}_data.pkl"
                ]
                
                for file_to_delete in files_to_delete:
                    file_path = os.path.join(model_dir, file_to_delete)
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            print(f"Fehler beim Löschen von {file_path}: {e}")

# ---------------------------------------------------------
# Erweiterte train_all_models Funktion
# ---------------------------------------------------------

def train_all_models_with_persistence(df):
    """Trainiert alle ML-Modelle mit automatischem Speichern und Laden"""
    
    data_hash = calculate_data_hash(df)
    
    # Prüfe, ob bereits trainierte Modelle für diese Daten existieren
    models_to_train = {
        "Decision Tree": train_decision_tree,
        "Random Forest": train_random_forest,
        "Logistic Regression": run_logistic_regression_classification,
        "XGBoost": run_xgboost_classification,
        "LightGBM": run_lightgbm_classification,
        "Neural Network": run_neural_network_classification
    }
    
    loaded_models = {}
    models_need_training = {}
    
    # Versuche alle Modelle zu laden
    st.info(" Suche nach gespeicherten Modellen...")
    for model_name in models_to_train.keys():
        loaded_result = load_model_results(model_name, data_hash)
        if loaded_result:
            loaded_models[model_name] = loaded_result
            st.success(f" {model_name} erfolgreich geladen")
        else:
            models_need_training[model_name] = models_to_train[model_name]
            st.info(f" {model_name} muss trainiert werden")
    
    # Trainiere nur die Modelle, die nicht geladen werden konnten
    if models_need_training:
        training_container = st.container()
        
        with training_container:
            st.info(f" Trainiere {len(models_need_training)} Modell(e)...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            df_encoded = apply_ordinal_encoding(df)
            
            total_models = len(models_need_training)
            
            for i, (model_name, model_func) in enumerate(models_need_training.items()):
                try:
                    status_text.text(f"Trainiere {model_name}... ({i+1}/{total_models})")
                    
                    # Trainiere das Modell (gleiche Logik wie vorher)
                    if model_name == "Decision Tree":
                        result = model_func(df_encoded)
                    elif model_name == "Random Forest":
                        result = model_func(df_encoded)
                        if result and isinstance(result, dict):
                            if 'accuracy' in result and 'precision' in result:
                                if 'metrics' not in result:
                                    result['metrics'] = {
                                        'accuracy': result.get('accuracy'),
                                        'precision': result.get('precision'),
                                        'recall': result.get('recall'),
                                        'f1': result.get('f1')
                                    }
                    elif model_name == "Logistic Regression":
                        result = model_func(df_encoded, target_col=target_col)
                    elif model_name == "XGBoost":
                        result = model_func(df_encoded, target_col=target_col, plot_avg_loss_curves=False)
                        if "train_loss" in result and "val_loss" in result:
                            st.session_state["all_train_losses"][model_name] = result["train_loss"]
                            st.session_state["all_val_losses"][model_name] = result["val_loss"]
                    elif model_name == "LightGBM":
                        result = model_func(df_encoded)
                    elif model_name == "Neural Network":
                        result = model_func(df_encoded, target_col=target_col)
                        if "train_loss" in result and "val_loss" in result:
                            st.session_state["all_train_losses"][model_name] = result["train_loss"]
                            st.session_state["all_val_losses"][model_name] = result["val_loss"]
                    
                    # Speichere das trainierte Modell
                    saved_info = save_model_results(model_name, result, data_hash)
                    if saved_info:
                        st.success(f" {model_name} gespeichert")      
                    loaded_models[model_name] = result
                    st.session_state["training_status"][model_name] = "Trainiert & Gespeichert"
                    
                except Exception as e:
                    st.session_state["training_status"][model_name] = f" Fehler: {str(e)[:50]}..."
                    st.error(f"Fehler beim Trainieren von {model_name}: {e}")
                    
                progress_bar.progress((i + 1) / total_models)
            
            status_text.success("Training und Speichern abgeschlossen!")
            
            import time
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            training_container.empty()
    
    else:
        st.success("Alle Modelle wurden erfolgreich aus dem Speicher geladen!")
    
    # Aktualisiere Session State mit allen Modellen (geladen + trainiert)
    st.session_state["model_results"] = loaded_models
    st.session_state["models_trained"] = True
    
    # Aktualisiere Loss-Kurven für Session State
    for model_name, result in loaded_models.items():
        if "train_loss" in result and "val_loss" in result:
            st.session_state["all_train_losses"][model_name] = result["train_loss"]
            st.session_state["all_val_losses"][model_name] = result["val_loss"]

# ---------------------------------------------------------
# Erweiterte Session State Initialisierung
# ---------------------------------------------------------

def init_session_state_extended():
    """Erweiterte Session State Initialisierung mit Persistierung"""
    defaults = {
        "filtered_data": None,
        "uploaded_data": None,
        "predictions": None,
        "data_hash": None,
        "models_trained": False,
        "model_results": {},
        "all_train_losses": {},
        "all_val_losses": {},
        "training_status": {},
        "use_model_persistence": True,  # Schalter für Persistierung
        "model_cache_info": {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
# ---------------------------------------------------------
# 2. Hilfsfunktionen
# ---------------------------------------------------------
def calculate_data_hash(df):
    """Berechnet einen Hash-Wert für die Daten um Änderungen zu erkennen"""
    return hashlib.md5(df.to_string().encode()).hexdigest()
def train_all_models(df):
    """Trainiert alle ML-Modelle automatisch und speichert die Ergebnisse"""
    
    # Container für das Training-UI
    training_container = st.container()
    
    with training_container:
        st.info("🔄 Trainiere alle ML-Modelle automatisch...")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        df_encoded = apply_ordinal_encoding(df)
        
        models = {
            "Decision Tree": train_decision_tree,
            "Random Forest": train_random_forest,
            "Logistic Regression": run_logistic_regression_classification,
            "XGBoost": run_xgboost_classification,
            "LightGBM": run_lightgbm_classification,
            "Neural Network": run_neural_network_classification
        }
        
        results = {}
        total_models = len(models)
        
        for i, (model_name, model_func) in enumerate(models.items()):
            try:
                status_text.text(f"Trainiere {model_name}... ({i+1}/{total_models})")
                
                if model_name == "Decision Tree":
                    result = model_func(df_encoded)
                    results[model_name] = result
                    
                elif model_name == "Random Forest":
                    result = model_func(df_encoded)
                    # Random Forest Ergebnis-Struktur anpassen
                    if result and isinstance(result, dict):
                        # Prüfe verschiedene mögliche Schlüssel für Metriken
                        if 'accuracy' in result and 'precision' in result:
                            # Metrics sind bereits im Hauptdict
                            if 'metrics' not in result:
                                result['metrics'] = {
                                    'accuracy': result.get('accuracy'),
                                    'precision': result.get('precision'),
                                    'recall': result.get('recall'),
                                    'f1': result.get('f1')
                                }
                        results[model_name] = result
                    
                elif model_name == "Logistic Regression":
                    result = model_func(df_encoded, target_col=target_col)
                    results[model_name] = result
                    
                elif model_name == "XGBoost":
                    result = model_func(df_encoded, target_col=target_col, plot_avg_loss_curves=False)
                    results[model_name] = result
                    # Speichere Loss-Kurven
                    if "train_loss" in result and "val_loss" in result:
                        st.session_state["all_train_losses"][model_name] = result["train_loss"]
                        st.session_state["all_val_losses"][model_name] = result["val_loss"]
                    
                elif model_name == "LightGBM":
                    result = model_func(df_encoded)
                    results[model_name] = result
                    
                elif model_name == "Neural Network":
                    result = model_func(df_encoded, target_col=target_col)
                    results[model_name] = result
                    # Speichere Loss-Kurven falls vorhanden
                    if "train_loss" in result and "val_loss" in result:
                        st.session_state["all_train_losses"][model_name] = result["train_loss"]
                        st.session_state["all_val_losses"][model_name] = result["val_loss"]
                
                st.session_state["training_status"][model_name] = "Erfolgreich"
                
            except Exception as e:
                st.session_state["training_status"][model_name] = f" Fehler: {str(e)[:50]}..."
                st.error(f"Fehler beim Trainieren von {model_name}: {e}")
                
            progress_bar.progress((i + 1) / total_models)
        
        status_text.success("Training abgeschlossen!")
        
        st.session_state["model_results"] = results
        st.session_state["models_trained"] = True
        
        # UI-Elemente nach kurzer Zeit ausblenden
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        training_container.empty()


def show_shap_plots_lightgbm(df, model_output):
    st.markdown("## SHAP Analyse (LightGBM)")

    model = model_output["model"]
    X_test = model_output["X_test"]

    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)

    # Versuch: Probability-Space
    try:
        explainer = shap.TreeExplainer(model, model_output="probability")
        shap_values = explainer.shap_values(X_test, check_additivity=False)
        base_value = explainer.expected_value
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1]
        elif isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]
        use_probability = True
    except Exception:
        # Fallback: Logit-Space → manuell Sigmoid
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test, check_additivity=False)
        base_value = explainer.expected_value
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1]
        elif isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]
        use_probability = False

    # --- Beeswarm ---
    st.markdown("### Beeswarm Plot (globale Feature-Bedeutung)")
    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=X_test,
        feature_names=X_test.columns,
    )
    fig_bee, _ = plt.subplots()
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    st.pyplot(fig_bee)

    # --- Waterfall ---
    st.markdown("### Waterfall Plot (lokale Erklärung eines Beispiels)")
    idx = st.number_input("Index eines Testdatensatzes auswählen",
                          min_value=0, max_value=len(X_test)-1, value=0, step=1)
    single_expl = explanation[idx]
    fig_wat, _ = plt.subplots()
    shap.plots.waterfall(single_expl, max_display=15, show=False)
    st.pyplot(fig_wat)

    # --- Konsistenzcheck ---
    with st.expander("Diagnose: Konsistenzcheck"):
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X_test.iloc[[idx]])[0, 1]
        else:
            p = float(model.predict(X_test.iloc[[idx]])[0])
        st.write(f"Modellvorhersage P(y=1|x): {p:.4f}")

        fx = single_expl.base_values + single_expl.values.sum()
        if use_probability:
            st.write(f"Waterfall f(x): {fx:.4f}  (sollte ≈ Vorhersage sein)")
        else:
            fx_prob = 1 / (1 + np.exp(-fx))
            st.write(f"Waterfall (logit) → Sigmoid: {fx_prob:.4f}  (sollte ≈ Vorhersage sein)")

def plot_roc_curve_simple(model_name, model_result):
    """Erstellt eine einfache ROC-Kurve für ein Modell"""
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    import plotly.graph_objects as go
    
    try:
        model = model_result["model"]
        
        # Falls Cross-Validation Daten verfügbar sind, simuliere 5-Fold ROC
        if st.session_state["uploaded_data"] is not None:
            df_encoded = apply_ordinal_encoding(st.session_state["uploaded_data"])
            X = df_encoded.drop(columns=["Aktiv"])
            y = df_encoded["Aktiv"]
            
            # 5-Fold Cross Validation für ROC
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            fig = go.Figure()
            aucs = []
            
            fold = 0
            for train_idx, test_idx in skf.split(X, y):
                fold += 1
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Vorhersage-Wahrscheinlichkeiten
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, "decision_function"):
                    y_proba = model.decision_function(X_test)
                else:
                    continue
                
                # ROC-Kurve berechnen
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                
                # Füge ROC-Kurve zum Plot hinzu
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'Fold {fold} (AUC = {roc_auc:.3f})',
                    line=dict(width=2)
                ))
            
            # Diagonale (Zufallslinie) hinzufügen
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Zufall (AUC = 0.500)',
                line=dict(dash='dash', color='gray')
            ))
            
            fig.update_layout(
                title=f'ROC Kurve - {model_name} (Durchschnitt AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f})',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                width=600,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Fallback: Verwende nur die Test-Daten aus dem Modell
            X_test = model_result["X_test"]
            y_test = model_result["y_test"]
            
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_proba = model.decision_function(X_test)
            else:
                st.info(f"Modell {model_name} unterstützt keine Wahrscheinlichkeits-Vorhersagen für ROC.")
                return
            
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                   name=f'ROC (AUC = {roc_auc:.3f})', line=dict(width=3)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                   name='Zufall (AUC = 0.500)', line=dict(dash='dash', color='gray')))
            
            fig.update_layout(
                title=f'ROC Kurve - {model_name} (AUC = {roc_auc:.3f})',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                width=600,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Fehler bei ROC-Kurve für {model_name}: {str(e)}")

def get_top_features(model_result, top_n=3):
    """Extrahiert die wichtigsten Features aus einem Modell"""
    try:
        # Prüfe verschiedene Möglichkeiten, Feature Importance zu finden
        feature_importance = None
        feature_names = None
        
        # 1. Direkt in model_result gespeichert
        if 'feature_importance' in model_result:
            feature_importance = model_result['feature_importance']
            if 'feature_names' in model_result:
                feature_names = model_result['feature_names']
        
        # 2. Aus dem Modell selbst extrahieren
        elif 'model' in model_result:
            model = model_result['model']
            
            # Für Tree-basierte Modelle (Random Forest, XGBoost, LightGBM)
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                if 'X_test' in model_result:
                    feature_names = model_result['X_test'].columns.tolist()
            
            # Für LightGBM spezifisch
            elif hasattr(model, 'feature_importance'):
                feature_importance = model.feature_importance()
                if 'X_test' in model_result:
                    feature_names = model_result['X_test'].columns.tolist()
            
            # Für Logistic Regression
            elif hasattr(model, 'coef_'):
                # Verwende absolute Werte der Koeffizienten
                feature_importance = abs(model.coef_[0]) if len(model.coef_.shape) > 1 else abs(model.coef_)
                if 'X_test' in model_result:
                    feature_names = model_result['X_test'].columns.tolist()
        
        # Wenn Feature Importance gefunden wurde
        if feature_importance is not None:
            # Wenn keine Feature-Namen vorhanden, verwende Indizes
            if feature_names is None:
                feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
            
            # Erstelle DataFrame und sortiere nach Importance
            feature_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            # Top N Features zurückgeben
            top_features = feature_df.head(top_n)['feature'].tolist()
            return ', '.join(top_features)
        
        return "N/A"
        
    except Exception as e:
        return f"Fehler: {str(e)[:20]}..."
    
def debug_model_results():
    """Debugging-Funktion um zu sehen, was in den Modell-Resultaten gespeichert ist"""
    st.markdown("### 🔍 Debug: Modell-Resultate Struktur")
    
    if not st.session_state.get("model_results"):
        st.warning("Keine Modell-Resultate vorhanden")
        return
    
    for model_name, result in st.session_state["model_results"].items():
        with st.expander(f"Debug Info: {model_name}"):
            st.write("**Verfügbare Keys:**", list(result.keys()))
            
            # Model-Objekt analysieren
            if 'model' in result:
                model = result['model']
                st.write("**Model Type:**", type(model).__name__)
                
                # Attribute des Modells anzeigen
                model_attrs = [attr for attr in dir(model) if not attr.startswith('_')]
                st.write("**Model Attributes (relevante):**")
                for attr in ['feature_importances_', 'coef_', 'feature_importance']:
                    if hasattr(model, attr):
                        st.success(f" {attr} vorhanden")
                        try:
                            value = getattr(model, attr)
                            if callable(value):
                                st.write(f"   - {attr}(): {type(value())}")
                            else:
                                st.write(f"   - {attr}: {type(value)}, Shape: {getattr(value, 'shape', 'N/A')}")
                        except Exception as e:
                            st.write(f"   - Fehler beim Zugriff: {e}")
                    else:
                        st.error(f" {attr} nicht vorhanden")
            
            # X_test analysieren
            if 'X_test' in result:
                st.write("**X_test Shape:**", result['X_test'].shape)
                st.write("**X_test Columns:**", list(result['X_test'].columns))
            else:
                st.error(" X_test nicht gespeichert")
            
            # Confusion Matrix analysieren
            for cm_key in ['conf_matrix', 'conf_matrix_df', 'confusion_matrix']:
                if cm_key in result:
                    st.success(f" {cm_key} vorhanden")
                    st.write(f"   - Type: {type(result[cm_key])}")
                    st.write(f"   - Shape: {getattr(result[cm_key], 'shape', 'N/A')}")
                else:
                    st.error(f" {cm_key} nicht vorhanden")
# 2. VERBESSERTE GET_TOP_FEATURES FUNKTION
def get_top_features_fixed(model_result, top_n=3):
    """Verbesserte Funktion zur Feature-Extraktion mit mehr Debugging"""
    try:
        feature_importance = None
        feature_names = None
        model_name = "Unknown"
        
        # Debug-Info sammeln
        debug_info = []
        
        # 1. Schaue nach bereits gespeicherten Feature Importance
        if 'feature_importance' in model_result:
            feature_importance = model_result['feature_importance']
            debug_info.append("Found stored feature_importance")
            
        # 2. Extrahiere aus dem Modell
        elif 'model' in model_result:
            model = model_result['model']
            model_name = type(model).__name__
            debug_info.append(f"Model type: {model_name}")
            
            # Für sklearn Random Forest, ExtraTreesClassifier etc.
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                debug_info.append("Using feature_importances_")
                
            # Für LightGBM
            elif hasattr(model, 'feature_importance') and callable(getattr(model, 'feature_importance')):
                try:
                    feature_importance = model.feature_importance(importance_type='gain')
                    debug_info.append("Using LightGBM feature_importance()")
                except:
                    feature_importance = model.feature_importance()
                    debug_info.append("Using LightGBM feature_importance() default")
            
            # Für XGBoost
            elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'get_importance'):
                try:
                    importance_dict = model.get_booster().get_importance()
                    if 'X_test' in model_result:
                        feature_names = model_result['X_test'].columns.tolist()
                        feature_importance = [importance_dict.get(f, 0) for f in feature_names]
                        debug_info.append("Using XGBoost get_importance()")
                    else:
                        debug_info.append("XGBoost: No X_test for feature names")
                except Exception as e:
                    debug_info.append(f"XGBoost error: {e}")
            
            # Für Logistic Regression
            elif hasattr(model, 'coef_'):
                coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                feature_importance = abs(coef)
                debug_info.append("Using logistic regression coefficients")
            
            # Für Neural Networks - oft keine direkte Feature Importance
            else:
                debug_info.append(f"No feature importance method found for {model_name}")
        
        # 3. Feature-Namen extrahieren
        if 'X_test' in model_result and model_result['X_test'] is not None:
            feature_names = model_result['X_test'].columns.tolist()
            debug_info.append(f"Feature names from X_test: {len(feature_names)} features")
        elif 'feature_names' in model_result:
            feature_names = model_result['feature_names']
            debug_info.append("Feature names from stored feature_names")
        
        # 4. Verarbeitung der Feature Importance
        if feature_importance is not None and len(feature_importance) > 0:
            # Konvertiere zu numpy array falls nötig
            import numpy as np
            feature_importance = np.array(feature_importance)
            
            # Standard Feature-Namen falls keine vorhanden
            if feature_names is None or len(feature_names) != len(feature_importance):
                feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
                debug_info.append("Using default feature names")
            
            # DataFrame erstellen und sortieren
            feature_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            # Top N Features
            top_features = feature_df.head(top_n)['feature'].tolist()
            result = ', '.join(top_features)
            
            # Debug info für Entwicklung (kann später entfernt werden)
            if st.session_state.get("debug_mode", False):
                st.write(f"**{model_name} Debug:**", debug_info)
            
            return result
        
        else:
            debug_msg = f"No feature importance found. Debug: {'; '.join(debug_info)}"
            if st.session_state.get("debug_mode", False):
                st.write(f"**{model_name}:**", debug_msg)
            return "N/A"
            
    except Exception as e:
        error_msg = f"Error: {str(e)[:50]}..."
        if st.session_state.get("debug_mode", False):
            st.error(f"Feature extraction error: {e}")
        return error_msg
# ---------------------------------------------------------
# 3. Sidebar-Konfiguration
# ---------------------------------------------------------
# In der Sidebar-Konfiguration hinzufügen:
st.sidebar.title('Modell-Einstellungen')
st.session_state["use_model_persistence"] = st.sidebar.checkbox(
    "Modelle automatisch speichern/laden", 
    value=st.session_state.get("use_model_persistence", True),
    help="Aktiviert automatisches Speichern und Laden trainierter Modelle"
)

# Bestehende Sidebar-Elemente...
st.sidebar.title('Sidebar')
upload_file = st.sidebar.file_uploader('Hochladen einer Datei mit Mitarbeiterdaten', type=["csv"])

# Modell-Management Sidebar hinzufügen
#show_model_management_sidebar()

st.sidebar.title('Navigation')
page = st.sidebar.selectbox('Wählen Sie aus, was Sie anzeigen möchten:', ['Analyse', 'Vorhersage','Management-Dashboard','Evaluierung'])
# ---------------------------------------------------------
# 4. Datei-Upload und automatisches Training
# ---------------------------------------------------------
if upload_file is not None:
    try:
        # Lade neue Daten
        df = pd.read_csv(upload_file, sep=";", encoding="ansi")
        new_hash = calculate_data_hash(df)
        
        # Prüfe, ob sich die Daten geändert haben
        if st.session_state["data_hash"] != new_hash:
            st.session_state["uploaded_data"] = df
            st.session_state["data_hash"] = new_hash
            st.session_state["models_trained"] = False
            st.session_state["model_results"] = {}
            st.session_state["all_train_losses"] = {}
            st.session_state["all_val_losses"] = {}
            st.session_state["training_status"] = {}
            
            # Wähle Training-Methode basierend auf Einstellungen
            if st.session_state.get("use_model_persistence", True):
                # Mit Persistierung - lädt/speichert automatisch
                train_all_models_with_persistence(df)
            else:
                # Ohne Persistierung - wie bisher
                train_all_models(df)
        else:
            # Daten haben sich nicht geändert
            if not st.session_state.get("models_trained", False):
                if st.session_state.get("use_model_persistence", True):
                    # Versuche Modelle zu laden, auch wenn sie nicht im Session State sind
                    train_all_models_with_persistence(df)
                
    except Exception as e:
        st.error(f"Fehler beim Laden der Datei: {e}")

# ---------------------------------------------------------
# 5. Zusätzliche Funktionen für bessere Benutzererfahrung
# ---------------------------------------------------------

def show_model_cache_status():
    """Zeigt Status der Modell-Caches an mit Lösch-Button"""
    if st.session_state.get("use_model_persistence", True):
        models_info = get_saved_models_info()
        current_hash = st.session_state.get("data_hash", "")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader(" Cache-Status")
        
        if models_info:
            current_models = [m for m in models_info if m['data_hash'] == current_hash[:8]]
            if current_models:
                st.sidebar.success(f" {len(current_models)} Modelle im Cache")
            else:
                st.sidebar.info("ℹ Keine Modelle für aktuelle Daten")
        else:
            st.sidebar.info("ℹ Kein Cache vorhanden")
        
        # Einfacher Cache-Lösch-Button
        if st.sidebar.button(" Cache löschen", help="Löscht alle gespeicherten Modelle und Session-Daten"):
            if clear_all_cache():
                st.rerun()
def clear_all_cache():
    """Löscht den kompletten Cache (Festplatte + Session State)"""
    try:
        # 1. Festplattencache löschen
        model_dir = "saved_models"
        if os.path.exists(model_dir):
            import shutil
            shutil.rmtree(model_dir)
        
        # 2. Session State Cache löschen
        cache_keys = [
            "model_results",
            "all_train_losses", 
            "all_val_losses",
            "training_status",
            "models_trained",
            "prediction_results"
        ]
        
        for key in cache_keys:
            if key in st.session_state:
                st.session_state[key] = {} if key.endswith(('results', 'losses', 'status')) else False
        
        st.success(" Cache erfolgreich gelöscht!")
        return True
        
    except Exception as e:
        st.error(f" Fehler beim Löschen des Caches: {e}")
        return False
# Fügen Sie diese Funktion zu Ihrer Sidebar hinzu
show_model_cache_status()

# ---------------------------------------------------------
# 6. Hilfsfunktion für Confusion Matrix mit Erklärung
# ---------------------------------------------------------
def plot_confusion_matrix_interactive(cm, df_test, target_col="Aktiv"):
    labels = ["Inaktiv", "Aktiv"]  # 0 = Inaktiv, 1 = Aktiv
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
# ---------------------------------------------------------
# 7. Korrelationsmatrix mit Plotly
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
        yaxis_showgrid=False
    )
    return fig
# ---------------------------------------------------------
# 8. Seiten-Funktionen
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
# 9. Analyse-Funktionen
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
        st.markdown("---")
        st.subheader("Korrelationsmatrix")
        fig = plot_ordinal_correlation_matrix(df)
        st.plotly_chart(fig, use_container_width=True, height=800)
    else:
        st.warning("Bitte lade zuerst eine CSV-Datei hoch.")
        
    # Diagramm 4 - Scatterplot
        # Scatterplot
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
        # Wir nehmen hier den Mittelwert der Aktiv-Variable pro Kombination
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
# 10. Vorhersage-Funktionen (vereinfacht für trainierte Modelle)
# ---------------------------------------------------------
def prediction_model():
    if st.session_state["uploaded_data"] is None:
        st.warning("Bitte laden Sie zunächst eine Datei hoch.")
        return
    
    df = st.session_state["uploaded_data"]
    
    # Debug-Modus Toggle
    # st.session_state["debug_mode"] = st.checkbox("Debug-Modus aktivieren", value=False)
    
    if st.session_state.get("debug_mode", False):
        debug_model_results()
        st.markdown("---")
    
    # Erweiterte Liste der verfügbaren Modelle/Analysen
    available_models = list(st.session_state.get("model_results", {}).keys())
    additional_methods = ["Kausalanalyse", "Szenario Simulation"]
    
    # Kombiniere trainierte Modelle mit zusätzlichen Methoden
    all_options = available_models + additional_methods
    
    if not all_options:
        st.warning("Keine Modelle oder Analysemethoden verfügbar. Bitte erst eine CSV-Datei hochladen.")
        return
    
    model_choice = st.selectbox("Wähle einen ML-Algorithmus oder eine Analysemethode", all_options)
    
    # Behandlung der trainierten ML-Modelle
    if model_choice in st.session_state["model_results"]:
        results = st.session_state["model_results"][model_choice]
        
        st.markdown(f"## Modell: {model_choice}")
        st.info(f"{model_choice} – Klassifikation mit Modellbewertung (bereits trainiert)")
        
        # Metriken anzeigen
        metrics = results.get('metrics', {})
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if 'accuracy' in metrics:
                    st.metric("Genauigkeit", f"{metrics['accuracy']:.3f}")
            with col2:
                if 'precision' in metrics:
                    st.metric("Präzision", f"{metrics['precision']:.3f}")
            with col3:
                if 'recall' in metrics:
                    st.metric("Recall", f"{metrics['recall']:.3f}")
            with col4:
                if 'f1' in metrics:
                    st.metric("F1-Score", f"{metrics['f1']:.3f}")
        
        # CONFUSION MATRIX FIX - Spezielle Behandlung für Random Forest
        confusion_matrix_data = None
        test_data = None
        y_test_values = None
        y_pred_values = None
        
        # Fall 1: Random Forest mit last_test_idx und last_test_pred
        if "last_test_idx" in results and "last_test_pred" in results:
            try:
                # Erstelle Testdaten
                test_indices = results["last_test_idx"]
                predictions = results["last_test_pred"]
                
                test_data = df.iloc[test_indices].copy()
                test_data["Vorhersage"] = predictions
                test_data.insert(0, "Neuer_Index", range(len(test_data)))
                
                # Extrahiere y_test aus den Original-Daten
                y_test_values = df.iloc[test_indices]["Aktiv"].values
                y_pred_values = predictions
                
                # Erstelle Confusion Matrix
                from sklearn.metrics import confusion_matrix
                confusion_matrix_data = confusion_matrix(y_test_values, y_pred_values)
                
                st.markdown("###  Testdaten des letzten Folds")
                st.dataframe(test_data)
                
                if st.session_state.get("debug_mode", False):
                    st.success(" Confusion Matrix aus last_test_idx/last_test_pred erstellt")
                    st.write(f"y_test shape: {y_test_values.shape}, y_pred shape: {y_pred_values.shape}")
                
            except Exception as e:
                if st.session_state.get("debug_mode", False):
                    st.error(f"Fehler bei last_test_idx/last_test_pred: {e}")
        
        # Fall 2: Standard-Methode mit X_test, y_test, y_pred
        elif "X_test" in results and "y_test" in results and "y_pred" in results:
            try:
                X_test = results["X_test"]
                y_test_values = results["y_test"]
                y_pred_values = results["y_pred"]
                
                # Erstelle Testdaten-DataFrame
                test_data = X_test.copy()
                test_data["Aktiv"] = y_test_values
                test_data["Vorhersage"] = y_pred_values
                test_data.insert(0, "Neuer_Index", range(len(test_data)))
                
                # Erstelle Confusion Matrix
                from sklearn.metrics import confusion_matrix
                confusion_matrix_data = confusion_matrix(y_test_values, y_pred_values)
                
                st.markdown("### Testdaten")
                st.dataframe(test_data)
                
                if st.session_state.get("debug_mode", False):
                    st.success(" Confusion Matrix aus X_test/y_test/y_pred erstellt")
                
            except Exception as e:
                if st.session_state.get("debug_mode", False):
                    st.error(f"Fehler bei X_test/y_test/y_pred: {e}")
        
        # Fall 3: Suche nach bereits gespeicherter Confusion Matrix
        else:
            for cm_key in ['conf_matrix', 'conf_matrix_df', 'confusion_matrix']:
                if cm_key in results:
                    confusion_matrix_data = results[cm_key]
                    if st.session_state.get("debug_mode", False):
                        st.success(f" Gespeicherte Confusion Matrix gefunden: {cm_key}")
                    break
        
        # Zeige Confusion Matrix an
        if confusion_matrix_data is not None and test_data is not None:

            # Konvertiere zu numpy array falls nötig
            if hasattr(confusion_matrix_data, 'values'):
                cm_array = confusion_matrix_data.values
            else:
                cm_array = confusion_matrix_data
            
            plot_confusion_matrix_interactive(cm_array, test_data)
            
        else:
            st.warning("Confusion Matrix konnte nicht angezeigt werden.")
            if st.session_state.get("debug_mode", False):
                st.write("**Debug Info:**")
                st.write("- Verfügbare Schlüssel in results:", list(results.keys()))
                
                if "last_test_idx" in results:
                    st.write(f"- last_test_idx Länge: {len(results['last_test_idx'])}")
                if "last_test_pred" in results:
                    st.write(f"- last_test_pred Länge: {len(results['last_test_pred'])}")
                if "X_test" in results:
                    st.write(f"- X_test Shape: {results['X_test'].shape}")
                
                # Versuche manuell eine Confusion Matrix zu erstellen
                if "last_test_idx" in results and "last_test_pred" in results:
                    try:
                        test_indices = results["last_test_idx"]
                        predictions = results["last_test_pred"]
                        actual_values = df.iloc[test_indices]["Aktiv"].values
                        
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(actual_values, predictions)
                        
                        st.success("Manuelle Confusion Matrix Erstellung erfolgreich!")
                        st.write("Confusion Matrix:", cm)
                        
                        # Zeige sie trotzdem an
                        test_data_manual = df.iloc[test_indices].copy()
                        test_data_manual["Vorhersage"] = predictions
                        plot_confusion_matrix_interactive(cm, test_data_manual)
                        
                    except Exception as e:
                        st.error(f"Auch manuelle Erstellung fehlgeschlagen: {e}")
        
        # Feature Importance anzeigen (falls verfügbar)
        if st.checkbox("Feature Importance anzeigen"):
            try:
                top_features = get_top_features_fixed(results, top_n=10)
                if top_features != "N/A":
                    st.success(f"**Top 10 Features für {model_choice}:** {top_features}")
                else:
                    st.info("Keine Feature Importance verfügbar.")
                    
                    # Versuche direkt aus dem Modell zu extrahieren
                    if "model" in results:
                        model = results["model"]
                        if hasattr(model, 'feature_importances_'):
                            importance = model.feature_importances_
                            if "X_test" in results:
                                feature_names = results["X_test"].columns.tolist()
                                feature_df = pd.DataFrame({
                                    'Feature': feature_names,
                                    'Importance': importance
                                }).sort_values('Importance', ascending=False)
                                
                                st.dataframe(feature_df.head(10))
                                
                                # Balkendiagramm
                                fig = px.bar(feature_df.head(10), 
                                           x='Importance', y='Feature', 
                                           orientation='h',
                                           title=f"Top 10 Features - {model_choice}")
                                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.write("Feature Importances gefunden, aber keine Feature-Namen verfügbar")
                                st.write("Importances:", importance[:10])
                        else:
                            st.info("Modell hat keine feature_importances_ Eigenschaft")
            except Exception as e:
                st.error(f"Fehler bei Feature Importance: {e}")
        
        # SHAP Analyse (nur für LightGBM)
        if st.checkbox("SHAP Analyse anzeigen"):
            try:
                if model_choice == "LightGBM":
                    show_shap_plots_lightgbm(apply_ordinal_encoding(df), results)
                elif model_choice == "Random Forest":
                    # Lade das letzte gespeicherte Random-Forest-Modell                           
                            # SHAP Analyse aufrufen
                            run_shap_analysis(
                                model=results("clf"),
                                X_test=results("X_test"),
                                X_train=results("X_train"),  # Nur falls X_train gespeichert wurde
                                model_name="Random Forest"
                            )
            except Exception as e:
                st.error(f"SHAP Analyse Fehler: {e}")

    # Rest bleibt gleich für Kausalanalyse und Szenario Simulation...
    elif model_choice == "Kausalanalyse":
        st.markdown("## Kausalanalyse")
        st.info("Führe Kausalanalyse durch...")
        
        try:
            df_encoded = apply_ordinal_encoding_causal(df)
            run_causal_analysis(df_encoded)
        except Exception as e:
            st.error(f"Fehler bei der Kausalanalyse: {e}")
    
    elif model_choice == "Szenario Simulation":
        st.markdown("## Szenario Simulation")
        st.info("Führe Szenario Simulation durch...")
        
        try:
            df_encoded = apply_ordinal_encoding(df)
            
            if "Gehaltsentwicklung" not in df_encoded.columns:
                st.error("Die Spalte 'Gehaltsentwicklung' wurde nicht in den Daten gefunden.")
                st.write("Verfügbare Spalten:", df_encoded.columns.tolist())
                return
            
            wert_haeufigkeiten = df_encoded["Gehaltsentwicklung"].value_counts()
            
            st.title("Auswertung der Gehaltsentwicklung")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Häufigkeitsverteilung als Tabelle")
                freq_df = pd.DataFrame({
                    'Gehaltsentwicklung': wert_haeufigkeiten.index,
                    'Anzahl': wert_haeufigkeiten.values,
                    'Prozent': (wert_haeufigkeiten.values / len(df_encoded) * 100).round(2)
                })
                st.dataframe(freq_df)
                
                st.subheader("Statistische Kennzahlen")
                st.write(f"**Gesamtanzahl:** {len(df_encoded)}")
                st.write(f"**Anzahl verschiedener Werte:** {len(wert_haeufigkeiten)}")
                st.write(f"**Häufigster Wert:** {wert_haeufigkeiten.index[0]} ({wert_haeufigkeiten.iloc[0]} mal)")
            
            with col2:
                st.subheader("Häufigkeitsverteilung als Balkendiagramm")
                fig = px.bar(
                    x=wert_haeufigkeiten.index,
                    y=wert_haeufigkeiten.values,
                    labels={'x': 'Gehaltsentwicklung', 'y': 'Anzahl'},
                    title="Verteilung der Gehaltsentwicklung"
                )
                fig.update_layout(xaxis_title="Gehaltsentwicklung", yaxis_title="Anzahl")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Prozentuale Verteilung")
                fig_pie = px.pie(
                    values=wert_haeufigkeiten.values,
                    names=wert_haeufigkeiten.index,
                    title="Prozentuale Verteilung der Gehaltsentwicklung"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
        except Exception as e:
            st.error(f"Fehler bei der Szenario Simulation: {e}")
# ---------------------------------------------------------
# 10. Management Dashboard
# ---------------------------------------------------------
def management_dashboard():
    if st.session_state["uploaded_data"] is None:
        st.warning("Bitte laden Sie zunächst eine Datei hoch.")
        return
    df = st.session_state["uploaded_data"]
    
    # Verwende bereits trainierte LightGBM-Ergebnisse falls verfügbar
    if "LightGBM" in st.session_state.get("model_results", {}):
        model_output = st.session_state["model_results"]["LightGBM"]
    else:
        # Fallback: Trainiere LightGBM falls nicht verfügbar
        df_encoded = apply_ordinal_encoding(df)
        model_output = run_lightgbm_classification(df_encoded)
    
    # Speichere die Vorhersagen im session_state für andere Seiten
    st.session_state["prediction_results"] = {
        "y_test": model_output.get("y_test", []),
        "y_pred": model_output.get("y_pred", []),
    }
    show_hr_dashboard(model_output)
    
    # Confusion Matrix anzeigen
    if "last_test_idx" in model_output and "last_test_pred" in model_output:
        df_last_test = df.iloc[model_output["last_test_idx"]].copy()
        df_last_test["Vorhersage"] = model_output["last_test_pred"]
        df_last_test.insert(0, "Neuer_Index", range(len(df_last_test)))
        
        st.markdown("### Confusion Matrix mit Erklärungen")
        if "conf_matrix_df" in model_output:
            plot_confusion_matrix_interactive(model_output["conf_matrix_df"].values, df_last_test)
# ---------------------------------------------------------
# 11. Erweiterte Evaluierung mit allen Modellen
# ---------------------------------------------------------
def evaluate():
    """Erweiterte Evaluierung mit fixen Features"""
    st.title("Modell-Evaluierung und Vergleich")
    
    if not st.session_state.get("models_trained", False):
        st.warning("Keine trainierten Modelle verfügbar. Bitte erst eine CSV hochladen.")
        return
    
    # Debug-Toggle
    st.session_state["debug_mode"] = st.sidebar.checkbox("Debug-Modus", value=False)
    
    if st.session_state.get("debug_mode", False):
        with st.expander("Debug-Informationen"):
            debug_model_results()
    
    # Tab-Layout
    tab1, tab2, tab3 = st.tabs(["Loss Kurven", "Modellvergleich", "ROC-Kurven"])
    with tab1:
        st.subheader("Verlustfunktionen aller verfügbaren Modelle")
        
        # Alle Modelle mit Loss-Kurven anzeigen
        models_with_loss = list(st.session_state.get("all_train_losses", {}).keys())
        
        if not models_with_loss:
            st.info("Keine Loss-Kurven verfügbar. Modelle wie XGBoost, LightGBM und Neural Networks speichern Loss-Kurven.")
            return
        # Zeige alle verfügbaren Loss-Kurven
        if models_with_loss:
            for i, model_name in enumerate(models_with_loss):
                train_losses = st.session_state["all_train_losses"].get(model_name, [])
                val_losses = st.session_state["all_val_losses"].get(model_name, [])
                if train_losses:
                    # Berechne Durchschnitt falls mehrere Runs vorhanden
                    if isinstance(train_losses[0], list):
                        # Multiple folds - berechne Durchschnitt
                        max_len = max(len(l) for l in train_losses)
                        mean_train_loss = np.nanmean([
                            np.pad(l, (0, max_len - len(l)), constant_values=np.nan) 
                            for l in train_losses
                        ], axis=0)
                        mean_val_loss = np.nanmean([
                            np.pad(l, (0, max_len - len(l)), constant_values=np.nan) 
                            for l in val_losses
                        ], axis=0)
                    else:
                        mean_train_loss = train_losses
                        mean_val_loss = val_losses
                    # Erstelle das Plotly-Diagramm
                    fig = go.Figure()
                    
                    # Überprüfe, ob es gültige Daten zum Plotten gibt
                    if not np.all(np.isnan(mean_train_loss)):
                        fig.add_trace(go.Scatter(y=mean_train_loss, mode='lines', name='Training Loss',
                                                 line=dict(color='blue', width=2)))
                    
                    if not np.all(np.isnan(mean_val_loss)):
                        fig.add_trace(go.Scatter(y=mean_val_loss, mode='lines', name='Validation Loss',
                                                 line=dict(color='red', width=2)))
                    fig.update_layout(
                        title=f"Loss Curve - {model_name}",
                        xaxis_title="Epoch / Iteration",
                        yaxis_title="Loss",
                        hovermode="x unified",
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                    )
                    
                    # Zeige das Diagramm nur, wenn es mindestens eine gültige Kurve enthält
                    if fig.data:
                        st.plotly_chart(fig, use_container_width=True)
                        # HIER IST DIE EINFACHE PDF-EXPORT ERGÄNZUNG (nur 5 Zeilen!)
                        if st.button(f" Als PDF speichern", key=f"pdf_{model_name}"):
                            pdf_file = save_loss_curve_as_pdf(model_name, train_losses, val_losses)
                            if pdf_file:
                                st.success(f" PDF gespeichert als: {pdf_file}")
                        st.markdown("---")
                    else:
                        st.info(f"Keine gültigen Loss-Daten für {model_name} gefunden.")
    with tab2:  # Hauptfokus auf Modellvergleich
        st.subheader("Übersicht aller trainierten Modelle")
        
        model_results = st.session_state.get("model_results", {})
        
        # Erstelle erweiterte Übersichtstabelle mit verbesserter Feature-Extraktion
        overview_data = []
        for model_name in model_results.keys():
            metrics = model_results[model_name].get("metrics", {})
            
            # Verwende die verbesserte Feature-Extraktion
            top_features = get_top_features_fixed(model_results[model_name], top_n=3)
            
            overview_data.append({
                "Modell": model_name,
                "Genauigkeit": f"{metrics.get('accuracy', 0):.3f}" if 'accuracy' in metrics else "N/A",
                "Präzision": f"{metrics.get('precision', 0):.3f}" if 'precision' in metrics else "N/A",
                "Recall": f"{metrics.get('recall', 0):.3f}" if 'recall' in metrics else "N/A",
                "F1-Score": f"{metrics.get('f1', 0):.3f}" if 'f1' in metrics else "N/A",
                "Top 3 Features": top_features
            })
        
        if overview_data:
            overview_df = pd.DataFrame(overview_data)
            st.dataframe(overview_df, use_container_width=True)
            
            # Feature Importance Details in einem Expander
            with st.expander("🔍 Detaillierte Feature Importance Analyse"):
                selected_model = st.selectbox(
                    "Modell für detaillierte Feature-Analyse auswählen:",
                    list(model_results.keys())
                )
                
                if selected_model and selected_model in model_results:
                    model_result = model_results[selected_model]
                    
                    # Versuche detaillierte Feature Importance zu extrahieren
                    try:
                        feature_importance = None
                        feature_names = None
                        
                        # Feature Importance extrahieren (gleiche Logik wie oben)
                        if 'feature_importance' in model_result:
                            feature_importance = model_result['feature_importance']
                            if 'feature_names' in model_result:
                                feature_names = model_result['feature_names']
                        elif 'model' in model_result:
                            model = model_result['model']
                            
                            if hasattr(model, 'feature_importances_'):
                                feature_importance = model.feature_importances_
                                if 'X_test' in model_result:
                                    feature_names = model_result['X_test'].columns.tolist()
                            elif hasattr(model, 'feature_importance'):
                                feature_importance = model.feature_importance()
                                if 'X_test' in model_result:
                                    feature_names = model_result['X_test'].columns.tolist()
                            elif hasattr(model, 'coef_'):
                                feature_importance = abs(model.coef_[0]) if len(model.coef_.shape) > 1 else abs(model.coef_)
                                if 'X_test' in model_result:
                                    feature_names = model_result['X_test'].columns.tolist()
                        
                        if feature_importance is not None:
                            if feature_names is None:
                                feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
                            
                            # Erstelle detailliertes DataFrame
                            feature_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': feature_importance,
                                'Importance_Prozent': (feature_importance / feature_importance.sum()) * 100
                            }).sort_values('Importance', ascending=False)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"**Top 10 Features für {selected_model}:**")
                                st.dataframe(feature_df.head(10))
                            
                            with col2:
                                # Balkendiagramm für Top 10 Features
                                fig_features = px.bar(
                                    feature_df.head(10),
                                    x='Importance',
                                    y='Feature',
                                    orientation='h',
                                    title=f"Top 10 Features - {selected_model}"
                                )
                                fig_features.update_layout(yaxis={'categoryorder': 'total ascending'})
                                st.plotly_chart(fig_features, use_container_width=True)
                        else:
                            st.info(f"Keine Feature Importance für {selected_model} verfügbar.")
                            
                    except Exception as e:
                        st.error(f"Fehler bei Feature-Analyse: {e}")
            
            # Leistungsvergleich direkt unter der Tabelle
            st.markdown("---")
            st.subheader("Leistungsvergleich")
            
            # Rest des bestehenden Codes für Leistungsvergleich...
            if len(model_results) > 1:
                metrics_data = []
                for model_name, result in model_results.items():
                    metrics = result.get("metrics", {})
                    if metrics:
                        metrics_data.append({
                            "Modell": model_name,
                            "Genauigkeit": metrics.get('accuracy', 0),
                            "Präzision": metrics.get('precision', 0),
                            "Recall": metrics.get('recall', 0),
                            "F1-Score": metrics.get('f1', 0)
                        })
                
                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    
                    # Zwei Spalten für die Visualisierungen
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Radar Chart für Modellvergleich
                        st.markdown("##### Radar Chart - Modellvergleich")
                        
                        all_values = []
                        for _, row in metrics_df.iterrows():
                            all_values.extend([row['Genauigkeit'], row['Präzision'], row['Recall'], row['F1-Score']])
                        
                        min_value = max(0, min(all_values) - 0.05)
                        max_value = min(1, max(all_values) + 0.05)
                        
                        if max_value - min_value < 0.1:
                            center = (max_value + min_value) / 2
                            min_value = max(0, center - 0.1)
                            max_value = min(1, center + 0.1)
                        
                        fig = go.Figure()
                        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
                        
                        for i, (_, row) in enumerate(metrics_df.iterrows()):
                            fig.add_trace(go.Scatterpolar(
                                r=[row['Genauigkeit'], row['Präzision'], row['Recall'], row['F1-Score']],
                                theta=['Genauigkeit', 'Präzision', 'Recall', 'F1-Score'],
                                fill='toself',
                                name=row['Modell'],
                                line=dict(width=3, color=colors[i % len(colors)]),
                                fillcolor=colors[i % len(colors)],
                                opacity=0.6
                            ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[min_value, max_value],
                                    tickmode='linear',
                                    tick0=min_value,
                                    dtick=(max_value - min_value) / 5,
                                    tickformat='.3f'
                                )),
                            showlegend=True,
                            height=450
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Bar Chart Vergleich
                        st.markdown("##### F1-Score Vergleich")
                        
                        fig_bar = px.bar(
                            metrics_df.sort_values('F1-Score', ascending=False),
                            x='Modell',
                            y='F1-Score',
                            color='F1-Score',
                            color_continuous_scale='viridis'
                        )
                        
                        fig_bar.update_layout(height=400)
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Top 3 Modelle
                    st.markdown("##### Top 3 Modelle (nach F1-Score)")
                    top_models = metrics_df.nlargest(3, 'F1-Score')[['Modell', 'F1-Score']]
                    
                    cols = st.columns(3)
                    for i, (_, row) in enumerate(top_models.iterrows()):
                        with cols[i]:
                            if i == 0:
                                st.success(f" **{row['Modell']}**\nF1: {row['F1-Score']:.3f}")
                            elif i == 1:
                                st.info(f" **{row['Modell']}**\nF1: {row['F1-Score']:.3f}")
                            else:
                                st.warning(f" **{row['Modell']}**\nF1: {row['F1-Score']:.3f}")
                else:
                    st.info("Mindestens 2 Modelle nötig für einen Vergleich.")
            else:
                st.info("Mindestens 2 Modelle nötig für einen Vergleich.")

    with tab3:
        # ROC Kurven Sektion mit PDF-Export
        st.subheader("ROC Kurven aller verfügbaren Modelle mit PDF-Export")
        
        available_models = list(st.session_state.get("model_results", {}).keys())
        
        if not available_models:
            st.info("Keine trainierten Modelle für ROC-Kurven verfügbar.")
        else:
            for model_name in available_models:
                model_result = st.session_state["model_results"].get(model_name, {})
                
                if ("X_test" in model_result and "y_test" in model_result and 
                    "model" in model_result):
                    
                    try:
                        # Verwende die neue Funktion mit PDF-Export
                        plot_roc_curve_simple_with_export(model_name, model_result)
                        st.markdown("---")
                    except Exception as e:
                        st.info(f"ROC-Kurve für {model_name} konnte nicht erstellt werden: {str(e)[:50]}...")
        
def check_dependencies():
    """Prüft, ob alle erforderlichen Pakete installiert sind"""
    try:
        import joblib
        import pickle
        return True
    except ImportError as e:
        st.error(f"""
        Fehlende Abhängigkeiten für Modell-Persistierung: {e}
        
        Installieren Sie die fehlenden Pakete:
        ```
        pip install joblib
        ```
        
        Modelle werden nur im Session State gespeichert.
        """)
        st.session_state["use_model_persistence"] = False
        return False

# Führen Sie diese Prüfung am Anfang aus
if not check_dependencies():
    st.session_state["use_model_persistence"] = False     

# ---------------------------------------------------------
# 11. PDF EXPORT LÖSUNG
# ---------------------------------------------------------
# Alternative: Spezialisierte Funktion für verschiedene Modelltypen

def save_loss_curve_as_pdf(model_name, train_losses, val_losses):
    """Robuste PDF-Export Funktion für alle Modelltypen"""
    try:
        # Debug: Struktur der Daten anzeigen (erweitert)
        print(f"Debug {model_name}:")
        print(f"train_losses type: {type(train_losses)}")
        print(f"train_losses length/shape: {len(train_losses) if hasattr(train_losses, '__len__') else 'no length'}")
        if hasattr(train_losses, '__len__') and len(train_losses) > 0:
            print(f"First element type: {type(train_losses[0])}")
            if hasattr(train_losses[0], '__len__'):
                print(f"First element length: {len(train_losses[0])}")
            # Zeige Längen aller Elemente (bei CV-Folds)
            if isinstance(train_losses, list):
                lengths = [len(x) if hasattr(x, '__len__') else 1 for x in train_losses]
                print(f"All element lengths: {lengths}")
        
        # Robuste Behandlung verschiedener Datenstrukturen
        processed_train_losses = None
        processed_val_losses = None
        
        # Fall 1: Normale Liste von Zahlen (einfachster Fall)
        if isinstance(train_losses, (list, np.ndarray)) and len(train_losses) > 0:
            if isinstance(train_losses[0], (int, float, np.number)):
                processed_train_losses = train_losses
                processed_val_losses = val_losses
            
            # Fall 2: Liste von Listen (mehrere Folds/Runs)
            elif isinstance(train_losses[0], (list, np.ndarray)):
                # Nimm den ersten Fold/Run
                processed_train_losses = train_losses[0]
                processed_val_losses = val_losses[0] if len(val_losses) > 0 else []
            
            # Fall 3: Leere Liste oder anderer unerwarteter Typ
            else:
                print(f"Unerwarteter Datentyp in train_losses[0]: {type(train_losses[0])}")
                processed_train_losses = []
                processed_val_losses = []
        
        # Fall 4: Leere oder None Daten
        elif not train_losses:
            print("train_losses ist leer oder None")
            processed_train_losses = []
            processed_val_losses = []
        
        # Zusätzliche Sicherheitsprüfungen
        if processed_train_losses is None:
            processed_train_losses = []
        if processed_val_losses is None:
            processed_val_losses = []
            
        # Sichere Konvertierung zu numpy arrays (verhindert inhomogene Arrays)
        def safe_to_array(data):
            if not data:
                return np.array([])
            try:
                # Versuche direkte Konvertierung
                return np.array(data)
            except ValueError as e:
                if "inhomogeneous" in str(e):
                    # Falls inhomogen: Nimm das längste Element oder berechne Durchschnitt
                    if isinstance(data, list) and len(data) > 0:
                        # Nimm einfach das erste Element
                        return np.array(data[0]) if hasattr(data[0], '__len__') else np.array([data[0]])
                    else:
                        return np.array([])
                else:
                    raise e
        
        processed_train_losses = safe_to_array(processed_train_losses)
        processed_val_losses = safe_to_array(processed_val_losses)
        
        # Überprüfe, ob wir gültige Daten haben
        if len(processed_train_losses) == 0 and len(processed_val_losses) == 0:
            print(f"Keine gültigen Loss-Daten für {model_name} gefunden")
            return None
        
        # Plot erstellen mit besserer Lesbarkeit
        plt.figure(figsize=(12, 8))  # Größere Figur
        
        # Schriftgrößen definieren
        TITLE_SIZE = 18
        LABEL_SIZE = 24
        TICK_SIZE = 22
        LEGEND_SIZE = 24
        
        # Plotte nur verfügbare Daten mit dickeren Linien
        if len(processed_train_losses) > 0:
            plt.plot(processed_train_losses, 'b-', label='Training Loss', linewidth=3)
        
        if len(processed_val_losses) > 0:
            plt.plot(processed_val_losses, 'r-', label='Validation Loss', linewidth=3)
        
        # Falls keine Daten vorhanden sind, erstelle einen leeren Plot mit Hinweis
        if len(processed_train_losses) == 0 and len(processed_val_losses) == 0:
            plt.text(0.5, 0.5, 'Keine Loss-Daten verfügbar', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=plt.gca().transAxes, fontsize=20)
        
        # Achsenbeschriftungen und Titel mit größeren Schriftgrößen
        plt.xlabel('Epoch / Iteration', fontsize=LABEL_SIZE, fontweight='bold')
        plt.ylabel('Loss', fontsize=LABEL_SIZE, fontweight='bold')
        # plt.title(f'Loss Curve - {model_name}', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
        
        # Tick-Labels größer machen
        plt.xticks(fontsize=TICK_SIZE)
        plt.yticks(fontsize=TICK_SIZE)
        
        # Legende mit größerer Schrift
        plt.legend(fontsize=LEGEND_SIZE, loc='best', framealpha=0.9)
        
        # Grid mit besserer Sichtbarkeit
        plt.grid(True, alpha=0.3, linewidth=1)
        
        # Sicherer Dateiname
        safe_name = "".join(c for c in model_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        filename = f"loss_curve_{safe_name}.pdf"
        
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close()
        
        return filename
        
    except Exception as e:
        print(f"Detaillierter Fehler in save_loss_curve_as_pdf: {str(e)}")
        print(f"train_losses: {train_losses}")
        print(f"val_losses: {val_losses}")
        st.error(f"PDF-Fehler für {model_name}: {str(e)}")
        return None
# Hilfsfunktionen für verschiedene Modelltypen
def extract_neural_network_losses(losses):
    """Extrahiert Loss-Daten aus Neural Network Strukturen"""
    if losses is None:
        return []
    
    # Keras History Objekt
    if hasattr(losses, 'history'):
        return losses.history.get('loss', [])
    
    # Dict mit 'loss' Key (Keras-Style)
    if isinstance(losses, dict) and 'loss' in losses:
        return losses['loss']
    
    # Normale Liste/Array
    if isinstance(losses, (list, np.ndarray)):
        return np.array(losses).flatten()
    
    return []


def extract_boosting_losses(losses):
    """Extrahiert Loss-Daten aus Boosting-Modellen"""
    if losses is None:
        return []
    
    # XGBoost/LightGBM haben meist einfache Listen
    if isinstance(losses, list):
        # Falls es eine Liste von Listen ist (mehrere Folds)
        if len(losses) > 0 and isinstance(losses[0], list):
            return losses[0]  # Nimm ersten Fold
        else:
            return losses
    
    if isinstance(losses, np.ndarray):
        return losses.flatten()
    
    return []


def extract_generic_losses(train_losses, val_losses):
    """Generische Extraktion für unbekannte Strukturen"""
    def safe_extract(data):
        if data is None:
            return []
        if isinstance(data, (list, np.ndarray)):
            if len(data) > 0 and isinstance(data[0], (list, np.ndarray)):
                return np.array(data[0]).flatten()
            else:
                return np.array(data).flatten()
        return []
    
    return safe_extract(train_losses), safe_extract(val_losses)
# ROC-Kurven PDF Export Funktion
def save_roc_curve_as_pdf(model_name, fpr, tpr, roc_auc, cv_results=None):
    """
    Speichert ROC-Kurve als PDF-Datei
    
    Args:
        model_name: Name des Modells
        fpr: False Positive Rate (einzelne Kurve oder Liste für CV)
        tpr: True Positive Rate (einzelne Kurve oder Liste für CV)
        roc_auc: AUC-Wert(e)
        cv_results: Optional - Cross-Validation Ergebnisse als Dict
    """
    try:
        # Größere Figur für bessere Lesbarkeit
        plt.figure(figsize=(12, 9))
        
        # Schriftgrößen definieren
        TITLE_SIZE = 18
        LABEL_SIZE = 24
        TICK_SIZE = 22
        LEGEND_SIZE = 20
        
        # Fall 1: Cross-Validation mit mehreren Kurven
        if cv_results and isinstance(cv_results, dict):
            # Zeige alle CV-Folds
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            for i, (fold_fpr, fold_tpr, fold_auc) in enumerate(zip(
                cv_results.get('fpr_list', []), 
                cv_results.get('tpr_list', []), 
                cv_results.get('auc_list', [])
            )):
                plt.plot(fold_fpr, fold_tpr, 
                        color=colors[i % len(colors)], 
                        alpha=0.8, 
                        linewidth=2.5,
                        label=f'Fold {i+1} (AUC = {fold_auc:.3f})')
            
            # Durchschnitts-AUC anzeigen
            mean_auc = np.mean(cv_results.get('auc_list', []))
            std_auc = np.std(cv_results.get('auc_list', []))
            
        # Fall 2: Einzelne ROC-Kurve
        else:
            # Behandle verschiedene Datenstrukturen
            if isinstance(fpr, list) and len(fpr) > 0 and isinstance(fpr[0], (list, np.ndarray)):
                # Multiple folds - nimm den ersten oder berechne Durchschnitt
                fpr_plot = fpr[0]
                tpr_plot = tpr[0]
                auc_plot = roc_auc[0] if isinstance(roc_auc, list) else roc_auc
            else:
                fpr_plot = fpr
                tpr_plot = tpr
                auc_plot = roc_auc
            
            # Hauptkurve plotten
            plt.plot(fpr_plot, tpr_plot, 
                    color='#2E86AB', 
                    linewidth=4, 
                    label=f'ROC (AUC = {auc_plot:.3f})')
        
        # Diagonale (Zufallslinie) hinzufügen
        plt.plot([0, 1], [0, 1], 
                color='gray', 
                linestyle='--', 
                linewidth=3, 
                alpha=0.8,
                label='Zufall (AUC = 0.500)')
        
        # Achsenbeschriftungen und Formatierung
        plt.xlabel('False Positive Rate', fontsize=LABEL_SIZE, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=LABEL_SIZE, fontweight='bold')
        
        # Achsen-Limits und Ticks
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xticks(np.arange(0, 1.1, 0.2), fontsize=TICK_SIZE)
        plt.yticks(np.arange(0, 1.1, 0.2), fontsize=TICK_SIZE)
        
        # Grid mit besserer Sichtbarkeit
        plt.grid(True, alpha=0.3, linewidth=1)
        
        # Legende
        plt.legend(fontsize=LEGEND_SIZE, loc='lower right', framealpha=0.9)
        
        # Titel optional (kann auskommentiert werden)
        # plt.title(f'ROC Curve - {model_name}', fontsize=TITLE_SIZE, fontweight='bold', pad=20)
        
        # Sicherer Dateiname
        safe_name = "".join(c for c in model_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        filename = f"roc_curve_{safe_name}.pdf"
        
        # Als PDF speichern
        plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()
        
        return filename
        
    except Exception as e:
        print(f"Fehler beim Speichern der ROC-Kurve für {model_name}: {e}")
        return None


# Erweiterte Funktion für ROC mit Cross-Validation
def save_roc_curve_cv_as_pdf(model_name, model_result, df_encoded=None):
    """
    Erweiterte ROC-PDF-Funktion mit automatischer Cross-Validation
    """
    try:
        from sklearn.metrics import roc_curve, auc
        from sklearn.model_selection import StratifiedKFold
        
        model = model_result["model"]
        
        # Verwende entweder übergebene Daten oder Session State
        if df_encoded is None and st.session_state.get("uploaded_data") is not None:
            df_encoded = apply_ordinal_encoding(st.session_state["uploaded_data"])
        
        if df_encoded is None:
            return save_roc_curve_simple_pdf(model_name, model_result)
        
        X = df_encoded.drop(columns=["Aktiv"])
        y = df_encoded["Aktiv"]
        
        # 5-Fold Cross Validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        fpr_list = []
        tpr_list = []
        auc_list = []
        
        fold = 0
        for train_idx, test_idx in skf.split(X, y):
            fold += 1
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Vorhersage-Wahrscheinlichkeiten
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_proba = model.decision_function(X_test)
            else:
                continue
            
            # ROC-Kurve berechnen
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            auc_list.append(roc_auc)
        
        # CV-Ergebnisse als Dict
        cv_results = {
            'fpr_list': fpr_list,
            'tpr_list': tpr_list,
            'auc_list': auc_list
        }
        
        return save_roc_curve_as_pdf(model_name, None, None, None, cv_results)
        
    except Exception as e:
        print(f"Fehler bei CV-ROC für {model_name}: {e}")
        return save_roc_curve_simple_pdf(model_name, model_result)


def save_roc_curve_simple_pdf(model_name, model_result):
    """
    Fallback-Funktion für einfache ROC-Kurve aus Modellergebnissen
    """
    try:
        from sklearn.metrics import roc_curve, auc
        
        X_test = model_result["X_test"]
        y_test = model_result["y_test"]
        model = model_result["model"]
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            return None
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        return save_roc_curve_as_pdf(model_name, fpr, tpr, roc_auc)
        
    except Exception as e:
        print(f"Fehler bei einfacher ROC für {model_name}: {e}")
        return None     

# Erweiterte plot_roc_curve_simple Funktion mit PDF-Export
def plot_roc_curve_simple_with_export(model_name, model_result):
    """Erstellt eine ROC-Kurve mit PDF-Export-Button"""
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    import plotly.graph_objects as go
    
    try:
        model = model_result["model"]
        
        # Falls Cross-Validation Daten verfügbar sind, simuliere 5-Fold ROC
        if st.session_state["uploaded_data"] is not None:
            df_encoded = apply_ordinal_encoding(st.session_state["uploaded_data"])
            X = df_encoded.drop(columns=["Aktiv"])
            y = df_encoded["Aktiv"]
            
            # 5-Fold Cross Validation für ROC
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            fig = go.Figure()
            aucs = []
            cv_data = {'fpr_list': [], 'tpr_list': [], 'auc_list': []}
            
            fold = 0
            for train_idx, test_idx in skf.split(X, y):
                fold += 1
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Vorhersage-Wahrscheinlichkeiten
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, "decision_function"):
                    y_proba = model.decision_function(X_test)
                else:
                    continue
                
                # ROC-Kurve berechnen
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                
                # Daten für PDF-Export sammeln
                cv_data['fpr_list'].append(fpr)
                cv_data['tpr_list'].append(tpr)
                cv_data['auc_list'].append(roc_auc)
                
                # Füge ROC-Kurve zum Plot hinzu
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'Fold {fold} (AUC = {roc_auc:.3f})',
                    line=dict(width=2)
                ))
            
            # Diagonale (Zufallslinie) hinzufügen
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Zufall (AUC = 0.500)',
                line=dict(dash='dash', color='gray')
            ))
            
            fig.update_layout(
                title=f'ROC Kurve - {model_name} (Durchschnitt AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f})',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                width=600,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # PDF-Export Button für CV-Version
            if st.button(f"ROC als PDF speichern", key=f"roc_pdf_{model_name}"):
                pdf_file = save_roc_curve_as_pdf(model_name, None, None, None, cv_data)
                if pdf_file:
                    st.success(f"ROC-PDF gespeichert als: {pdf_file}")
                else:
                    st.error("Fehler beim Speichern der ROC-PDF")
            
        else:
            # Fallback: Verwende nur die Test-Daten aus dem Modell
            X_test = model_result["X_test"]
            y_test = model_result["y_test"]
            
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_proba = model.decision_function(X_test)
            else:
                st.info(f"Modell {model_name} unterstützt keine Wahrscheinlichkeits-Vorhersagen für ROC.")
                return
            
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                   name=f'ROC (AUC = {roc_auc:.3f})', line=dict(width=3)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                   name='Zufall (AUC = 0.500)', line=dict(dash='dash', color='gray')))
            
            fig.update_layout(
                title=f'ROC Kurve - {model_name} (AUC = {roc_auc:.3f})',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                width=600,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # PDF-Export Button für einfache Version
            if st.button(f"ROC als PDF speichern", key=f"roc_pdf_{model_name}"):
                pdf_file = save_roc_curve_as_pdf(model_name, fpr, tpr, roc_auc)
                if pdf_file:
                    st.success(f"ROC-PDF gespeichert als: {pdf_file}")
                else:
                    st.error("Fehler beim Speichern der ROC-PDF")
            
    except Exception as e:
        st.error(f"Fehler bei ROC-Kurve für {model_name}: {str(e)}")
# ---------------------------------------------------------
# 12. Navigation über Sidebar
# ---------------------------------------------------------
if page == 'Analyse':
    interactive_plot()
elif page == "Vorhersage":
    prediction_model()
elif page == "Management-Dashboard":
    management_dashboard()
elif page == "Evaluierung":
    evaluate()
