import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import plotly.express as px
import seaborn as sns


def run_causal_forest_analysis(df: pd.DataFrame):
    """
    Führt eine Causal Forest Analyse mit binärem Treatment und binärem Outcome durch.

    Parameter:
    - df: Pandas DataFrame mit den nötigen Spalten
    - treatment_threshold: Schwellenwert zur Definition der binären Treatment-Variable

    Gibt zurück:
    - model: das trainierte CausalForestDML-Modell
    - te_pred: geschätzte individuelle Behandlungseffekte
    - conf_int: Konfidenzintervalle der Effekte als Tupel (untere Grenze, obere Grenze)
    """

    # Feature-, Treatment- und Zielvariablen
    feature_cols = [
        "Alter", "2023_Pers_Beurteilung", "2024_Pers_Beurteilung",
        "Betriebszugehoerigkeit", "Zeit_auf_Position", "2024_Ziele", "2023_Ziele"
    ]
    X = df[feature_cols]
    T = df["Gehaltserhöhung"].astype(np.int64).values
    y = df["Aktiv"].astype(np.int64).values

    # Modell trainieren
    est = CausalForestDML(random_state=42)
    est.fit(Y=y, T=T, X=X)

    # Behandlungseffekt schätzen
    te_pred = est.effect(X)

    # Konfidenzintervalle berechnen
    lb, ub = est.effect_interval(X) # lb = untere Grenze (lower bound), ub = obere Grenze (upper bound), jeweils ein NumPy-Array.
    # Konfidenzintervall-Paare kombinieren
    conf_int = list(zip(lb, ub))
    # Berechnet den Median (Zentralwert) der geschätzten Behandlungseffekte.
    # Nützlich zur groben Interpretation: Ein positiver Median → die meisten Personen haben vermutlich einen leicht positiven Effekt.
    median_te = np.median(te_pred)

    # Interquartilsabstand (IQR) G
    iqr_te = np.percentile(te_pred, 75) - np.percentile(te_pred, 25) # Gibt an, in welchem Bereich die mittleren 50 % der Treatment Effects liegen.

    st.write(f"Median des Behandlungseffekts: {median_te:.3f} (bzw. {median_te*100:.1f}%)")
    st.write(f"IQR des Effekts: {iqr_te:.3f}")
    # Beispielhafte Ausgabe
    #st.write("Behandlungseffekt (TE):", te_pred[:5]) 
    #st.write("Konfidenzintervall:", conf_int[:5]) 
    sign_pos = np.sum(np.array(lb) > 0)
    sign_neg = np.sum(np.array(ub) < 0)
    # Anzahl Gesamtfälle
    total = len(te_pred)

    st.write(f"Positive signifikante Effekte: {sign_pos} von {total} ({sign_pos/total:.1%})")
    st.write(f"Negative signifikante Effekte: {sign_neg} von {total} ({sign_neg/total:.1%})")
    # DataFrame für Plotly vorbereiten
        # Dropdown-Auswahl für X-Achse
    selected_x = st.selectbox(
        "Variable für X-Achse im Scatterplot wählen:",
        options=["Alter", "Betriebszugehoerigkeit", "Zeit_auf_Position", "2024_Pers_Beurteilung", "2023_Pers_Beurteilung", "2024_Ziele", "2023_Ziele"]
    )

    # DataFrame für Plotly vorbereiten
    plot_df = df.copy()
    plot_df["Treatment Effect"] = te_pred
    # Treatment-Größen anzeigen
    st.write(f"Anzahl Treatment-Gruppe: {df['Gehaltserhöhung'].sum()} von {len(df)}")

    # Scatterplot
    fig = px.scatter(
        plot_df,
        x=selected_x,
        y="Treatment Effect",
        color="Treatment Effect",
        color_continuous_scale="RdBu",
        title=f"Individueller Behandlungseffekt nach {selected_x}"
    )
    st.plotly_chart(fig, key="scatter_te_plot")

    # 2. Verlauf der Effekte über Schwellenwerte – Plotly-Version
    st.header("Verlauf des Median-Treatment-Effekts über Gehaltsschwellen")

    # Feature-, Treatment- und Zielvariablen
    feature_cols = [
        "Alter", "2023_Pers_Beurteilung", "2024_Pers_Beurteilung",
        "Betriebszugehoerigkeit", "Zeit_auf_Position", "2024_Ziele", "2023_Ziele"
    ]
    target_col = "Aktiv"
    treatment_feature = "Gehaltsentwicklung"

    import math
    max_val = math.ceil(df["Gehaltsentwicklung"].max() / 5) * 5
    thresholds = list(range(0, max_val + 1, 5))
    median_effects, lower_bounds, upper_bounds = [], [], []

    for thresh in thresholds:
        df["Gehaltserhöhung"] = (df[treatment_feature] >= thresh).astype(int)
        X = df[feature_cols]
        T = df["Gehaltserhöhung"].astype(np.int64).values
        y = df[target_col].astype(np.int64).values

        est_temp = CausalForestDML(random_state=42)
        est_temp.fit(Y=y, T=T, X=X)

        te_pred = est_temp.effect(X)
        lb, ub = est_temp.effect_interval(X)

        median_effects.append(np.median(te_pred))
        lower_bounds.append(np.median(lb))
        upper_bounds.append(np.median(ub))

    # Plotly: Kurve mit Konfidenzintervall
    import plotly.graph_objects as go

    fig_curve = go.Figure()

    # Median-Kurve
    fig_curve.add_trace(go.Scatter(
        x=thresholds,
        y=median_effects,
        mode="lines+markers",
        name="Median Treatment Effect",
        line=dict(color="blue"),
        marker=dict(size=6)
    ))

    # Konfidenzintervall (als Fläche)
    fig_curve.add_trace(go.Scatter(
        x=thresholds + thresholds[::-1],
        y=upper_bounds + lower_bounds[::-1],
        fill='toself',
        fillcolor='rgba(173,216,230,0.3)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name="Konfidenzintervall"
    ))

    fig_curve.update_layout(
        title="Median-Behandlungseffekt in Abhängigkeit vom Gehaltsschwellenwert",
        xaxis_title="Schwellenwert Gehaltserhöhung",
        yaxis_title="∆ Bleibewahrscheinlichkeit (Median Treatment Effect)",
        template="plotly_white",
        hovermode="x unified"
    )

    st.plotly_chart(fig_curve, use_container_width=True)

    return est, te_pred, conf_int, fig

@st.cache_data
def calculate_median_effects_over_thresholds(df, feature_cols, target_col, treatment_feature):
    import math
    
    # Arbeite mit einer Kopie des DataFrames
    df_work = df.copy()
    
    max_val = math.ceil(df_work[treatment_feature].max() / 5) * 5
    thresholds = list(range(0, max_val + 1, 5))

    median_effects, lower_bounds, upper_bounds = [], [], []

    for thresh in thresholds:
        df_work["Gehaltserhöhung"] = (df_work[treatment_feature] >= thresh).astype(int)
        X = df_work[feature_cols]
        T = df_work["Gehaltserhöhung"].astype(np.int64).values
        y = df_work[target_col].astype(np.int64).values

        est_temp = CausalForestDML(random_state=42)
        est_temp.fit(Y=y, T=T, X=X)

        te_pred = est_temp.effect(X)
        lb, ub = est_temp.effect_interval(X)

        median_effects.append(np.median(te_pred))
        lower_bounds.append(np.median(lb))
        upper_bounds.append(np.median(ub))

    return thresholds, median_effects, lower_bounds, upper_bounds