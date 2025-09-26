import shap
import matplotlib.pyplot as plt
import streamlit as st


def run_shap_analysis(model, X_test, X_train, model_name="Modell"):
    
    # Index zurücksetzen, damit positional indexing passt
    X_test_reset = X_test.reset_index(drop=True)
    
    st.subheader(f"SHAP Analyse für: {model_name}")
        # Dropdown zur Auswahl der Instanz
    instance_index = st.selectbox(
        "Instanz auswählen (nach Zeilenindex im Testset):",
        options=list(range(len(X_test_reset))),
        index=0
    )
    # TreeExplainer verwenden
    explainer = shap.TreeExplainer(model, X_train)
    shap_values = explainer(X_test)

    # Vorhergesagte Klasse der ausgewählten Instanz (Index bezogen auf X_test_reset)
    pred_probs = model.predict_proba(X_test_reset)
    # pred_class = pred_probs[instance_index].argmax()
    st.write(f"Vorhergesagte Klasse der Instanz #{instance_index}: 1")

    # Neues Explanation-Objekt für vorhergesagte Klasse erstellen
    shap_exp = shap.Explanation(
        values=shap_values.values[instance_index, :, 1],
        base_values=explainer.expected_value[1],
        data=X_test_reset.iloc[instance_index].values,
        feature_names=X_test.columns.tolist()
    )

    
    # Waterfall Plot für die Instanz und vorhergesagte Klasse
    st.write(f"**SHAP Waterfall Plot für Instanz #{instance_index} :**")
    fig_waterfall = plt.figure(figsize=(10, 5))
    shap.plots.waterfall(shap_exp)
    plt.tight_layout()  
    plt.savefig("RandomForest_SHAP_Waterfall.pdf",dpi=600,bbox_inches="tight", pad_inches=0.2)  # Speichert die Grafik als PDF
    st.pyplot(fig_waterfall)
    

    # SHAP Werte für alle Instanzen der vorhergesagten Klasse (Beeswarm Plot)
    shap_exp_class = shap.Explanation(
        values=shap_values.values[:, :, 1],
        base_values=explainer.expected_value[1],
        data=shap_values.data,
        feature_names=X_test.columns.tolist()
    )
    # Beeswarm Plot
    st.write(f"**SHAP Beeswarm Plot:**")
    fig_beeswarm = plt.figure(figsize=(10, 5))
    shap.plots.beeswarm(shap_exp_class)
    plt.tight_layout()
    plt.savefig("RandomForest_SHAP_Beeswarm.pdf",dpi=600)  # Speichert die Grafik als PDF
    st.pyplot(fig_beeswarm)

    # Feature Importance Plot
    st.write(f"**SHAP Feature Importance Plot:**")
    fig_importance = plt.figure(figsize=(10, 5))
    shap.plots.bar(shap_exp_class)
    plt.tight_layout()
    plt.savefig("RandomForest_SHAP_Feature Importance.pdf",dpi=600)  # Speichert die Grafik als PDF
    st.pyplot(fig_importance)

    
    