# Decision Support System für HR-Management
## Projektziel
Das Ziel dieses Projekts ist die Entwicklung eines Decision Support Systems (DSS), das HR-Abteilungen in zwei zentralen Bereichen unterstützt:
1. **Datenanalyse:** Überblick über den vorliegenden Datensatz
2. **Fluktuationsvorhersage:** Identifikation von Mitarbeitern mit erhöhtem Kündigungsrisiko auf Basis historischer HR-Daten.
3. **Szenariosimulation:** Simulation der Auswirkungen von HR-Maßnahmen (z. B. Gehaltsanpassungen, flexible Arbeitszeitmodelle) auf die Fluktuationsrate.
Das System soll datenbasiert Empfehlungen für die Mitarbeiterbindung und die strategische Personalplanung geben.
## Verwendete Technologien
Das Projekt verwendet folgende Programmiersprache und Bibliotheken:

- **Python**
- **Streamlit** (st)
- **pandas**
- **numpy**
- **scikit-learn** (sklearn)
- **joblib**
- **matplotlib** (plt)
- **plotly** (plotly.graph_objects, plotly.express)
- **shap**
- **scipy**
- **xgboost**
- **lightgbm**
- **econml**
- **os**, **json**, **hashlib**, **datetime**, **time**,**logging**,**warnings** (Standardbibliotheken)

Weitere Bibliotheken können je nach verwendetem Modul oder Notebook benötigt werden.
## Installation
1. Repository klonen:
	```bash
	git clone <REPO-URL>
	cd Thesis
	```
2. Abhängigkeiten installieren:
	```bash
	# Beispiel für Python
	pip install -r requirements.txt
	```
## Nutzung
- Die wichtigsten Skripte und Notebooks:
	- `App.py`: Hauptanwendung
	- `ml_forest.py`, `ml_lightgbm.py`, ...: Verschiedene ML-Modelle
	- `shap_analysis.py`: Modellinterpretation mit SHAP
	- `PUB_CausalML_Gehaltsentwicklung.ipynb`: Beispiel-Notebook zur Gehaltsentwicklung
*Weitere Hinweise zur Ausführung und zu den Daten können hier ergänzt werden.*
## Daten
- Die Analyse basiert auf historischen HR-Daten (z. B. `data_randomized.csv`).
- Bitte sicherstellen, dass die Daten im richtigen Format vorliegen.
# Thesis