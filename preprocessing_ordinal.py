import pandas as pd
# Definiere das Mapping für die ordinalen Spalten
ordinal_cols_mapping = [
    {"col": "Alter", "mapping": {
        'Unter 18': 0, '18-24': 1, '25-30': 2, '31-40': 3,
        '41-50': 4, '51-60': 5, 'Über 60': 6
    }},
    {"col": "Betriebszugehoerigkeit", "mapping": {
        '0-2': 0, '"3-5"': 1, '"6-8"': 2, '"9-11"': 3, '"12-14"': 4,
        '15-17': 5, '18-20': 6, '21-23': 7, '24-26': 8,
        '27-29': 9, '30-32': 10, '33-35': 11, '36-38': 12,
        '39-41': 13, '42-44': 14, '45-47': 15, '48-50': 16
    }},
    {"col": "Zeit_auf_Position", "mapping": {
        '0-2': 0, '"3-5"': 1, '"6-8"': 2, '"9-11"': 3, '"12-14"': 4,
        '15-17': 5, '18-20': 6
    }},
    {"col": "2023_Ziele", "mapping": {
        'Unrated': 0, 'Unterdurchschnitt': 1,
        'Durchschnitt': 2, 'Überdurchschnitt': 3
    }},
    {"col": "2023_Pers_Beurteilung", "mapping": {
        'Unrated': 0, 'Unterdurchschnitt': 1,
        'Durchschnitt': 2, 'Überdurchschnitt': 3
    }},
    {"col": "2024_Ziele", "mapping": {
        'Unrated': 0, 'Unterdurchschnitt': 1,
        'Durchschnitt': 2, 'Überdurchschnitt': 3
    }},
    {"col": "2024_Pers_Beurteilung", "mapping": {
        'Unrated': 0, 'Unterdurchschnitt': 1,
        'Durchschnitt': 2, 'Überdurchschnitt': 3
    }},
    {"col": "Krankenstand_2023", "mapping": {
        'kleiner 1%': 0, '1,0 - 3%': 1, '3,1 - 5%': 2,
        '5,1% - 10%': 3, '10,1% - 15%': 4, '15,1 - 20%': 5,
        '20,01 - 40%': 6, '40,01 - 60 %': 7,
        '60,01 - 75%': 8, '75,01 - 100 %': 9
    }},
    {"col": "Krankenstand_2024", "mapping": {
        'kleiner 1%': 0,'1,0 - 3%': 1, '3,1 - 5%': 2, '5,1% - 10%': 3,
        '10,1% - 15%': 4, '15,1 - 20%': 5, '20,01 - 40%': 6,
        '40,01 - 60 %': 7, '60,01 - 75%': 8,
        '75,01 - 100 %': 9
    }},
    {"col": "Gehaltsentwicklung", "mapping": {
        '"-100%-96%"': 0, '"-95%-91%"': 1, '"-90%-86%"': 2,
        '"-85%-81%"': 3, '"-80%-76%"': 4, '"-75%-71%"': 5,
        '"-70%-66%"': 6, '"-65%-61%"': 7, '"-60%-56%"': 8,
        '"-55%-51%"': 9, '"-50%-46%"': 10, '"-45%-41%"': 11,
        '"-40%-36%"': 12, '"-35%-31%"': 13, '"-30%-26%"': 14,
        '"-25%-21%"': 15, '"-20%-16%"': 16, '"-15%-11%"': 17,
        '"-10%-6%"': 18, '"-5%-1%"': 19, '0%-4%': 20,
        '5%-9%': 21, '10%-14%': 22, '15%-19%': 23,
        '20%-24%': 24, '25%-29%': 25, '30%-34%': 26,
        '35%-39%': 27, '40%-44%': 28, '45%-49%': 29,
        '50%-54%': 30, '55%-59%': 31, '60%-64%': 32,
        '65%-69%': 33, '70%-74%': 34, '75%-79%': 35,
        '80%-84%': 36, '85%-89%': 37, '90%-94%': 38,
        '95%-100%': 39
    }}
]


# Nominale Features (ohne natürliche Ordnung)
nominal_cols = ["Geschlecht", "Bereich", "FTE", "FK"]

def apply_ordinal_encoding(df: pd.DataFrame) -> pd.DataFrame:
    df_encoded = df.copy()

    # Liste der zu behaltenden Spalten
    target_cols = [item["col"] for item in ordinal_cols_mapping]
    keep_cols = target_cols + nominal_cols + ["Aktiv"]
    df_encoded = df_encoded[[col for col in keep_cols if col in df_encoded.columns]]

    # --- Ordinale Codierung ---
    for item in ordinal_cols_mapping:
        col = item["col"]
        mapping = item["mapping"]

        if col in df_encoded.columns:
            df_encoded[f"{col}_vorhanden"] = df_encoded[col].notna().astype(int)
            df_encoded[col] = df_encoded[col].astype(str).map(mapping)
            df_encoded[col] = df_encoded[col].fillna(-1)
            df_encoded[col] = df_encoded[col].replace("nan", -1)
            df_encoded[col] = df_encoded[col].astype(int)
    # Konstante Spalten entfernen
    constant_cols = [col for col in df_encoded.columns if df_encoded[col].nunique() == 1]
    df_encoded = df_encoded.drop(columns=constant_cols)
    # --- One-Hot-Encoding für nominale Features ---
    for col in nominal_cols:
        if col in df_encoded.columns:
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=False)
            # Boolean → int
            dummies = dummies.astype(int)
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)

    # Zielspalte
    if "Aktiv" in df_encoded.columns:
        df_encoded["Aktiv"] = df_encoded["Aktiv"].astype(int)

    return df_encoded