# Proyecto final - Humedad relativa del aire en Barranquilla
# Gabriel Padilla y Juan Noguera
# arbol de decision que predice si la humedad es Baja, Media o Alta con la hora y el mes
# datos: datos.gov.co (IDEAM) - Humedad del Aire

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# busca el csv en la carpeta actual o en /content (colab)
archivo = "Humedad_del_Aire_20261028.csv"
for c in [".", "data", "../data", "/content"]:
    if os.path.exists(c + "/" + archivo):
        archivo = c + "/" + archivo
        break

# limpieza
df = pd.read_csv(archivo)
df = df[df["DescripcionSensor"] == "Humedad del aire 2 mt"].reset_index(drop=True)   # el archivo tambien trae humedad del suelo
df["Municipio"] = df["Municipio"].str.upper()
df = df.drop_duplicates().reset_index(drop=True)
df = df[df["ValorObservado"] > 0].reset_index(drop=True)   # humedad en 0 es error del sensor
df["FechaObservacion"] = pd.to_datetime(df["FechaObservacion"], format="%Y %b %d %I:%M:%S %p", errors="coerce")
df["ANIO"] = df["FechaObservacion"].dt.year
df["NUM_MES"] = df["FechaObservacion"].dt.month
df = df.dropna(subset=["FechaObservacion"]).reset_index(drop=True)
df = df[["FechaObservacion", "ANIO", "NUM_MES", "NombreEstacion", "Municipio", "ValorObservado"]]
df = df.rename(columns={"ValorObservado": "HUMEDAD"})
df["HORA"] = df["FechaObservacion"].dt.hour
df_completo = df.copy()

# muestra de 10000 datos y clases: Baja < 75, Media de 75 a 88, Alta >= 88
df = df.sample(n=10000, random_state=42).sort_values("FechaObservacion").reset_index(drop=True)
df["CLASE"] = ["Baja" if h < 75 else "Media" if h < 88 else "Alta" for h in df["HUMEDAD"]]
print(df["CLASE"].value_counts())

# arbol del colab: usa la misma humedad para sacar la clase, por eso da 100% (solo para comparar)
X_train, X_test, y_train, y_test = train_test_split(df[["HUMEDAD"]], df["CLASE"], test_size=0.25, random_state=42, stratify=df["CLASE"])
arbol_original = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
precision_original = accuracy_score(y_test, arbol_original.predict(X_test))

# arbol con la hora y el mes, 75% para entrenar y 25% para probar
FEATURES = ["HORA", "NUM_MES"]
X_train, X_test, y_train, y_test = train_test_split(df[FEATURES], df["CLASE"], test_size=0.25, random_state=42, stratify=df["CLASE"])
arbol = DecisionTreeClassifier(max_depth=4, random_state=42)   # con mas profundidad casi no mejora
arbol.fit(X_train, y_train)
y_pred = arbol.predict(X_test)

# resultados (lo comparamos con decir siempre la clase mas comun)
base = y_train.value_counts().idxmax()
orden = ["Baja", "Media", "Alta"]
print("Precisión árbol original:", precision_original)
print("Siempre", base, ":", accuracy_score(y_test, [base] * len(y_test)))
print("Precisión:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred, labels=orden))   # filas: real, columnas: prediccion
print("Importancia de HORA:", round(arbol.feature_importances_[0], 3))
print("Importancia de NUM_MES:", round(arbol.feature_importances_[1], 3))

plt.figure(figsize=(40, 12))
plot_tree(arbol, feature_names=FEATURES, class_names=arbol.classes_, filled=True, fontsize=11)
plt.show()

# exporto el arbol y los datos para la pagina (model.js)
t = arbol.tree_
hum_hoja = pd.Series(df.loc[X_train.index, "HUMEDAD"].values).groupby(arbol.apply(X_train)).mean()

def nodo(i):
    p = t.value[i][0] / t.value[i][0].sum()
    n = {"n": int(t.n_node_samples[i]), "p": [round(float(x), 3) for x in p], "g": round(float(t.impurity[i]), 3),
         "v": [int(round(float(x) * t.n_node_samples[i])) for x in p]}
    if t.children_left[i] != -1:
        n.update(f=FEATURES[t.feature[i]], t=round(float(t.threshold[i]), 2), l=nodo(t.children_left[i]), r=nodo(t.children_right[i]))
    else:
        n["h"] = round(float(hum_hoja[i]), 1)
    return n

# por cada mes y hora: promedio real, cuantos datos hay y % de baja, media y alta
reales = []
for m in range(1, 13):
    fila = []
    for h in range(24):
        d = df_completo[(df_completo["NUM_MES"] == m) & (df_completo["HORA"] == h)]["HUMEDAD"]
        fila.append([round(float(d.mean()), 1), int(len(d)), round(float((d < 75).mean() * 100)),
                     round(float(((d >= 75) & (d < 88)).mean() * 100)), round(float((d >= 88).mean() * 100))])
    reales.append(fila)

modelo = {
    "meta": {"fuente": "IDEAM vía datos.gov.co: Humedad del Aire (uext-mhny)", "estacion": "Escuela Naval Barranquilla",
             "desde": str(df["FechaObservacion"].min().date()), "hasta": str(df["FechaObservacion"].max().date()),
             "registros": len(df), "train": len(X_train), "test": len(X_test)},
    "clases": list(arbol.classes_), "cortes": [75, 88],
    "precision": round(accuracy_score(y_test, y_pred), 3),
    "precision_base": round(accuracy_score(y_test, [base] * len(y_test)), 3),
    "precision_original": round(precision_original, 3),
    "matriz": confusion_matrix(y_test, y_pred, labels=orden).tolist(), "orden": orden,
    "importancias": {f: round(float(v), 3) for f, v in zip(FEATURES, arbol.feature_importances_)},
    "profundidad": int(arbol.get_depth()), "hojas": int(arbol.get_n_leaves()),
    "arbol": nodo(0),
    "promedio_hora": [round(float(x), 1) for x in df.groupby("HORA")["HUMEDAD"].mean()],
    "reales": reales,
    "prueba": [[int(h), int(m), orden.index(r), orden.index(p)] for h, m, r, p in zip(X_test["HORA"], X_test["NUM_MES"], y_test, y_pred)],
}
salida = "web/model.js" if os.path.exists("web") else "pagina_web/model.js" if os.path.exists("pagina_web") else "model.js"
with open(salida, "w", encoding="utf-8") as f:
    f.write("window.MODEL=" + json.dumps(modelo, ensure_ascii=False) + ";\n")
print("guardado en", salida)
