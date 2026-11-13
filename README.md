# Humedad relativa del aire en Barranquilla - árbol de decisión

Proyecto final. Clasificamos la humedad relativa del aire de Barranquilla en Baja (menos de 75 %), Media (75 a 88 %) o Alta (88 % o más) con un árbol de decisión (`DecisionTreeClassifier` de scikit-learn).

Datos: "Humedad del Aire" del IDEAM en datos.gov.co (estación Escuela Naval Barranquilla).
https://www.datos.gov.co/Ambiente-y-Desarrollo-Sostenible/Humedad-del-Aire/uext-mhny

## Archivos

- `data/Humedad_del_Aire_20261028.csv`: los datos, tal cual se descargaron del portal.
- `notebooks/01_humedad_original.ipynb`: primer notebook. Limpieza y árbol usando la humedad.
- `notebooks/02_humedad_arbol_hora_mes.ipynb`: árbol que predice la clase con la hora y el mes.
- `Arbol.py`: el mismo proceso del notebook 02 en un script (`python Arbol.py`). Da el mismo resultado y genera `web/model.js`.
- `web/`: página para probar el modelo (abrir `index.html`).

## Qué hicimos

En el notebook 01 el árbol usa la humedad para predecir la clase, y como la clase sale de la misma humedad da 100 %, pero no predice nada nuevo.

En el notebook 02 dejamos la misma limpieza y las mismas clases, y el árbol usa la hora y el mes, que salen de la columna `FechaObservacion`. Con `max_depth=4` (16 hojas) acierta el 58 % de los datos de prueba, contra 47 % de decir siempre "Media". La hora es la que más pesa (92 %): la humedad baja en la tarde y sube en la noche y la madrugada.

## Cómo correrlo

En Colab: subir el CSV a `/content`, abrir el notebook 02 y ejecutar todo.
En Jupyter: dejar el CSV en la misma carpeta del notebook o en `data/`.

Librerías: pandas, numpy, matplotlib, scikit-learn.

Página: https://gabrieldevco.github.io/humedad-barranquilla/
