# 🏎️ Antonio Lobato Garage — Proyecto de Regresión

Proyecto grupal del Bootcamp de Análisis de Datos. **Antonio Lobato Garage** es una aplicación de Streamlit que permite explorar datos de vehículos de segunda mano y estimar su precio con tres modelos de regresión: **Ridge, Random Forest y CatBoost**.

La pregunta que guía el proyecto es: **¿qué precio podemos estimar para un vehículo de segunda mano a partir de características como la marca, el modelo, el año y el kilometraje?** Es un problema de regresión porque el objetivo, `price`, es un valor numérico.

## 1. Cómo ejecutar la demo

Desde la carpeta que contiene `app.py`:

```bash
python -m pip install streamlit pandas numpy plotly scikit-learn catboost joblib
python -m streamlit run app.py
```

Coloca `autos_limpio.csv` junto a `app.py` o sube el CSV desde el panel lateral. El archivo subido tiene prioridad. La aplicación intenta leer primero con separador `;` y, si obtiene una sola columna, vuelve a leer con `,`.

La consigna de entrega pide un repositorio con el **notebook** y una **demo en Streamlit con un dashboard analítico**. Este README documenta el flujo comprobable en `app.py`; el notebook debe acompañar la entrega para mostrar el trabajo realizado fuera de la aplicación.

## 2. Datos y preparación

### Selección y filtrado de datos

El conjunto de datos original contenía **4.006 registros**. Tras analizar la distribución de precios, se identificaron **231 vehículos con un precio superior a 100.000 $ (5,77 % del total)**.

Se decidió excluir estos registros y trabajar con los **3.775 vehículos restantes**, cuyo precio no supera los 100.000 $.

Esta decisión responde a la presencia de valores extremos que ampliaban considerablemente el rango de precios y dificultaban el entrenamiento y la capacidad predictiva de los modelos de Machine Learning. De este modo, se delimitó el ámbito de estudio para mejorar la estabilidad del entrenamiento y reducir la influencia de observaciones excepcionalmente elevadas.

| Columna | Uso en la aplicación |
| --- | --- |
| `price` | Variable objetivo obligatoria. La interfaz muestra los precios con el símbolo `$`. |
| `price_tipo` | Variable objetivo obligatoria limpia. 
| `brand`, `model` | Necesarias para seleccionar el vehículo en el formulario de predicción. |
| `milage` | Kilometraje en millas. El CSV debe conservar la grafía `milage` para que la aplicación reconozca este campo. |
| `model_year` | Año del modelo. Se usa también para calcular la antigüedad. |
| `engine` | Descripción del motor. Si está disponible, se extraen potencia en HP, cilindrada en litros, número de cilindros y arquitectura. |


El siguiente fragmento muestra cómo se convierte `price` a número y se descartan los registros que no sirven como ejemplos de entrenamiento:

```python
df["price"] = pd.to_numeric(
    df["price"].astype(str).str.replace(r"[$,]", "", regex=True),
    errors="coerce"
)

df = df.dropna(subset=["price"]).drop_duplicates().copy()
df = df[df["price"] > 0].copy()
```

`errors="coerce"` convierte en valores ausentes los precios que no se pueden interpretar. Después se eliminan esos registros, los duplicados y los precios no positivos. Si existe `model_year`, se calcula `vehicle_age = 2026 - model_year`: **2026 es una referencia fija en el código**. Todas las columnas resultantes salvo `price` se usan como entradas de los modelos.

La fuente de los datos ha sido [Regression of Used Car Prices](https://www.kaggle.com/competitions/playground-series-s4e9). `app.py` lee un CSV local o subido por el usuario.

## 3. Análisis exploratorio: distribución y dispersión

El **Dashboard** resume el número de vehículos y el precio medio, mediano y máximo. El histograma permite ver dónde se concentran los precios y si la distribución presenta una cola hacia valores altos. Si el CSV contiene `model_year`, también se muestra el precio mediano por año del modelo.

```python
fig = px.histogram(
    df, x="price", nbins=50,
    title="Distribución de precios",
    labels={"price": "Precio ($)", "count": "Vehículos"}
)
st.plotly_chart(fig, use_container_width=True)
```

El gráfico de **kilometraje frente a precio** representa cada vehículo como un punto. Sirve para observar si los precios tienden a variar con el kilometraje, cuánto se dispersan para valores similares y dónde aparecen casos alejados del resto. El gráfico permite explorar una asociación; no demuestra que el kilometraje explique por sí solo las diferencias de precio.

```python
if "milage" in df.columns:
    fig = px.scatter(
        df, x="milage", y="price", opacity=0.5,
        title="Kilometraje frente a precio",
        labels={"milage": "Kilometraje (millas)", "price": "Precio ($)"}
    )
    st.plotly_chart(fig, use_container_width=True)
```

La pestaña **Exploración de datos** permite filtrar la tabla por marca y año cuando esas columnas existen. El resumen de valores ausentes y las estadísticas descriptivas corresponden al conjunto preparado completo, no a la selección filtrada. La app revisada no incluye una matriz de correlaciones.

## 4. Entrenamiento de los modelos

### 4.1. Separación de entrenamiento y prueba

Primero se separan las variables de entrada (`X`) del precio (`y`). La división reserva un **20 % para prueba** y utiliza la semilla `42` para reproducir la partición con los mismos datos.

```python
X = df.drop(columns=["price"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
```

### 4.2. Filtrado de precios atípicos mediante IQR

Los límites se calculan **solo con los precios de entrenamiento**. Luego se aplican a ambos conjuntos. El fragmento muestra cómo se obtiene cada máscara; después, el código la usa para seleccionar las filas correspondientes en `X` e `y`.

```python
q1 = y_train.quantile(0.25)
q3 = y_train.quantile(0.75)
iqr = q3 - q1

low = q1 - 1.5 * iqr
high = q3 + 1.5 * iqr

mask_train = y_train.between(low, high)
mask_test = y_test.between(low, high)
```

Calcular los límites con `y_train` evita usar los precios de prueba para fijar el umbral. **Aun así, se excluyen casos de prueba según su precio real**: las métricas describen el rendimiento dentro del rango retenido, no sobre todos los vehículos ni sobre los precios extremos. La app muestra cuántas filas quedan antes y después del filtrado.

### 4.3. Preprocesamiento y ajuste

| Modelo | Tratamiento de las variables |
| --- | --- |
| **Ridge** | Imputa los valores numéricos con la mediana, los estandariza e imputa y codifica las categorías mediante *one-hot encoding*. Usa `alpha=1`. |
| **Random Forest** | Imputa los valores numéricos con la mediana y aplica *one-hot encoding* a las categorías tras imputarlas. No estandariza los valores numéricos. Usa 300 árboles y una profundidad máxima de 15. |
| **CatBoost** | Recibe las variables categóricas directamente. Las convierte a texto y sustituye sus ausentes por `Desconocido`. Usa 1.000 iteraciones, profundidad 6 y tasa de aprendizaje `0.05`. |

En Ridge y Random Forest, el preprocesamiento forma parte del `Pipeline` y se ajusta al llamar a `fit` sobre el conjunto de entrenamiento filtrado. Este es el pipeline de Ridge:

```python
Pipeline([
    ("preprocessor", preprocessor_ridge),
    ("modelo", TransformedTargetRegressor(
        regressor=Ridge(alpha=1),
        func=np.log1p,
        inverse_func=np.expm1
    ))
])
```

Los tres modelos aprenden sobre `log1p(price)`, una transformación que comprime las diferencias entre precios altos. En Ridge y Random Forest, `TransformedTargetRegressor` devuelve automáticamente las predicciones a la escala original. En CatBoost se hace de forma explícita:

```python
catboost_model.fit(
    X_train_cb,
    np.log1p(y_train_f),
    cat_features=cat_cols_cb
)

pred_log = catboost_model.predict(X_test_cb)
pred_cb = np.maximum(0, np.expm1(pred_log))
```

## 5. Evaluación y dispersión: precio real frente a predicho

Los tres modelos se comparan sobre **el mismo conjunto de prueba filtrado**. Las métricas se calculan en la escala original del precio. Este fragmento corresponde al bucle de evaluación de Ridge y Random Forest; CatBoost se evalúa con las mismas métricas.

```python
modelo.fit(X_train_f, y_train_f)
pred = modelo.predict(X_test_f)

resultados.append({
    "Modelo": nombre,
    "MAE": mean_absolute_error(y_test_f, pred),
    "RMSE": np.sqrt(mean_squared_error(y_test_f, pred)),
    "R²": r2_score(y_test_f, pred)
})
```

| Métrica | Cómo interpretarla |
| --- | --- |
| **MAE** | Error absoluto medio, en unidades de precio. Cuanto menor, mejor. La aplicación ordena la tabla por esta métrica. |
| **RMSE** | Raíz del error cuadrático medio, también en unidades de precio. Da más peso a los errores grandes. |
| **R²** | Mide el rendimiento frente a una referencia que predice el precio medio del conjunto evaluado. Puede ser negativo. |

El segundo gráfico de dispersión responde a una pregunta distinta de la del EDA: **¿cuánto se acercan las predicciones a los precios reales?** El eje X muestra el precio real y el eje Y el predicho. La línea discontinua `y = x` representa una predicción exacta: por encima, el modelo sobreestima; por debajo, subestima.

```python
comparacion = pd.DataFrame({
    "Precio real": real,
    "Precio predicho": pred
})

fig = px.scatter(
    comparacion, x="Precio real", y="Precio predicho",
    title=f"{nombre_modelo}: precio real frente a predicho",
    opacity=0.55
)
limite = max(comparacion.max().max(), 1)
fig.add_shape(
    type="line", x0=0, y0=0, x1=limite, y1=limite,
    line=dict(dash="dash")
)
```

La pestaña **Comparativa ML** destaca el modelo con menor MAE para el CSV cargado. No hay un ganador fijo ni cifras de rendimiento codificadas. El gráfico permite detectar patrones de sobreestimación o subestimación, aunque la app no muestra un gráfico independiente de residuos ni una visualización de importancia de variables.

## 6. Demo de predicción y persistencia

En **Predecir precio**, el usuario elige el algoritmo —CatBoost por defecto—, la marca y el modelo. También puede modificar el año y el kilometraje si esos campos están disponibles. Para las demás características, la app toma la mediana de las variables numéricas y el valor más frecuente de las categóricas entre los vehículos de la misma marca y modelo.

Los modelos y sus resultados se guardan con Joblib en `modelos/autoprice_modelos.pckl`. La app reutiliza el paquete si coinciden su versión y la huella SHA-256 del CSV. El botón **Reentrenar modelos ahora** borra el archivo guardado y reinicia la ejecución, pero no limpia la caché interna de `entrenar_modelos`, por lo que puede reutilizar el ajuste en memoria.

La predicción es **orientativa**, no una tasación. Además, en la implementación actual, editar el año en el formulario no recalcula `vehicle_age`: el año introducido y la antigüedad usada por el modelo pueden quedar desalineados.

## 7. Correspondencia con la rúbrica del Bootcamp

| Nivel | Evidencia en `app.py` | Qué queda sin acreditar con este archivo |
| --- | --- | --- |
| **Esencial** | Regresión de precios, EDA con histograma y gráficos de dispersión, dashboard, demo de predicción, MAE, RMSE, R² y gráfico de precio real frente a predicho. | No se acredita el **overfitting inferior al 5 %**. Tampoco se muestran una matriz de correlaciones, importancia de variables ni un gráfico específico de residuos en la app. |
| **Medio** | Dos modelos de tipo *ensemble*: Random Forest y CatBoost. La app admite la carga manual de un CSV. | No se acreditan validación cruzada, búsqueda de hiperparámetros, recogida de feedback ni un pipeline de ingestión de nuevos registros. Cargar un CSV manualmente no equivale a ese pipeline. |
| **Avanzado** | Persistencia local de modelos con Joblib. | No se acreditan Docker, almacenamiento en base de datos, despliegue ni tests unitarios. |
| **Experto** | Reutilización del paquete local cuando coinciden la versión y el CSV. | No se acreditan A/B testing, monitorización de *data drift* ni sustitución automática de modelos basada en métricas. |

**Comprobación del umbral de overfitting.** La comparativa actual solo calcula métricas de prueba, no la diferencia entre entrenamiento y validación que pide la rúbrica. Para documentar ese requisito en el notebook, hay que indicar **qué métrica se compara y cómo se calcula la diferencia porcentual**, mostrar ambos valores y reservar la partición de prueba para la evaluación final. Por ejemplo, si el criterio acordado es el incremento relativo del MAE, se puede calcular `100 * (MAE_validación - MAE_entrenamiento) / MAE_entrenamiento`, siempre que el MAE de entrenamiento sea mayor que cero. Esta fórmula es una propuesta de comprobación, no un resultado del proyecto.

La entrega demuestra las dos competencias de la consigna mediante evidencias distintas: el **dashboard y la exploración** sirven para evaluar el conjunto de datos, mientras que el **entrenamiento, la comparación de modelos y la demo** muestran cómo se aplica el aprendizaje automático a un problema de regresión.
