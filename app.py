import re
import io
import hashlib
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor

# ---------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------

st.set_page_config(
    page_title="AutoPrice AI",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 AutoPrice AI")
st.caption("Exploración de datos y predicción del precio de vehículos")

# ---------------------------------------------------------
# CARGA Y PREPARACIÓN DE DATOS
# ---------------------------------------------------------

def extraer_potencia(texto):
    if pd.isna(texto):
        return np.nan
    m = re.search(r"(\d+(?:\.\d+)?)\s*HP", str(texto), re.I)
    return float(m.group(1)) if m else np.nan


def extraer_cilindrada(texto):
    if pd.isna(texto):
        return np.nan
    m = re.search(r"(\d+(?:\.\d+)?)\s*L\b", str(texto), re.I)
    return float(m.group(1)) if m else np.nan


def extraer_cilindros(texto):
    if pd.isna(texto):
        return np.nan
    m = re.search(r"(\d+)\s*Cylinder", str(texto), re.I)
    return float(m.group(1)) if m else np.nan


def extraer_arquitectura(texto):
    if pd.isna(texto):
        return "Desconocido"
    texto = str(texto).lower()
    if "straight" in texto:
        return "Straight"
    if "flat" in texto or "boxer" in texto:
        return "Flat"
    m = re.search(r"\bv\d+\b", texto)
    return m.group().upper() if m else "Otra"


def preparar_datos(df):
    df = df.copy()

    if "price" not in df.columns:
        raise ValueError("El CSV debe contener una columna llamada 'price'.")

    df["price"] = pd.to_numeric(
        df["price"].astype(str).str.replace(r"[$,]", "", regex=True),
        errors="coerce"
    )

    if "milage" in df.columns:
        df["milage"] = pd.to_numeric(
            df["milage"].astype(str).str.replace(r"[^\d.]", "", regex=True),
            errors="coerce"
        )

    if "model_year" in df.columns:
        df["model_year"] = pd.to_numeric(df["model_year"], errors="coerce")
        df["vehicle_age"] = 2026 - df["model_year"]

    if "engine" in df.columns:
        df["engine_hp"] = df["engine"].apply(extraer_potencia)
        df["engine_liters"] = df["engine"].apply(extraer_cilindrada)
        df["engine_cylinders"] = df["engine"].apply(extraer_cilindros)
        df["engine_architecture"] = df["engine"].apply(extraer_arquitectura)

    df = df.dropna(subset=["price"]).drop_duplicates().copy()
    df = df[df["price"] > 0].copy()

    return df


@st.cache_data(show_spinner=False)
def cargar_csv(file_bytes):
    # El notebook original utiliza separador punto y coma.
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), sep=";")

        if len(df.columns) == 1:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=",")

        return preparar_datos(df)

    except Exception as e:
        raise ValueError(f"No se ha podido leer el CSV: {e}")


# ---------------------------------------------------------
# ENTRENAMIENTO DE LOS MODELOS
# ---------------------------------------------------------

@st.cache_resource(show_spinner="Entrenando los modelos. La primera ejecución puede tardar...")
def entrenar_modelos(csv_bytes):

    df = cargar_csv(csv_bytes)

    X = df.drop(columns=["price"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Aplicamos el criterio IQR únicamente calculado con train.
    q1 = y_train.quantile(0.25)
    q3 = y_train.quantile(0.75)
    iqr = q3 - q1

    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr

    mask_train = y_train.between(low, high)
    mask_test = y_test.between(low, high)

    X_train_f = X_train.loc[mask_train].copy()
    y_train_f = y_train.loc[mask_train].copy()

    X_test_f = X_test.loc[mask_test].copy()
    y_test_f = y_test.loc[mask_test].copy()

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_process = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=10))
    ])

    preprocessor_ridge = ColumnTransformer([
        ("num", num_ridge, numeric_cols),
        ("cat", cat_process, categorical_cols)
    ])

    num_tree = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    cat_tree = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=10))
    ])

    preprocessor_tree = ColumnTransformer([
        ("num", num_tree, numeric_cols),
        ("cat", cat_tree, categorical_cols)
    ])

    modelos = {
        "Ridge": Pipeline([
            ("preprocessor", preprocessor_ridge),
            ("modelo", TransformedTargetRegressor(
                regressor=Ridge(alpha=1),
                func=np.log1p,
                inverse_func=np.expm1
            ))
        ]),

        "Random Forest": Pipeline([
            ("preprocessor", preprocessor_tree),
            ("modelo", TransformedTargetRegressor(
                regressor=RandomForestRegressor(
                    n_estimators=300,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1
                ),
                func=np.log1p,
                inverse_func=np.expm1
            ))
        ])
    }

    resultados = []
    predicciones = {}
    modelos_entrenados = {}

    for nombre, modelo in modelos.items():

        modelo.fit(X_train_f, y_train_f)
        pred = modelo.predict(X_test_f)

        predicciones[nombre] = pred
        modelos_entrenados[nombre] = modelo

        resultados.append({
            "Modelo": nombre,
            "MAE": mean_absolute_error(y_test_f, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test_f, pred)),
            "R²": r2_score(y_test_f, pred)
        })

    # CatBoost trabaja directamente con las variables categóricas.
    X_train_cb = X_train_f.copy()
    X_test_cb = X_test_f.copy()

    cat_cols_cb = X_train_cb.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    for col in cat_cols_cb:
        X_train_cb[col] = X_train_cb[col].fillna("Desconocido").astype(str)
        X_test_cb[col] = X_test_cb[col].fillna("Desconocido").astype(str)

    catboost_model = CatBoostRegressor(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=5,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
        allow_writing_files=False
    )

    catboost_model.fit(
        X_train_cb,
        np.log1p(y_train_f),
        cat_features=cat_cols_cb
    )

    pred_log = catboost_model.predict(X_test_cb)
    pred_cb = np.maximum(0, np.expm1(pred_log))

    predicciones["CatBoost"] = pred_cb
    modelos_entrenados["CatBoost"] = catboost_model

    resultados.append({
        "Modelo": "CatBoost",
        "MAE": mean_absolute_error(y_test_f, pred_cb),
        "RMSE": np.sqrt(mean_squared_error(y_test_f, pred_cb)),
        "R²": r2_score(y_test_f, pred_cb)
    })

    metricas = pd.DataFrame(resultados).sort_values("MAE")

    return {
        "df": df,
        "X_columns": X.columns.tolist(),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "X_test": X_test_f,
        "y_test": y_test_f,
        "predicciones": predicciones,
        "modelos": modelos_entrenados,
        "metricas": metricas,
        "filas_train_original": len(y_train),
        "filas_train_filtrado": len(y_train_f),
        "filas_test_original": len(y_test),
        "filas_test_filtrado": len(y_test_f)
    }


# ---------------------------------------------------------
# SELECCIÓN DEL DATASET
# ---------------------------------------------------------

# ---------------------------------------------------------
# CARGA O ENTRENAMIENTO CON PERSISTENCIA JOBLIB
# ---------------------------------------------------------

RUTA_MODELOS = os.path.join("modelos", "autoprice_modelos.pckl")
VERSION_MODELO = 1


@st.cache_resource(show_spinner="Comprobando modelos guardados...")
def cargar_o_entrenar_modelos(csv_bytes):
    """Carga el paquete guardado si corresponde a este CSV; si no, entrena y guarda."""

    hash_csv = hashlib.sha256(csv_bytes).hexdigest()

    # 1. Si existe un archivo guardado y pertenece exactamente a este CSV,
    # recuperamos los modelos sin volver a entrenarlos.
    if os.path.exists(RUTA_MODELOS):
        try:
            paquete = joblib.load(RUTA_MODELOS)

            if (
                paquete.get("version") == VERSION_MODELO
                and paquete.get("hash_csv") == hash_csv
                and "resultados" in paquete
            ):
                st.sidebar.success("Modelos cargados desde disco (Joblib).")
                return paquete["resultados"]

        except Exception as e:
            st.sidebar.warning(
                f"No se pudo cargar el archivo de modelos guardado. "
                f"Se volverán a entrenar. Detalle: {e}"
            )

    # 2. Si no existe o el CSV ha cambiado, entrenamos los modelos.
    resultados_nuevos = entrenar_modelos(csv_bytes)

    # 3. Guardamos los modelos, métricas y datos auxiliares para reutilizarlos.
    try:
        os.makedirs(os.path.dirname(RUTA_MODELOS), exist_ok=True)

        paquete = {
            "version": VERSION_MODELO,
            "hash_csv": hash_csv,
            "resultados": resultados_nuevos
        }

        joblib.dump(paquete, RUTA_MODELOS, compress=3)

        st.sidebar.success(
            f"Modelos entrenados y guardados en {RUTA_MODELOS}"
        )

    except Exception as e:
        st.sidebar.warning(
            f"Los modelos funcionan, pero no se pudieron guardar: {e}"
        )

    return resultados_nuevos


st.sidebar.header("📂 Datos")

archivo = st.sidebar.file_uploader(
    "Sube el archivo autos_limpio.csv",
    type=["csv"]
)

if archivo is not None:
    csv_bytes = archivo.getvalue()

else:
    try:
        with open("autos_limpio.csv", "rb") as f:
            csv_bytes = f.read()

        st.sidebar.success("Usando autos_limpio.csv del proyecto")

    except FileNotFoundError:
        st.info(
            "Para empezar, sube el archivo **autos_limpio.csv** desde el panel lateral. "
            "El notebook utiliza ese CSV como dataset."
        )
        st.stop()


if st.sidebar.button("Reentrenar modelos ahora"):

    cargar_o_entrenar_modelos.clear()

    if os.path.exists(RUTA_MODELOS):
        os.remove(RUTA_MODELOS)

    st.rerun()


try:
    resultados = cargar_o_entrenar_modelos(csv_bytes)

except Exception as e:
    st.error(f"Error al preparar los datos o entrenar los modelos: {e}")
    st.stop()


df = resultados["df"]


# ---------------------------------------------------------
# NAVEGACIÓN
# ---------------------------------------------------------

seccion = st.sidebar.radio(
    "Navegación",
    ["Dashboard", "Exploración de datos", "Comparativa ML", "Predecir precio"]
)


# ---------------------------------------------------------
# 1. DASHBOARD
# ---------------------------------------------------------

if seccion == "Dashboard":

    st.header("Resumen del mercado de vehículos")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Vehículos", f"{len(df):,}")
    c2.metric("Precio medio", f"${df['price'].mean():,.0f}")
    c3.metric("Precio mediano", f"${df['price'].median():,.0f}")
    c4.metric("Precio máximo", f"${df['price'].max():,.0f}")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df, x="price", nbins=50,
            title="Distribución de precios",
            labels={"price": "Precio ($)", "count": "Vehículos"}
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "model_year" in df.columns:

            year_price = df.groupby("model_year", as_index=False)["price"].median()

            fig = px.line(
                year_price, x="model_year", y="price",
                title="Precio mediano según año",
                labels={"model_year": "Año", "price": "Precio mediano ($)"}
            )

            st.plotly_chart(fig, use_container_width=True)

    if "milage" in df.columns:

        fig = px.scatter(
            df, x="milage", y="price", opacity=0.5,
            title="Kilometraje frente a precio",
            labels={"milage": "Kilometraje (millas)", "price": "Precio ($)"}
        )

        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# 2. EXPLORACIÓN
# ---------------------------------------------------------

elif seccion == "Exploración de datos":

    st.header("Exploración del dataset")

    st.write("Dimensiones:", df.shape[0], "filas y", df.shape[1], "columnas")

    filtro_df = df.copy()

    if "brand" in df.columns:

        marcas = sorted(df["brand"].dropna().astype(str).unique())

        seleccion = st.multiselect("Filtrar por marca", marcas, default=[])

        if seleccion:
            filtro_df = filtro_df[filtro_df["brand"].astype(str).isin(seleccion)]

    if "model_year" in df.columns:

        years = df["model_year"].dropna()

        if not years.empty:

            minimo, maximo = int(years.min()), int(years.max())

            rango = st.slider(
                "Año del modelo",
                minimo,
                maximo,
                (minimo, maximo)
            )

            filtro_df = filtro_df[
                filtro_df["model_year"].between(rango[0], rango[1])
            ]

    st.metric("Vehículos tras aplicar filtros", f"{len(filtro_df):,}")

    st.dataframe(filtro_df, use_container_width=True)

    st.subheader("Valores ausentes")

    ausentes = df.isna().sum().sort_values(ascending=False)
    ausentes = ausentes[ausentes > 0]

    if len(ausentes):

        st.dataframe(
            ausentes.rename("Valores ausentes").to_frame(),
            use_container_width=True
        )

    else:
        st.success("No hay valores ausentes en el dataset preparado.")

    st.subheader("Estadísticas descriptivas")

    st.dataframe(df.describe(include="all").T, use_container_width=True)


# ---------------------------------------------------------
# 3. COMPARATIVA DE MODELOS
# ---------------------------------------------------------

elif seccion == "Comparativa ML":

    st.header("Comparativa de modelos de Machine Learning")

    st.caption(
        "Los tres modelos se entrenan con la misma partición y el mismo filtrado IQR."
    )

    st.dataframe(
        resultados["metricas"].style.format({
            "MAE": "${:,.2f}",
            "RMSE": "${:,.2f}",
            "R²": "{:.4f}"
        }),
        use_container_width=True
    )

    mejor = resultados["metricas"].iloc[0]

    st.success(
        f"Menor MAE: {mejor['Modelo']} — ${mejor['MAE']:,.2f}"
    )

    fig = px.bar(
        resultados["metricas"],
        x="Modelo", y="MAE", color="Modelo",
        title="Comparación del error absoluto medio (MAE)",
        labels={"MAE": "MAE ($)"}
    )

    st.plotly_chart(fig, use_container_width=True)

    nombre_modelo = st.selectbox(
        "Ver predicciones del modelo",
        list(resultados["predicciones"].keys())
    )

    real = resultados["y_test"].to_numpy()
    pred = resultados["predicciones"][nombre_modelo]

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

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Detalles del filtrado IQR"):

        st.write(
            f"Entrenamiento: {resultados['filas_train_original']:,} registros originales "
            f"→ {resultados['filas_train_filtrado']:,} tras IQR."
        )

        st.write(
            f"Prueba: {resultados['filas_test_original']:,} registros originales "
            f"→ {resultados['filas_test_filtrado']:,} tras IQR."
        )


# ---------------------------------------------------------
# 4. PREDICCIÓN
# ---------------------------------------------------------

elif seccion == "Predecir precio":

    st.header("¿Cuánto vale este vehículo?")
    st.write("Introduce sus características para estimar su precio.")

    modelo_nombre = st.selectbox(
        "Modelo de predicción",
        list(resultados["modelos"].keys()),
        index=list(resultados["modelos"].keys()).index("CatBoost")
    )

    entrada = {}

    columnas = resultados["X_columns"]
    numericas = resultados["numeric_cols"]
    categoricas = resultados["categorical_cols"]

    with st.form("formulario_prediccion"):

        st.subheader("Características del vehículo")

        col_a, col_b = st.columns(2)

        for i, col in enumerate(columnas):

            contenedor = col_a if i % 2 == 0 else col_b

            with contenedor:

                if col in numericas:

                    serie = pd.to_numeric(df[col], errors="coerce").dropna()

                    if serie.empty:
                        entrada[col] = 0.0

                    else:
                        minimo = float(serie.min())
                        maximo = float(serie.max())
                        mediana = float(serie.median())

                        if minimo == maximo:
                            entrada[col] = minimo

                        else:
                            entrada[col] = st.number_input(
                                col.replace("_", " ").capitalize(),
                                min_value=minimo,
                                max_value=maximo,
                                value=min(max(mediana, minimo), maximo)
                            )

                else:

                    valores = (
                        df[col]
                        .dropna()
                        .astype(str)
                        .value_counts()
                        .head(100)
                        .index.tolist()
                    )

                    valores = sorted(valores)

                    opciones = ["Desconocido"] + [
                        v for v in valores if v != "Desconocido"
                    ]

                    entrada[col] = st.selectbox(
                        col.replace("_", " ").capitalize(),
                        opciones,
                        key=f"pred_{col}"
                    )

        enviado = st.form_submit_button(
            "Calcular precio estimado",
            type="primary"
        )

    if enviado:

        fila = pd.DataFrame([entrada], columns=columnas)

        if modelo_nombre == "CatBoost":

            for col in categoricas:
                fila[col] = fila[col].fillna("Desconocido").astype(str)

        modelo = resultados["modelos"][modelo_nombre]

        prediccion = float(modelo.predict(fila)[0])

        st.divider()
        st.subheader("Precio estimado")

        st.metric(
            "Valor estimado del vehículo",
            f"${max(0, prediccion):,.2f}"
        )

        st.caption(
            "Es una estimación estadística basada en los datos del dataset, "
            "no una tasación profesional ni una garantía del precio de mercado."
        )


st.sidebar.divider()
st.sidebar.caption("AutoPrice AI · Proyecto de Machine Learning")