# AutoPrice AI — Streamlit

## Archivos
- `app.py`: aplicación Streamlit.
- `autos_limpio.csv`: dataset utilizado en el notebook.
- `requirements.txt`: dependencias.

## Preparación en Windows (PowerShell)

```powershell
# Crear y activar un entorno virtual (si todavía no tienes uno)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
streamlit run app.py
```

## Dataset
Coloca `autos_limpio.csv` en la misma carpeta que `app.py`, o súbelo desde el panel lateral de la aplicación.

El CSV debe contener una columna `price`. El código intenta leer primero el separador `;`, igual que el notebook, y después `,`.

La primera carga entrena Ridge, Random Forest y CatBoost. Las siguientes interacciones reutilizan los modelos mediante la caché de Streamlit.
