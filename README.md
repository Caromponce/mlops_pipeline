# MLOps Pipeline — Predicción de Riesgo Crediticio

## Caso de negocio

Una entidad financiera necesita anticipar qué clientes podrían caer en mora antes de que ocurra, para gestionar el riesgo de su cartera de créditos. El objetivo es construir un modelo de clasificación binaria que prediga si un cliente pagará a tiempo (`Pago_atiempo = 1`) o caerá en mora (`Pago_atiempo = 0`), utilizando información disponible **al momento del otorgamiento del crédito**.

El desafío principal del caso es el fuerte desbalance de clases: aproximadamente el 95% de los créditos se pagan a tiempo y solo el 5% caen en mora. Esto implica que un modelo que prediga siempre "pago puntual" tendría 95% de accuracy sin detectar ningún caso de riesgo real.

La métrica principal adoptada es el **ROC-AUC** (equivalente al coeficiente de Gini = 2×AUC − 1), estándar en la industria financiera para modelos de scoring crediticio, ya que evalúa la capacidad de ordenar clientes por riesgo independientemente del umbral de decisión.

---

## Principales hallazgos

### EDA

- El dataset contiene 10.763 registros con 23 variables originales.
- Se identificaron **3 variables con data leakage**: `saldo_mora` y `saldo_mora_codeudor` (miden directamente la mora, que es el target) y `puntaje` (score interno generado post-otorgamiento).
- Las variables con mayor diferencia de distribución entre clases son `huella_consulta`, `creditos_sectorReal` y `tendencia_ingresos`.
- Variables como `saldo_total`, `saldo_principal` y `promedio_ingresos_datacredito` presentan alta asimetría positiva → se aplica transformación log1p.
- La variable `tendencia_ingresos` es ordinal con 4 categorías: Decreciente, Estable, Creciente, Sin_historial.

### Feature Engineering

- Se creó la feature derivada `ratio_carga_financiera` = cuota / salario, que captura el esfuerzo financiero del cliente.
- Se extrajeron `año_prestamo` y `mes_prestamo` de la fecha del crédito.
- El pipeline de preprocesamiento tiene 4 ramas con `ColumnTransformer`:
  - Numéricas estándar → imputación por mediana
  - Numéricas asimétricas → imputación por mediana + log1p
  - Categóricas nominales → imputación por moda + OneHotEncoder
  - Categóricas ordinales → imputación por moda + OrdinalEncoder
- Se aplicó **SMOTE** exclusivamente sobre el set de entrenamiento para balancear clases, evitando data leakage hacia el set de evaluación.
- Dimensiones finales: X_train (16.402 × 27), X_test (2.153 × 27).

### Modelado

Se compararon 4 modelos con las métricas obtenidas en el set de test:

| Modelo | ROC-AUC | F1 Macro | Recall Mora |
|---|---|---|---|
| Gradient Boosting | **0.677** | 0.513 | 0.029 |
| Random Forest | 0.593 | 0.514 | 0.029 |
| Logistic Regression | 0.532 | 0.453 | 0.392 |
| KNN | 0.514 | 0.463 | 0.235 |

**Modelo seleccionado: Gradient Boosting Classifier**
- ROC-AUC = 0.677 → Gini = 0.353
- Threshold óptimo ajustado a ~0.75 para maximizar la detección de mora (recall clase 0)
- Las variables de mayor importancia son: `huella_consulta`, `creditos_sectorReal`, `tendencia_ingresos`

El recall bajo en la clase minoritaria (mora) es consistente con la naturaleza del dataset: los predictores disponibles al momento del otorgamiento tienen poder discriminativo moderado, lo cual es esperable en riesgo crediticio sin historial de comportamiento previo del cliente.

**Métricas**  
En este proyecto el objetivo de negocio no es predecir bien en general, sino detectar los créditos que van a caer en mora antes de que ocurran. Por lo cual se priorizan las métricas ROC-AUC y Recall.

ROC-AUC mide qué tan bien el modelo ordena a los clientes por nivel de riesgo, independientemente del umbral que se elija para tomar la decisión final. Un AUC de 0.677 significa que el modelo distingue correctamente el orden de riesgo entre dos clientes al azar el 67,7% de las veces, lo cual tiene valor operativo real aunque el número parezca moderado.

Recall de la clase mora mide cuántos casos de mora reales logra capturar el modelo. Como solo el 5% del dataset es mora, un modelo que ignore completamente esa clase obtiene 95% de accuracy igual. El accuracy es entonces una métrica engañosa que no sirve. Lo que puede costar al banco es el falso negativo: aprobar un crédito que después no se paga. Por eso el recall sobre la clase minoritaria es la métrica operativa más directa, aunque mejorarla tiene un costo: aumenta los falsos positivos (clientes solventes clasificados como riesgo), lo que se gestiona ajustando el threshold. 


### Monitoreo

Se implementó un sistema de monitoreo de **data drift** con muestreo periódico mensual. Las métricas calculadas son:

- **KS test** (Kolmogorov-Smirnov) para variables numéricas
- **PSI** (Population Stability Index) para variables numéricas
- **Jensen-Shannon divergence** para variables numéricas
- **Chi-cuadrado** para variables categóricas

Resultados de la simulación de 6 períodos:

| Período | Variables con drift crítico |
|---|---|
| 2025-01 | 0% — línea base estable |
| 2025-02 | 14.8% — primeras señales |
| 2025-03 | 29.6% — pico, cambio estructural |
| 2025-04 | 22.2% — drift sostenido |
| 2025-05 | 22.2% — drift sostenido |
| 2025-06 | 22.2% — drift sostenido |

A partir del período 2025-03 se recomienda reentrenamiento del modelo dado que el drift supera el umbral crítico de PSI > 0.20 en múltiples variables.

### Despliegue

Se implementó una API REST con FastAPI que expone el modelo entrenado como servicio de predicción. La API soporta predicción por lotes vía JSON y carga de archivos CSV, e incluye health check y documentación interactiva automática (Swagger UI). El servicio fue containerizado con Docker para garantizar reproducibilidad en cualquier entorno.

---

## Estructura del proyecto

```
mlops_pipeline/
│
├── mlops_pipeline/
│   └── src/
│       ├── cargar_datos.py              # Carga y validación del dataset
│       ├── comprension_eda.ipynb        # Análisis exploratorio de datos
│       ├── ft_engineering.py            # Pipeline de feature engineering
│       ├── model_training_evaluation.py # Entrenamiento y evaluación de modelos
│       ├── model_monitoring.py          # Detección de data drift
│       ├── app_streamlit.py             # Dashboard de monitoreo
│       └── model_deploy.py              # API REST de predicción (FastAPI)
│
├── models/
│   └── best_model.joblib               # Modelo y threshold serializados
│
├── monitoring/
│   ├── reference_stats.pkl             # Estadísticas de referencia (train)
│   └── drift_log.csv                   # Log histórico de drift por período
│
├── Base_de_datos.xlsx                  # Dataset original
├── Dockerfile                          # Imagen Docker para despliegue
├── .dockerignore
├── requirements.txt
└── README.md
```

---

## Instalación

```bash
git clone https://github.com/Caromponce/mlops_pipeline.git
cd mlops_pipeline
pip install -r requirements.txt
```

### Dependencias principales

```
pandas
numpy
scikit-learn>=1.3.0
imbalanced-learn
joblib
scipy
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
pydantic>=2.0.0
streamlit
plotly
openpyxl
```

---

## Cómo ejecutar

### 1. Feature Engineering

```bash
python src/ft_engineering.py
```

Preprocesa los datos y devuelve los sets de entrenamiento y evaluación listos para modelar.

### 2. Entrenamiento y evaluación

```bash
python src/model_training_evaluation.py
```

Entrena los 4 modelos, los compara, selecciona el mejor, optimiza el threshold y guarda el modelo en `models/best_model.joblib`. También guarda las estadísticas de referencia para monitoreo.

### 3. Monitoreo de drift

```bash
python src/model_monitoring.py
```

Genera el log de drift en `monitoring/drift_log.csv` comparando la distribución de referencia contra batches simulados de producción.

### 4. Dashboard de monitoreo

```bash
streamlit run src/app_streamlit.py
```

Abre la aplicación web de monitoreo en `http://localhost:8501`.

---

### 5. API de predicción

La API expone el modelo de Gradient Boosting como servicio REST con FastAPI. Soporta predicción individual y por lotes vía JSON o archivo CSV.

#### Opción A — Uvicorn directo (desarrollo)

```bash
uvicorn mlops_pipeline.src.model_deploy:app --host 0.0.0.0 --port 8000 --reload
```

La API queda disponible en `http://localhost:8000`. La documentación interactiva (Swagger UI) se abre en `http://localhost:8000/docs`.

#### Opción B — Docker (producción)

```bash
# Construir la imagen
docker build -t mlops-credit-api .

# Ejecutar el contenedor
docker run -p 8000:8000 mlops-credit-api
```

Para sobreescribir el threshold desde fuera del contenedor:

```bash
docker run -p 8000:8000 -e PREDICT_THRESHOLD=0.50 mlops-credit-api
```

#### Endpoints disponibles

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Información general del servicio |
| GET | `/health` | Estado del servicio y modelo |
| GET | `/features` | Lista de features que espera el modelo |
| POST | `/predict` | Predicción por lotes vía JSON |
| POST | `/predict/csv` | Predicción por lotes vía archivo CSV |

#### Ejemplo formato de entrada (POST /predict)

```json
{
  "records": [
    {
      "_descripcion": "CASO 1 — Bajo riesgo",
      "capital_prestado": 8000000,
      "plazo_meses": 36,
      "edad_cliente": 42,
      "salario_cliente": 4500000,
      "total_otros_prestamos": 1,
      "cuota_pactada": 280000,
      "cant_creditosvigentes": 2,
      "huella_consulta": 1,
      "creditos_sectorFinanciero": 1,
      "creditos_sectorCooperativo": 0,
      "creditos_sectorReal": 1,
      "año_prestamo": 2024,
      "mes_prestamo": 3,
      "puntaje_datacredito": 720,
      "ratio_carga_financiera": 0.06,
      "saldo_total": 6500000,
      "saldo_principal": 5000000,
      "promedio_ingresos_datacredito": 4200000,
      "tipo_credito_4": 1,
      "tipo_credito_6": 0,
      "tipo_credito_7": 0,
      "tipo_credito_9": 0,
      "tipo_credito_10": 0,
      "tipo_credito_68": 0,
      "tipo_laboral_Empleado": 1,
      "tipo_laboral_Independiente": 0,
      "tendencia_ingresos": 3
    },
    {
      "_descripcion": "CASO 2 - Mora Extrema",
      "capital_prestado": 35000000,
      "plazo_meses": 72,
      "edad_cliente": 26,
      "salario_cliente": 900000,
      "total_otros_prestamos": 12,
      "cuota_pactada": 850000,
      "cant_creditosvigentes": 15,
      "huella_consulta": 18,
      "creditos_sectorFinanciero": 5,
      "creditos_sectorCooperativo": 4,
      "creditos_sectorReal": 8,
      "año_prestamo": 2024,
      "mes_prestamo": 11,
      "puntaje_datacredito": 300,
      "ratio_carga_financiera": 0.94,
      "saldo_total": 68000000,
      "saldo_principal": 55000000,
      "promedio_ingresos_datacredito": 800000,
      "tipo_credito_4": 0,
      "tipo_credito_6": 1,
      "tipo_credito_7": 1,
      "tipo_credito_9": 1,
      "tipo_credito_10": 0,
      "tipo_credito_68": 1,
      "tipo_laboral_Empleado": 0,
      "tipo_laboral_Independiente": 1,
      "tendencia_ingresos": 0
    },
    {
      "_descripcion": "CASO 3 — Riesgo intermedio",
      "capital_prestado": 5000000,
      "plazo_meses": 24,
      "edad_cliente": 35,
      "salario_cliente": 2800000,
      "total_otros_prestamos": 3,
      "cuota_pactada": 310000,
      "cant_creditosvigentes": 4,
      "huella_consulta": 4,
      "creditos_sectorFinanciero": 2,
      "creditos_sectorCooperativo": 0,
      "creditos_sectorReal": 2,
      "año_prestamo": 2024,
      "mes_prestamo": 6,
      "puntaje_datacredito": 580,
      "ratio_carga_financiera": 0.11,
      "saldo_total": 9000000,
      "saldo_principal": 7000000,
      "promedio_ingresos_datacredito": 2500000,
      "tipo_credito_4": 0,
      "tipo_credito_6": 0,
      "tipo_credito_7": 0,
      "tipo_credito_9": 1,
      "tipo_credito_10": 0,
      "tipo_credito_68": 0,
      "tipo_laboral_Empleado": 1,
      "tipo_laboral_Independiente": 0,
      "tendencia_ingresos": 2
    }
  ]
}
```

Los campos faltantes se imputan automáticamente con la mediana o moda según el tipo de variable.

#### Respuesta esperada
```json
{
  "total_records": 3,
  "threshold_used": 0.5,
  "predictions": [
    {
      "record_index": 0,
      "probability_mora": 0.369026,
      "prediction": 1,
      "risk_label": "PAGO_A_TIEMPO"
    },
    {
      "record_index": 1,
      "probability_mora": 0.520748,
      "prediction": 0,
      "risk_label": "MORA"
    },
    {
      "record_index": 2,
      "probability_mora": 0.33809,
      "prediction": 1,
      "risk_label": "PAGO_A_TIEMPO"
    }
  ],
  "summary": {
    "predicciones_mora": 1,
    "predicciones_pago_a_tiempo": 2,
    "tasa_mora_estimada": 0.3333
  }
}

```


#### Codificación de tendencia_ingresos

La variable `tendencia_ingresos` fue codificada ordinalmente durante el entrenamiento. La API acepta tanto el número como el string:

| String | Valor numérico |
|---|---|
| `"Decreciente"` | 0 |
| `"Sin_historial"` | 1 |
| `"Estable"` | 2 |
| `"Creciente"` | 3 |

#### Configuración del threshold

El modelo fue entrenado con un threshold óptimo de **0.75** (maximiza la detección de mora). Este valor se carga automáticamente desde `models/best_model.joblib`.

Para sobreescribir el threshold sin reconstruir la imagen:

```bash
# Con uvicorn
set PREDICT_THRESHOLD=0.50
uvicorn mlops_pipeline.src.model_deploy:app --host 0.0.0.0 --port 8000

# Con Docker
docker run -p 8000:8000 -e PREDICT_THRESHOLD=0.50 mlops-credit-api
```

> **Nota sobre el threshold:** El valor 0.75 está optimizado para producción (prioriza precisión sobre recall en mora). En demos o testing se puede bajar a 0.50 para observar clasificaciones de mora, dado que el modelo tiene ROC-AUC = 0.677 y las probabilidades rara vez superan 0.55 incluso en perfiles de alto riesgo.

---

## Versionado

| Tag | Rama | Descripción |
|---|---|---|
| V1.0.0 | main | Initial commit |
| V1.0.1 | main | Carga de datos y EDA |
| V1.0 | certification | Certificación V1.0 |
| V1.1.0 | main | Feature engineering, model training y modelo serializado |
| V1.1 | certification | Certificación V1.1 |
| V1.2.0 | main | Monitoreo de drift y dashboard Streamlit |
| V1.2 | certification | Certificación V1.2 |
| V1.3.0 | main | API REST de predicción (FastAPI + Docker) — merge a main con certificación final |
| V1.3 | certification | Certificación V1.3 |

---

## Tecnologías

- **Python 3.11+**
- **scikit-learn** — pipelines, modelos, métricas
- **imbalanced-learn** — SMOTE
- **scipy** — tests estadísticos (KS, Chi-cuadrado, Jensen-Shannon)
- **streamlit** — dashboard interactivo
- **plotly** — visualizaciones
- **joblib** — serialización de modelos
- **FastAPI + Uvicorn** — API REST de predicción
- **Pydantic v2** — validación de esquemas de entrada/salida
- **Docker** — contenedor para despliegue reproducible