# Feature Engineering Pipeline
# Input:  Base_de_datos.xlsx (raw)
# Output: X_train, X_test, y_train, y_test transformados

import pandas as pd
import numpy as np
from cargar_datos import loadData

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer
from imblearn.over_sampling import SMOTE


# ============================================================================
# 1. DEFINICIÓN DE GRUPOS DE VARIABLES (resultado de comprension_eda.ipynb)
# ============================================================================

# Variables a eliminar antes de modelar
COLS_DROP = [
    'fecha_prestamo',      # reemplazada por año/mes
    'saldo_mora',          # LEAKAGE: mide directamente la mora (= target)
    'saldo_mora_codeudor', # LEAKAGE: ídem para codeudor
    'puntaje',             # LEAKAGE.
]

# Numéricas — imputación con mediana
NUMERIC_COLS = [
    'capital_prestado', 'plazo_meses', 'edad_cliente',
    'salario_cliente', 'total_otros_prestamos', 'cuota_pactada',
    'cant_creditosvigentes', 'huella_consulta',
    'creditos_sectorFinanciero', 'creditos_sectorCooperativo', 'creditos_sectorReal',
    'año_prestamo', 'mes_prestamo',
    'puntaje_datacredito'   
]

# Numéricas con alta asimetría — se aplica log1p antes de imputar
NUMERIC_LOG_COLS = [
    'saldo_total', 'saldo_principal',
    'promedio_ingresos_datacredito'  
]

# Categóricas nominales — OneHotEncoder
CATEGORICAL_NOMINAL_COLS = ['tipo_credito', 'tipo_laboral']

# Categóricas ordinales — OrdinalEncoder
CATEGORICAL_ORDINAL_COLS = ['tendencia_ingresos']
ORDINAL_CATEGORIES = [['Decreciente', 'Estable', 'Creciente', 'Sin_historial']]

TARGET = 'Pago_atiempo'


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica transformaciones previas al pipeline:
    - Corrige tipos
    - Crea features derivados
    - Unifica nulos en categóricas
    """
    df = df.copy()

    # Extraer features de fecha
    df['año_prestamo'] = df['fecha_prestamo'].dt.year
    df['mes_prestamo'] = df['fecha_prestamo'].dt.month

    # Feature derivado: carga financiera (cuota / salario)
    # Evitar división por cero con np.where
    df['ratio_carga_financiera'] = np.where(
        df['salario_cliente'] > 0,
        df['cuota_pactada'] / df['salario_cliente'],
        np.nan  # se imputará con mediana en el pipeline
    )

    # Unificar nulos en tendencia_ingresos: NaN → 'Sin_historial'
    df['tendencia_ingresos'] = df['tendencia_ingresos'].fillna('Sin_historial').astype(str)

    # Imputar saldos nulos con 0 (ausencia de dato = sin deuda/mora)
    saldo_cols = ['saldo_mora', 'saldo_total', 'saldo_principal', 'saldo_mora_codeudor']
    df[saldo_cols] = df[saldo_cols].fillna(0)

    # Eliminar columnas que no aportan al modelo
    df = df.drop(columns=COLS_DROP)

    return df


# ============================================================
# 3. CONSTRUCCIÓN DEL PIPELINE (ColumnTransformer)
# ============================================================

def construir_pipeline() -> ColumnTransformer:
    """
    Construye el ColumnTransformer con tres ramas:
    - Numéricas estándar:  SimpleImputer(mediana)
    - Numéricas con sesgo: log1p + SimpleImputer(mediana)
    - Nominales:           SimpleImputer(moda) + OneHotEncoder
    - Ordinales:           SimpleImputer(moda) + OrdinalEncoder
    """

    # Rama 1: numéricas estándar
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # Rama 2: numéricas asimétricas → log1p primero
    numeric_log_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('log',     FunctionTransformer(np.log1p, validate=True, feature_names_out='one-to-one'))
    ])

    # Rama 3: categóricas nominales
    categorical_nominal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Rama 4: categóricas ordinales
    categorical_ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(
            categories=ORDINAL_CATEGORIES,
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num',         numeric_pipeline,           NUMERIC_COLS + ['ratio_carga_financiera']),
        ('num_log',     numeric_log_pipeline,        NUMERIC_LOG_COLS),
        ('cat_nominal', categorical_nominal_pipeline, CATEGORICAL_NOMINAL_COLS),
        ('cat_ordinal', categorical_ordinal_pipeline, CATEGORICAL_ORDINAL_COLS),
    ], remainder='drop')

    return preprocessor


# ============================================================
# 4. FUNCIÓN PRINCIPAL
# ============================================================

def preparar_datos(test_size: float = 0.2, random_state: int = 42, aplicar_smote: bool = True):
    """
    Pipeline completo de feature engineering.

    Parámetros:
        test_size:      proporción del set de evaluación (default 0.2)
        random_state:   semilla de reproducibilidad
        aplicar_smote:  si True, aplica SMOTE al set de entrenamiento

    Retorna:
        X_train, X_test, y_train, y_test
        preprocessor (fitted) — para reutilizar en predicción
    """
    # Carga
    df = loadData()

    # Feature engineering previo al pipeline
    df = crear_features(df)

    # Separar features y target
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # Split estratificado (desbalance 95/5)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y          # garantiza misma proporción de clases en ambos sets
    )

    # Construir y fitear el preprocessor SOLO sobre train para evitar data leakage
    preprocessor = construir_pipeline()
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep  = preprocessor.transform(X_test)

    # Recuperar nombres de columnas para interpretabilidad
    feature_names = [name.split('__', 1)[-1] 
                    for name in preprocessor.get_feature_names_out()]
    X_train_prep = pd.DataFrame(X_train_prep, columns=feature_names)
    X_test_prep  = pd.DataFrame(X_test_prep,  columns=feature_names)

    # Aplicar SMOTE sobre train para balancear clases
    if aplicar_smote:
        smote = SMOTE(random_state=random_state)
        X_train_prep, y_train = smote.fit_resample(X_train_prep, y_train)
        print(f"Distribución post-SMOTE: {pd.Series(y_train).value_counts().to_dict()}")

    print(f"X_train: {X_train_prep.shape} | X_test: {X_test_prep.shape}")
    print(f"Distribución y_train: {pd.Series(y_train).value_counts(normalize=True).round(3).to_dict()}")

    return X_train_prep, X_test_prep, y_train, y_test, preprocessor


# ============================================================
# 5. EJECUCIÓN DIRECTA 
# ============================================================

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, prep = preparar_datos()
    print("\nPrimeras columnas del dataset transformado:")
    print(X_train.head(2).to_string())