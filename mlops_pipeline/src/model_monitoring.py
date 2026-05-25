# Monitoreo de data drift sobre el modelo de crédito
# Referencia: X_train (distribución de entrenamiento)
# Actual:     batches simulados con perturbación progresiva

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from scipy import stats
from scipy.spatial.distance import jensenshannon
from ft_engineering import preparar_datos

# ============================================================
# CONSTANTES
# ============================================================

RUTA_MODELO    = 'models/best_model.joblib'
RUTA_REF_STATS = 'monitoring/reference_stats.pkl'
RUTA_LOG       = 'monitoring/drift_log.csv'
N_BINS         = 10        # bins para PSI y JS
N_BATCHES      = 6         # períodos simulados
BATCH_SIZE     = 500       # registros por batch

# Thresholds de alerta
PSI_WARN     = 0.10   # leve
PSI_CRITICAL = 0.20   # crítico (retraining recomendado)
KS_ALPHA     = 0.05   # p-value para rechazar H0 (misma distribución)
JS_WARN      = 0.10
JS_CRITICAL  = 0.20
CHI2_ALPHA   = 0.05


# ============================================================
# MÉTRICAS DE DRIFT
# ============================================================

def calcular_psi(referencia: np.ndarray, actual: np.ndarray, n_bins: int = N_BINS) -> float:
    """
    Population Stability Index.
    PSI < 0.10 → estable | 0.10-0.20 → leve | > 0.20 → crítico
    """
    bins = np.percentile(referencia, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)  # eliminar duplicados en distribuciones constantes
    if len(bins) < 2:
        return 0.0

    ref_counts, _ = np.histogram(referencia, bins=bins)
    act_counts, _ = np.histogram(actual,     bins=bins)

    ref_pct = (ref_counts + 1e-6) / len(referencia)  # suavizado para evitar log(0)
    act_pct = (act_counts + 1e-6) / len(actual)

    psi = np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct))
    return round(float(psi), 4)


def calcular_ks(referencia: np.ndarray, actual: np.ndarray) -> tuple:
    """KS test para variables numéricas. Retorna (statistic, p_value)."""
    stat, pval = stats.ks_2samp(referencia, actual)
    return round(float(stat), 4), round(float(pval), 4)


def calcular_js(referencia: np.ndarray, actual: np.ndarray, n_bins: int = N_BINS) -> float:
    """Jensen-Shannon divergence (0 = idénticas, 1 = completamente distintas)."""
    bins = np.percentile(referencia, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0

    ref_hist, _ = np.histogram(referencia, bins=bins, density=True)
    act_hist, _ = np.histogram(actual,     bins=bins, density=True)

    ref_hist = ref_hist + 1e-9
    act_hist = act_hist + 1e-9
    js = jensenshannon(ref_hist, act_hist)
    return round(float(js), 4)


def calcular_chi2(referencia: pd.Series, actual: pd.Series) -> tuple:
    """Chi-cuadrado para variables categóricas. Retorna (statistic, p_value)."""
    categorias = pd.Index(referencia.unique()).union(actual.unique())
    ref_counts = referencia.value_counts().reindex(categorias, fill_value=0)
    act_counts = actual.value_counts().reindex(categorias, fill_value=0)

    tabla = np.array([ref_counts.values, act_counts.values])
    # Filtrar categorías con 0 en ambas filas para evitar divisiones por cero
    mask = tabla.sum(axis=0) > 0
    tabla = tabla[:, mask]

    if tabla.shape[1] < 2:
        return 0.0, 1.0

    stat, pval, _, _ = stats.chi2_contingency(tabla)
    return round(float(stat), 4), round(float(pval), 4)


def nivel_alerta_psi(psi: float) -> str:
    if psi >= PSI_CRITICAL: return '🔴 CRÍTICO'
    if psi >= PSI_WARN:     return '🟡 LEVE'
    return '🟢 ESTABLE'

def nivel_alerta_ks(pval: float) -> str:
    return '🔴 DRIFT' if pval < KS_ALPHA else '🟢 ESTABLE'

def nivel_alerta_js(js: float) -> str:
    if js >= JS_CRITICAL: return '🔴 CRÍTICO'
    if js >= JS_WARN:     return '🟡 LEVE'
    return '🟢 ESTABLE'

def nivel_alerta_chi2(pval: float) -> str:
    return '🔴 DRIFT' if pval < CHI2_ALPHA else '🟢 ESTABLE'



# ============================================================
# REFERENCIA (se ejecuta UNA vez al entrenar)
# ============================================================

def guardar_referencia(X_train: pd.DataFrame):
    """
    Guarda estadísticas de la distribución de entrenamiento.
    Se llama desde model_training_evaluation.py o manualmente.
    """
    os.makedirs('monitoring', exist_ok=True)
    stats_ref = {
        'numericas': {col: X_train[col].values for col in X_train.select_dtypes(include='number').columns},
        'categoricas': {col: X_train[col] for col in X_train.select_dtypes(exclude='number').columns},
        'columnas': list(X_train.columns)
    }
    joblib.dump(stats_ref, RUTA_REF_STATS)
    print(f"Referencia guardada en {RUTA_REF_STATS} ({len(X_train)} registros)")


# ============================================================
# ANÁLISIS DE DRIFT
# ============================================================

def analizar_drift(X_actual: pd.DataFrame, referencia: dict, periodo: str) -> pd.DataFrame:
    """
    Compara X_actual contra la distribución de referencia.
    Retorna DataFrame con métricas por variable.
    """
    resultados = []

    for col in referencia['columnas']:
        if col not in X_actual.columns:
            continue

        ref_vals = referencia['numericas'].get(col)

        if ref_vals is not None:  # variable numérica
            act_vals = X_actual[col].dropna().values
            if len(act_vals) == 0:
                continue

            psi              = calcular_psi(ref_vals, act_vals)
            ks_stat, ks_pval = calcular_ks(ref_vals, act_vals)
            js               = calcular_js(ref_vals, act_vals)

            resultados.append({
                'periodo':   periodo,
                'variable':  col,
                'tipo':      'numérica',
                'psi':       psi,
                'alerta_psi': nivel_alerta_psi(psi),
                'ks_stat':   ks_stat,
                'ks_pval':   ks_pval,
                'alerta_ks': nivel_alerta_ks(ks_pval),
                'js':        js,
                'alerta_js': nivel_alerta_js(js),
                'chi2_stat': None,
                'chi2_pval': None,
                'alerta_chi2': None,
            })

        elif col in referencia['categoricas']:  # variable categórica
            ref_serie = referencia['categoricas'][col]
            act_serie = X_actual[col].dropna()
            chi2_stat, chi2_pval = calcular_chi2(ref_serie, act_serie)

            resultados.append({
                'periodo':    periodo,
                'variable':   col,
                'tipo':       'categórica',
                'psi':        None,
                'alerta_psi': None,
                'ks_stat':    None,
                'ks_pval':    None,
                'alerta_ks':  None,
                'js':         None,
                'alerta_js':  None,
                'chi2_stat':  chi2_stat,
                'chi2_pval':  chi2_pval,
                'alerta_chi2': nivel_alerta_chi2(chi2_pval),
            })

    return pd.DataFrame(resultados)

# ============================================================
# SIMULACIÓN DE DATOS DE PRODUCCIÓN
# ============================================================

def simular_batch(X_ref, factor_drift, seed):
    rng = np.random.default_rng(seed)
    batch = X_ref.sample(n=BATCH_SIZE, replace=True, random_state=seed).copy()

    num_cols = batch.select_dtypes(include='number').columns.tolist()
    
    # Solo perturbar un subconjunto de columnas
    n_cols_drift = max(1, len(num_cols) // 3)
    cols_con_drift = rng.choice(num_cols, size=n_cols_drift, replace=False)

    for col in cols_con_drift:
        std = batch[col].std()
        ruido = rng.normal(0, std * factor_drift, size=len(batch))
        batch[col] = batch[col] + ruido

    return batch



# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def ejecutar_monitoreo():
    """
    Orquesta el monitoreo completo:
    1. Carga modelo y referencia
    2. Genera batches simulados con drift progresivo
    3. Calcula métricas por período
    4. Guarda log y genera alertas
    """
    os.makedirs('monitoring', exist_ok=True)

    # Cargar datos y guardar referencia si no existe
    print("Cargando datos...")
    X_train, X_test, y_train, y_test, preprocessor = preparar_datos()

    if not os.path.exists(RUTA_REF_STATS):
        guardar_referencia(X_train)

    referencia = joblib.load(RUTA_REF_STATS)

    # Cargar modelo
    bundle = joblib.load(RUTA_MODELO)
    modelo    = bundle['modelo']
    threshold = bundle['threshold']

    # Simular períodos con drift progresivo
    factores_drift = np.linspace(0, 0.20, N_BATCHES)  # máximo 20% de perturbación
    fecha_base = datetime(2025, 1, 1)

    todos_resultados = []

    print(f"\nAnalizando {N_BATCHES} períodos de monitoreo...")
    for i, factor in enumerate(factores_drift):
        fecha = (pd.Timestamp('2025-01-01') + pd.DateOffset(months=i)).strftime('%Y-%m')
        batch = simular_batch(X_train, factor_drift=factor, seed=i * 42)

        # Predicciones del modelo sobre el batch
        proba = modelo.predict_proba(batch)[:, 1]
        batch['proba_pago'] = proba
        batch['prediccion'] = (proba >= threshold).astype(int)
        batch['periodo']    = fecha

        # Calcular drift
        df_drift = analizar_drift(batch.drop(columns=['proba_pago', 'prediccion', 'periodo']),
                                   referencia, periodo=fecha)
        todos_resultados.append(df_drift)

        # Alertas críticas
        criticas = df_drift[
            (df_drift['alerta_psi'] == '🔴 CRÍTICO') |
            (df_drift['alerta_ks']  == '🔴 DRIFT')   |
            (df_drift['alerta_chi2']== '🔴 DRIFT')
        ]
        if not criticas.empty:
            print(f"\n⚠️  [{fecha}] ALERTAS CRÍTICAS en: {', '.join(criticas['variable'].tolist())}")
        else:
            print(f"  [{fecha}] Sin alertas críticas (factor drift={factor:.2f})")

    # Consolidar y guardar log
    log_df = pd.concat(todos_resultados, ignore_index=True)
    log_df.to_csv(RUTA_LOG, index=False)
    print(f"\nLog guardado en {RUTA_LOG} ({len(log_df)} registros)")

    # Resumen final
    print("\n=== RESUMEN ÚLTIMA MUESTRA ===")
    ultimo = log_df[log_df['periodo'] == log_df['periodo'].max()]
    print(ultimo[['variable', 'tipo', 'psi', 'alerta_psi', 'ks_pval', 'alerta_ks',
                  'chi2_pval', 'alerta_chi2']].to_string(index=False))

    return log_df


if __name__ == '__main__':
    ejecutar_monitoreo()