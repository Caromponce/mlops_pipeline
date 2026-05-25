# Model Training & Evaluation
# Input:  ft_engineering.preparar_datos()
# Output: mejor modelo guardado en models/best_model.joblib

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model_monitoring import guardar_referencia
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)

from ft_engineering import preparar_datos

sns.set_theme(style='whitegrid')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# FUNCIÓN summarize_classification
# ============================================================

def summarize_classification(model_name: str, y_true, y_pred, y_prob=None) -> dict:
    """
    Resume las métricas principales de clasificación para un modelo.

    Parámetros:
        model_name: nombre del modelo para identificación
        y_true:     valores reales
        y_pred:     predicciones del modelo
        y_prob:     probabilidades de clase positiva (para ROC-AUC)

    Retorna:
        dict con métricas clave
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    summary = {
        'modelo':       model_name,
        'accuracy':     round(report['accuracy'], 4),
        'precision_0':  round(report['0']['precision'], 4),
        'recall_0':     round(report['0']['recall'], 4),
        'f1_0':         round(report['0']['f1-score'], 4),
        'precision_1':  round(report['1']['precision'], 4),
        'recall_1':     round(report['1']['recall'], 4),
        'f1_1':         round(report['1']['f1-score'], 4),
        'f1_macro':     round(report['macro avg']['f1-score'], 4),
        'roc_auc':      round(roc_auc_score(y_true, y_prob), 4) if y_prob is not None else None
    }

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, zero_division=0))
    if y_prob is not None:
        print(f"  ROC-AUC: {summary['roc_auc']}")

    return summary

# ============================================================
# FUNCIÓN optimizar_threshold
# ============================================================

def optimizar_threshold(modelo, X_test, y_test, modelo_nombre: str):
    """
    Busca el threshold óptimo que maximiza el F1 de la clase mora (0).
    Por defecto sklearn predice clase 1 si proba >= 0.5,
    pero con clases desbalanceadas conviene bajarlo.
    """
    if not hasattr(modelo, 'predict_proba'):
        print(f"{modelo_nombre} no soporta predict_proba, se omite.")
        return 0.5

    probas = modelo.predict_proba(X_test)[:, 1]  # proba de clase 1 (puntual)
    thresholds = np.arange(0.1, 0.9, 0.05)
    resultados = []

    for t in thresholds:
        y_pred_t = (probas >= t).astype(int)
        report = classification_report(y_test, y_pred_t, output_dict=True, zero_division=0)
        resultados.append({
            'threshold': round(t, 2),
            'recall_0':  round(report['0']['recall'], 3),
            'precision_0': round(report['0']['precision'], 3),
            'f1_0':      round(report['0']['f1-score'], 3),
            'f1_macro':  round(report['macro avg']['f1-score'], 3)
        })

    df_t = pd.DataFrame(resultados)

    # Graficamos cómo varían las métricas con el threshold
    plt.figure(figsize=(10, 4))
    plt.plot(df_t['threshold'], df_t['recall_0'],    label='Recall mora (0)',    marker='o')
    plt.plot(df_t['threshold'], df_t['precision_0'], label='Precision mora (0)', marker='s')
    plt.plot(df_t['threshold'], df_t['f1_0'],        label='F1 mora (0)',        marker='^')
    plt.plot(df_t['threshold'], df_t['f1_macro'],    label='F1 macro',           linestyle='--')
    plt.axvline(0.5, color='gray', linestyle=':', label='Threshold default (0.5)')
    plt.xlabel('Threshold')
    plt.title(f'Métricas vs Threshold — {modelo_nombre}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Threshold óptimo = mayor F1 para clase mora
    mejor = df_t.loc[df_t['f1_0'].idxmax()]
    print(f"\nThreshold óptimo para {modelo_nombre}: {mejor['threshold']}")
    print(f"  recall_0={mejor['recall_0']} | precision_0={mejor['precision_0']} | f1_0={mejor['f1_0']}")

    return mejor['threshold']

# ============================================================
# FUNCIÓN analizar_importancia_features
# ============================================================

def analizar_importancia_features(modelo, feature_names: list, modelo_nombre: str):
    """Grafica las features más importantes del modelo."""
    if not hasattr(modelo, 'feature_importances_'):
        print(f"{modelo_nombre} no tiene feature_importances_")
        return

    importancias = pd.Series(modelo.feature_importances_, index=feature_names)
    top20 = importancias.sort_values(ascending=False).head(20)

    plt.figure(figsize=(10, 6))
    top20.sort_values().plot(kind='barh', color='steelblue', alpha=0.8)
    plt.title(f'Top 20 features — {modelo_nombre}', fontweight='bold')
    plt.xlabel('Importancia')
    plt.tight_layout()
    plt.show()

    print(f"\nTop 5 features más importantes:")
    print(top20.head(5).to_string())

# ============================================================
# DEFINICIÓN DE MODELOS
# ============================================================

MODELOS = {
    'Logistic Regression': LogisticRegression(
        class_weight='balanced', max_iter=1000, random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, random_state=42
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=5, n_jobs=-1
    )
}


# ============================================================
# ENTRENAMIENTO Y EVALUACIÓN
# ============================================================

def entrenar_y_evaluar(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """
    Entrena todos los modelos, muestra métricas y retorna tabla resumen.
    """
    resultados = []

    for nombre, modelo in MODELOS.items():
        print(f"\nEntrenando {nombre}...")
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        y_prob = (modelo.predict_proba(X_test)[:, 1]
                  if hasattr(modelo, 'predict_proba') else None)

        resumen = summarize_classification(nombre, y_test, y_pred, y_prob)
        resumen['modelo_obj'] = modelo   # guardamos el objeto para luego
        resultados.append(resumen)

    return pd.DataFrame(resultados)


# ============================================================
# VISUALIZACIONES COMPARATIVAS
# ============================================================

def graficar_comparacion(df_resultados: pd.DataFrame):
    """
    Gráficos comparativos de métricas entre modelos.
    """
    metricas = ['accuracy', 'f1_macro', 'roc_auc', 'recall_0']
    colores  = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    fig, axes = plt.subplots(1, len(metricas), figsize=(16, 5))
    fig.suptitle('Comparación de modelos', fontsize=14, fontweight='bold')

    for i, (metrica, color) in enumerate(zip(metricas, colores)):
        valores = df_resultados.set_index('modelo')[metrica].sort_values()
        valores.plot(kind='barh', ax=axes[i], color=color, alpha=0.8)
        axes[i].set_title(metrica)
        axes[i].set_xlim(0, 1)
        for p in axes[i].patches:
            axes[i].annotate(f'{p.get_width():.3f}',
                             (p.get_width() + 0.01, p.get_y() + p.get_height() / 2),
                             va='center', fontsize=9)

    plt.tight_layout()
    plt.show()


def graficar_matrices_confusion(df_resultados, X_test, y_test):
    """
    Matrices de confusión para todos los modelos.
    """
    n = len(df_resultados)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    fig.suptitle('Matrices de confusión', fontsize=13, fontweight='bold')

    for i, row in df_resultados.iterrows():
        modelo = row['modelo_obj']
        y_pred = modelo.predict(X_test)
        disp = ConfusionMatrixDisplay(
            confusion_matrix(y_test, y_pred),
            display_labels=['Mora (0)', 'Puntual (1)']
        )
        disp.plot(ax=axes[i], colorbar=False, cmap='Blues')
        axes[i].set_title(row['modelo'])

    plt.tight_layout()
    plt.show()


def graficar_roc(df_resultados, X_test, y_test):
    """
    Curvas ROC superpuestas para todos los modelos.
    """
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')

    for _, row in df_resultados.iterrows():
        modelo = row['modelo_obj']
        if hasattr(modelo, 'predict_proba'):
            y_prob = modelo.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{row['modelo']} (AUC = {row['roc_auc']})")

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curvas ROC', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


# ============================================================
# SELECCIÓN Y GUARDADO DEL MEJOR MODELO
# ============================================================

def seleccionar_y_guardar(df_resultados: pd.DataFrame, X_test, y_test) -> str:
    cols_tabla = ['modelo', 'accuracy', 'f1_macro', 'roc_auc',
                  'recall_0', 'precision_0', 'f1_0']
    print("\n=== TABLA RESUMEN — THRESHOLD DEFAULT (0.5) ===")
    print(df_resultados[cols_tabla].sort_values('roc_auc', ascending=False)
                                   .to_string(index=False))

    # Selección por ROC-AUC (criterio estándar en crédito)
    mejor_idx    = df_resultados['roc_auc'].idxmax()
    mejor_fila   = df_resultados.loc[mejor_idx]
    mejor_modelo = mejor_fila['modelo_obj']
    mejor_nombre = mejor_fila['modelo']

    # Buscar threshold óptimo para el mejor modelo
    threshold_opt = optimizar_threshold(mejor_modelo, X_test, y_test, mejor_nombre)

    # Evaluar con threshold óptimo
    y_prob = mejor_modelo.predict_proba(X_test)[:, 1]
    y_pred_opt = (y_prob >= threshold_opt).astype(int)

    print(f"\n=== {mejor_nombre} CON THRESHOLD ÓPTIMO ({threshold_opt}) ===")
    resumen_opt = summarize_classification(
        f"{mejor_nombre} (threshold={threshold_opt})",
        y_test, y_pred_opt, y_prob
    )

    # Comparativa default vs óptimo
    print("\n=== COMPARATIVA: THRESHOLD 0.5 vs ÓPTIMO ===")
    comp = pd.DataFrame([
        {
            'threshold': 0.50,
            'recall_0':  mejor_fila['recall_0'],
            'precision_0': mejor_fila['precision_0'],
            'f1_0':      mejor_fila['f1_0'],
            'roc_auc':   mejor_fila['roc_auc']
        },
        {
            'threshold':   threshold_opt,
            'recall_0':    resumen_opt['recall_0'],
            'precision_0': resumen_opt['precision_0'],
            'f1_0':        resumen_opt['f1_0'],
            'roc_auc':     resumen_opt['roc_auc']
        }
    ])
    print(comp.to_string(index=False))

    # Guardar modelo + threshold juntos
    bundle = {'modelo': mejor_modelo, 'threshold': threshold_opt}
    ruta = os.path.join(MODELS_DIR, 'best_model.joblib')
    joblib.dump(bundle, ruta)

    print(f"\n✅ Modelo guardado: {mejor_nombre} | threshold={threshold_opt}")
    print(f"   Gini = {round(2 * mejor_fila['roc_auc'] - 1, 4)}")
    print(f"   Ruta: {ruta}")

    return ruta

# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

if __name__ == '__main__':
    print("Cargando y transformando datos...")
    X_train, X_test, y_train, y_test, _ = preparar_datos()
    guardar_referencia(X_train)

    print("\nEntrenando modelos...")
    df_resultados = entrenar_y_evaluar(X_train, X_test, y_train, y_test)

    print("\nGenerando visualizaciones...")
    graficar_comparacion(df_resultados)
    graficar_matrices_confusion(df_resultados, X_test, y_test)
    graficar_roc(df_resultados, X_test, y_test)


    # Optimizar threshold para el mejor modelo (Gradient Boosting)
    mejor_modelo_obj = df_resultados.loc[df_resultados['roc_auc'].idxmax(), 'modelo_obj']
    mejor_nombre     = df_resultados.loc[df_resultados['roc_auc'].idxmax(), 'modelo']

    threshold_optimo = optimizar_threshold(mejor_modelo_obj, X_test, y_test, mejor_nombre)

    # Re-evaluar con threshold óptimo
    probas_optimas = mejor_modelo_obj.predict_proba(X_test)[:, 1]
    y_pred_optimo  = (probas_optimas >= threshold_optimo).astype(int)
    print(f"\n=== GRADIENT BOOSTING CON THRESHOLD={threshold_optimo} ===")
    summarize_classification(
        f"{mejor_nombre} (t={threshold_optimo})",
        y_test, y_pred_optimo,
        probas_optimas
    )

    analizar_importancia_features(mejor_modelo_obj, X_train.columns.tolist(), mejor_nombre)
    
    seleccionar_y_guardar(df_resultados, X_test, y_test)