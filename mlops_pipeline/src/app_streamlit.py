# Aplicación de monitoreo de data drift — Proyecto Integrador M5

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

# ── Configuración de página ────────────────────────────────
st.set_page_config(
    page_title="Monitor de Drift — Crédito",
    page_icon="📊",
    layout="wide"
)

# ── Rutas ──────────────────────────────────────────────────
RUTA_LOG       = 'monitoring/drift_log.csv'
RUTA_REF_STATS = 'monitoring/reference_stats.pkl'
RUTA_MODELO    = 'models/best_model.joblib'

# ── Carga de datos (cacheada) ──────────────────────────────
@st.cache_data
def cargar_log():
    return pd.read_csv(RUTA_LOG)

@st.cache_resource
def cargar_referencia():
    return joblib.load(RUTA_REF_STATS)

@st.cache_resource
def cargar_modelo():
    return joblib.load(RUTA_MODELO)

# Verificar que los archivos existen
for ruta in [RUTA_LOG, RUTA_REF_STATS, RUTA_MODELO]:
    if not os.path.exists(ruta):
        st.error(f"Archivo no encontrado: `{ruta}`. Ejecutá `model_monitoring.py` primero.")
        st.stop()

df_log    = cargar_log()
referencia = cargar_referencia()
bundle    = cargar_modelo()
threshold = bundle['threshold']


# ── Sidebar ────────────────────────────────────────────────
st.sidebar.title("⚙️ Configuración")

periodos      = sorted(df_log['periodo'].unique())
periodo_sel   = st.sidebar.selectbox("Período a analizar", periodos, index=len(periodos)-1)
tipo_variable = st.sidebar.multiselect("Tipo de variable", ['numérica', 'categórica'],
                                        default=['numérica', 'categórica'])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Threshold del modelo:** `{threshold:.2f}`")
st.sidebar.markdown(f"**Períodos disponibles:** `{len(periodos)}`")

# ── Header ─────────────────────────────────────────────────
st.title("📊 Monitor de Data Drift — Riesgo Crediticio")
st.markdown(f"Período seleccionado: **{periodo_sel}** | Threshold: **{threshold:.2f}**")
st.divider()

# ── Datos del período seleccionado ─────────────────────────
df_periodo = df_log[
    (df_log['periodo'] == periodo_sel) &
    (df_log['tipo'].isin(tipo_variable))
].copy()




tab1, tab2, tab3, tab4 = st.tabs([
    "🚦 Resumen", "📈 Distribuciones", "📉 Evolución Temporal", "💡 Recomendaciones"
])

with tab1:
    st.subheader("Estado general del período")

    # KPIs resumen
    num_vars   = len(df_periodo)
    criticas   = df_periodo[
        (df_periodo['alerta_psi']  == '🔴 CRÍTICO') |
        (df_periodo['alerta_ks']   == '🔴 DRIFT')   |
        (df_periodo['alerta_chi2'] == '🔴 DRIFT')
    ]
    leves = df_periodo[
        (df_periodo['alerta_psi']  == '🟡 LEVE') |
        (df_periodo['alerta_js']   == '🟡 LEVE')
    ]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Variables monitoreadas", num_vars)
    col2.metric("🔴 Críticas",  len(criticas),  delta=None)
    col3.metric("🟡 Leves",     len(leves),     delta=None)
    col4.metric("🟢 Estables",  num_vars - len(criticas) - len(leves))

    st.markdown("#### Tabla de alertas por variable")

    # Columna de alerta consolidada
    def alerta_global(row):
        alertas = [row.get('alerta_psi'), row.get('alerta_ks'),
                   row.get('alerta_js'),  row.get('alerta_chi2')]
        if any('🔴' in str(a) for a in alertas): return '🔴 CRÍTICO'
        if any('🟡' in str(a) for a in alertas): return '🟡 LEVE'
        return '🟢 ESTABLE'

    df_periodo['estado'] = df_periodo.apply(alerta_global, axis=1)

    tabla = df_periodo[['variable', 'tipo', 'psi', 'ks_pval', 'js', 'chi2_pval', 'estado']].copy()
    tabla.columns = ['Variable', 'Tipo', 'PSI', 'KS p-valor', 'JS Divergencia', 'Chi² p-valor', 'Estado']

    # Colorear por estado
    def colorear(val):
        if '🔴' in str(val): return 'background-color: #ffcccc'
        if '🟡' in str(val): return 'background-color: #fff3cc'
        if '🟢' in str(val): return 'background-color: #ccffcc'
        return ''

    st.dataframe(
        tabla.style.map(colorear, subset=['Estado']),
        width='stretch', hide_index=True
    )

    # Barra de riesgo global
    pct_critico = len(criticas) / num_vars * 100 if num_vars > 0 else 0
    st.markdown("#### Índice de riesgo global")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct_critico,
        title={'text': "% Variables con drift crítico"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar':  {'color': "darkred"},
            'steps': [
                {'range': [0,  20],  'color': '#ccffcc'},
                {'range': [20, 50],  'color': '#fff3cc'},
                {'range': [50, 100], 'color': '#ffcccc'},
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'value': 50}
        }
    ))
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, width='stretch')



    with tab2:
        st.subheader("Distribución histórica vs período actual")

        vars_numericas = df_periodo[df_periodo['tipo'] == 'numérica']['variable'].tolist()

        if not vars_numericas:
            st.info("No hay variables numéricas en la selección actual.")
        else:
            variable_sel = st.selectbox("Seleccioná una variable", vars_numericas)

            ref_vals = referencia['numericas'].get(variable_sel)

            if ref_vals is not None:
                # Simulamos valores del período actual con el mismo factor de drift del log
                rng = np.random.default_rng(periodos.index(periodo_sel) * 42)
                factor = periodos.index(periodo_sel) / max(len(periodos) - 1, 1) * 0.5
                ref_sample = rng.choice(ref_vals, size=500, replace=True)
                act_vals = ref_sample + rng.normal(0, np.std(ref_vals) * factor, size=500)

                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=ref_vals, name='Referencia (train)',
                    opacity=0.6, marker_color='steelblue', nbinsx=30
                ))
                fig_dist.add_trace(go.Histogram(
                    x=act_vals, name=f'Actual ({periodo_sel})',
                    opacity=0.6, marker_color='tomato', nbinsx=30
                ))
                fig_dist.update_layout(
                    barmode='overlay',
                    title=f'Distribución de {variable_sel}',
                    xaxis_title=variable_sel,
                    yaxis_title='Frecuencia',
                    legend=dict(x=0.7, y=0.95),
                    height=400
                )
                st.plotly_chart(fig_dist, width='stretch')

                # Métricas de la variable seleccionada
                fila = df_periodo[df_periodo['variable'] == variable_sel].iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("PSI",           f"{fila['psi']:.4f}",  fila['alerta_psi'])
                c2.metric("KS p-valor",    f"{fila['ks_pval']:.4f}", fila['alerta_ks'])
                c3.metric("JS Divergencia",f"{fila['js']:.4f}",   fila['alerta_js'])



with tab3:
    st.subheader("Evolución del drift a lo largo del tiempo")

    vars_num_todas = df_log[df_log['tipo'] == 'numérica']['variable'].unique().tolist()
    var_temporal   = st.selectbox("Variable a observar", vars_num_todas, key='temporal')

    df_var = df_log[(df_log['variable'] == var_temporal) & (df_log['tipo'] == 'numérica')].copy()
    df_var = df_var.sort_values('periodo')

    if df_var.empty:
        st.warning("Sin datos para esta variable.")
    else:
        # PSI a lo largo del tiempo
        fig_psi = px.line(
            df_var, x='periodo', y='psi',
            title=f'PSI — {var_temporal} por período',
            markers=True, color_discrete_sequence=['steelblue']
        )
        fig_psi.add_hline(y=0.10, line_dash="dash", line_color="orange",
                           annotation_text="Umbral leve (0.10)")
        fig_psi.add_hline(y=0.20, line_dash="dash", line_color="red",
                           annotation_text="Umbral crítico (0.20)")
        fig_psi.update_layout(height=350)
        st.plotly_chart(fig_psi, width='stretch')

        # KS p-value a lo largo del tiempo
        fig_ks = px.line(
            df_var, x='periodo', y='ks_pval',
            title=f'KS p-valor — {var_temporal} por período',
            markers=True, color_discrete_sequence=['tomato']
        )
        fig_ks.add_hline(y=0.05, line_dash="dash", line_color="red",
                          annotation_text="α = 0.05 (umbral drift)")
        fig_ks.update_layout(height=350)
        st.plotly_chart(fig_ks, width='stretch')

        # Heatmap de alertas por variable y período
        st.markdown("#### Mapa de calor de drift (PSI)")
        df_pivot = df_log[df_log['tipo'] == 'numérica'].pivot_table(
            index='variable', columns='periodo', values='psi', aggfunc='mean'
        )
        fig_heat = px.imshow(
            df_pivot,
            color_continuous_scale=['#ccffcc', '#fff3cc', '#ffcccc'],
            zmin=0, zmax=0.3,
            title='PSI por variable y período',
            aspect='auto'
        )
        fig_heat.update_layout(height=500)
        st.plotly_chart(fig_heat, width='stretch')



    with tab4:
        st.subheader("💡 Recomendaciones automáticas")

        n_criticas = len(criticas)

        if n_criticas == 0:
            st.success("✅ El modelo opera dentro de parámetros normales. No se requieren acciones inmediatas.")
        elif n_criticas <= 3:
            st.warning(f"⚠️ Se detectaron **{n_criticas} variable(s)** con drift significativo.")
            st.markdown("""
            **Acciones sugeridas:**
            - Revisá las variables marcadas en rojo en la pestaña *Resumen*
            - Verificá si hubo cambios en la población o en el proceso de originación
            - Considerá reentrenar el modelo si el drift persiste más de 2 períodos
            """)
        else:
            st.error(f"🔴 ALERTA CRÍTICA: **{n_criticas} variables** presentan drift severo.")
            st.markdown("""
            **Se recomienda retraining inmediato:**
            - La distribución de los datos de entrada ha cambiado significativamente
            - El modelo puede estar generando predicciones poco confiables
            - Pasos sugeridos:
            1. Recolectar datos recientes etiquetados
            2. Reejecutar `ft_engineering.py` y `model_training_evaluation.py`
            3. Validar el nuevo modelo con el equipo de negocio antes de deploying
            """)

        # Detalle de variables críticas
        if not criticas.empty:
            st.markdown("#### Variables que requieren atención")
            st.dataframe(
                criticas[['variable', 'tipo', 'psi', 'alerta_psi', 'ks_pval', 'alerta_ks',
                        'chi2_pval', 'alerta_chi2']].rename(columns={
                    'variable': 'Variable', 'tipo': 'Tipo',
                    'psi': 'PSI', 'alerta_psi': 'Alerta PSI',
                    'ks_pval': 'KS p-valor', 'alerta_ks': 'Alerta KS',
                    'chi2_pval': 'Chi² p-valor', 'alerta_chi2': 'Alerta Chi²'
                }),
                width='stretch', hide_index=True
            )

        # Evolución histórica de alertas críticas
        st.markdown("#### Evolución de variables críticas por período")
        df_criticas_hist = df_log[
            (df_log['alerta_psi']  == '🔴 CRÍTICO') |
            (df_log['alerta_ks']   == '🔴 DRIFT')   |
            (df_log['alerta_chi2'] == '🔴 DRIFT')
        ].groupby('periodo').size().reset_index(name='n_criticas')

        if not df_criticas_hist.empty:
            fig_alertas = px.bar(
                df_criticas_hist, x='periodo', y='n_criticas',
                title='Variables críticas por período',
                color='n_criticas',
                color_continuous_scale=['#ccffcc', '#fff3cc', '#ffcccc']
            )
            fig_alertas.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_alertas, width='stretch')