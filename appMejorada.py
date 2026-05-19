import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score
)
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

st.set_page_config(page_title="Sistema de Pronóstico y Regresión", page_icon="📈", layout="wide")
st.title("📊 Sistema Integrado: Pronóstico Simple + Regresión Lineal")
st.caption("Desarrollado por: Elvis Jesus Apaza Yucra")

# ============================
# CARGA DE DATOS (igual que antes)
# ============================
st.sidebar.header("📁 Cargar archivo CSV o Excel")
archivo = st.sidebar.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xlsx", "xls"])

if archivo is not None:
    nombre = archivo.name.lower()
    if nombre.endswith(".csv"):
        try:
            df = pd.read_csv(archivo, sep=None, engine="python")
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()
    else:
        df = pd.read_excel(archivo)
else:
    try:
        df = pd.read_csv("ds_salaries.csv")
    except FileNotFoundError:
        st.warning("Sube un archivo o asegura que 'ds_salaries.csv' esté en el directorio.")
        st.stop()

df = df.dropna(axis=1, how="all")
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Variables numéricas y objetivo
columnas_numericas = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
if len(columnas_numericas) == 0:
    st.error("No hay columnas numéricas.")
    st.stop()

variable_objetivo = st.sidebar.selectbox("Selecciona la variable objetivo", columnas_numericas)

# ============================
# CREACIÓN DE DOS VISTAS CON TABS
# ============================
tab1, tab2 = st.tabs(["📆 Métodos de Pronóstico Simple", "📊 Regresión Lineal (Simple y Múltiple)"])

# ------------------------------------------------------------------------------
# VISTA 1: MÉTODOS DE PRONÓSTICO SIMPLE (SERIE TEMPORAL)
# ------------------------------------------------------------------------------
with tab1:
    st.header("📈 Pronóstico con métodos simples (serie temporal)")
    st.markdown("""
    **Métodos:** Ingenuo, Deriva, Media móvil, Media, Ingenuo Estacional.  
    **Interactividad:** Pasa el mouse sobre cualquier línea o punto para ver su valor exacto.
    """)

    # ==================================================
    # INTERPRETACIÓN GENERAL
    # ==================================================
    with st.expander("📖 ¿Qué son los métodos de pronóstico simple?", expanded=False):
        st.markdown(r"""
        Los **métodos de pronóstico simple** son técnicas básicas que predicen valores futuros basándose únicamente en el comportamiento pasado de la variable. 
        No requieren variables explicativas externas, solo la serie histórica.
        
        **¿Para qué sirven?**
        - Establecer una **línea base** de comparación para modelos más complejos.
        - Identificar patrones básicos como tendencia o estacionalidad.
        - Funcionan bien en series con comportamiento estable o fuertemente repetitivo.
        
        **Fórmula general:**
        $$
        \hat{y}_{t+h} = f(y_t, y_{t-1}, \dots, y_{t-n})
        $$
        Donde $h$ es el horizonte de pronóstico.
        """)

    # Detección de columna temporal (opcional)
    columnas_fecha = [col for col in df.columns if col.lower() in ['work_year','year','año','fecha','date','tiempo']]
    usar_columna_fecha = st.checkbox("Usar columna temporal para ordenar", value=len(columnas_fecha)>0)
    if usar_columna_fecha and columnas_fecha:
        col_tiempo = st.selectbox("Selecciona columna de tiempo", columnas_fecha)
        df_temp = df.sort_values(by=col_tiempo).reset_index(drop=True)
        st.success(f"Ordenado por {col_tiempo}")
    else:
        df_temp = df.reset_index(drop=True)
        st.info("No se usó columna temporal. Se respeta el orden de filas.")

    # Variable objetivo
    if variable_objetivo not in df_temp.columns:
        st.error(f"La variable '{variable_objetivo}' no existe.")
        st.stop()
    y_ts = df_temp[variable_objetivo].dropna().values
    if len(y_ts) < 3:
        st.error("Serie muy corta (<3 observaciones).")
        st.stop()

    # ==================================================
    # HORIZONTE FIJO (número de pasos a pronosticar)
    # ==================================================
    max_horizon = len(y_ts) - 1  # al menos 1 dato para entrenamiento
    horizon = st.number_input("Número de pasos a pronosticar (horizonte)", 
                              min_value=1, max_value=max_horizon, 
                              value=min(5, max_horizon), step=1)
    
    # Dividir: entrenamiento = todos excepto los últimos 'horizon' valores; prueba = esos últimos
    split = len(y_ts) - horizon
    if split < 1:
        st.error("El horizonte es demasiado grande. Reduce el número de pasos.")
        st.stop()
    train, test = y_ts[:split], y_ts[split:]

    col1, col2 = st.columns(2)
    col1.metric("📊 Entrenamiento", len(train))
    col2.metric("🔮 Prueba (valores reales para comparar)", len(test))

    # ==================================================
    # FÓRMULAS Y FUNCIONES DE PRONÓSTICO
    # ==================================================
    with st.expander("📐 Fórmulas de los métodos de pronóstico", expanded=False):
        st.markdown(r"""
        ### 1. Método Ingenuo
        $$
        \hat{y}_{t+1} = y_t
        $$
        **Explicación:** El pronóstico para cualquier periodo futuro es igual al último valor observado.

        ### 2. Método de la Deriva
        $$
        \hat{y}_{t+h} = y_t + h \cdot \frac{y_t - y_1}{t-1}
        $$

        ### 3. Media Móvil Simple (ventana \(k\))
        $$
        \hat{y}_{t+1} = \frac{y_t + y_{t-1} + \dots + y_{t-k+1}}{k}
        $$

        ### 4. Método de la Media
        $$
        \hat{y}_{t+h} = \frac{1}{n}\sum_{i=1}^{n} y_i
        $$

        ### 5. Ingenuo Estacional (período \(f\))
        $$
        \hat{y}_{t+h} = y_{t+h-f}
        $$
        """)

    # Funciones de pronóstico
    def naive(train, steps):
        return np.full(steps, train[-1])

    def drift(train, steps):
        if len(train) <= 1:
            return np.full(steps, train[-1])
        slope = (train[-1] - train[0]) / (len(train) - 1)
        last = train[-1]
        return np.array([last + slope * (i+1) for i in range(steps)])

    def moving_average(train, steps, window):
        window = min(window, len(train))
        if len(train) == 0:
            return np.full(steps, 0)
        ma = np.convolve(train, np.ones(window)/window, mode='valid')
        last_ma = ma[-1] if len(ma) > 0 else train[-1]
        return np.full(steps, last_ma)

    def mean_method(train, steps):
        return np.full(steps, np.mean(train))

    def naive_seasonal(train, steps, season):
        if season > len(train):
            season = len(train)
        forecasts = []
        for i in range(steps):
            idx = len(train) - season + (i % season)
            if idx < 0:
                idx = 0
            forecasts.append(train[idx])
        return np.array(forecasts)

    # Parámetros
    window_ma = st.slider("Ventana media móvil", 1, min(10, len(train)), 3)
    season = st.number_input("Período estacional", min_value=1, max_value=len(train), 
                             value=min(7, len(train)), step=1)

    steps = len(test)  # horizonte
    pred_naive = naive(train, steps)
    pred_drift = drift(train, steps)
    pred_ma = moving_average(train, steps, window_ma)
    pred_mean = mean_method(train, steps)
    pred_seasonal = naive_seasonal(train, steps, season)

    # ==================================================
    # TABLA DE VALORES PRONOSTICADOS
    # ==================================================
    st.subheader("📋 Tabla de valores pronosticados a futuro")
    df_pronostico = pd.DataFrame({
        "Periodo": list(range(1, steps+1)),
        "Ingenuo": pred_naive,
        "Deriva": pred_drift,
        f"Media móvil (w={window_ma})": pred_ma,
        "Media": pred_mean,
        f"Ing. Estacional (f={season})": pred_seasonal
    }).round(2)
    st.dataframe(df_pronostico, use_container_width=True)

    with st.expander("📖 Interpretación de la tabla de pronósticos", expanded=False):
        st.markdown(f"""
        Esta tabla muestra los **valores estimados** para cada uno de los {steps} periodos futuros.
        - **Periodo 1** = siguiente valor después del entrenamiento.
        - Puedes comparar estos valores con los reales de prueba (si existen) en las métricas de error.
        """)

    # ==================================================
    # MÉTRICAS DE ERROR (solo si hay valores de prueba)
    # ==================================================
    def mae(y_true, y_pred): return np.mean(np.abs(y_true - y_pred))
    def rmse(y_true, y_pred): return np.sqrt(np.mean((y_true - y_pred)**2))
    def mape(y_true, y_pred):
        mask = y_true != 0
        if np.sum(mask) == 0: return 100.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    resultados = pd.DataFrame({
        "Método": ["Ingenuo", "Deriva", f"Media móvil (w={window_ma})", "Media", f"Ing. Estacional (f={season})"],
        "MAE": [mae(test, pred_naive), mae(test, pred_drift), mae(test, pred_ma), mae(test, pred_mean), mae(test, pred_seasonal)],
        "RMSE": [rmse(test, pred_naive), rmse(test, pred_drift), rmse(test, pred_ma), rmse(test, pred_mean), rmse(test, pred_seasonal)],
        "MAPE (%)": [mape(test, pred_naive), mape(test, pred_drift), mape(test, pred_ma), mape(test, pred_mean), mape(test, pred_seasonal)]
    }).round(2)

    st.subheader("📊 Comparación de errores (contra valores reales de prueba)")
    st.dataframe(resultados, use_container_width=True)

    with st.expander("📖 Interpretación de las métricas de error", expanded=False):
        st.markdown(r"""
        **MAE** (Error Absoluto Medio): Promedio de errores absolutos.  
        **RMSE** (Raíz del Error Cuadrático Medio): Penaliza errores grandes.  
        **MAPE** (Error Porcentual Absoluto Medio): Error relativo en %.

        Menor valor significa mejor pronóstico.
        """)

    # ==================================================
    # GRÁFICOS INTERACTIVOS CON PLOTLY
    # ==================================================
    import plotly.graph_objects as go
    import scipy.interpolate as interp

    tiempo_train = list(range(len(train)))
    tiempo_test = list(range(len(train), len(train)+steps))

    # ---- GRÁFICO PRINCIPAL (optimizado para muchos datos) ----
    st.subheader("📈 Pronóstico global (interactivo)")
    st.markdown("**Pasa el mouse sobre cualquier línea o punto para ver su valor exacto.**")

    # Contenedor con scroll vertical (altura fija de 500px)
    with st.container(height=500):
        fig_main = go.Figure()
        # Entrenamiento: línea más fina, puntos más pequeños y semitransparentes
        fig_main.add_trace(go.Scatter(
            x=tiempo_train, y=train,
            mode='lines+markers', name='📊 Entrenamiento (real)',
            line=dict(color='blue', width=1.5),
            marker=dict(size=3, opacity=0.6),
            hovertemplate='Periodo: %{x}<br>Valor real: %{y:.2f}<extra></extra>'
        ))

        # Métodos
        metodos_plot = [
            (pred_naive, "Ingenuo", 'green', 'dash'),
            (pred_drift, "Deriva", 'orange', 'dashdot'),
            (pred_ma, f"Media móvil (w={window_ma})", 'red', 'dot'),
            (pred_mean, "Media", 'purple', 'solid'),
            (pred_seasonal, f"Ing. Estacional (f={season})", 'brown', 'longdash')
        ]
        for pred, nombre, color, dash in metodos_plot:
            fig_main.add_trace(go.Scatter(
                x=tiempo_test, y=pred,
                mode='lines+markers', name=nombre,
                line=dict(color=color, width=1.5, dash=dash),
                marker=dict(size=4, symbol='x', opacity=0.7),
                hovertemplate=f'{nombre}<br>Periodo: %{{x}}<br>Valor: %{{y:.2f}}<extra></extra>'
            ))

        fig_main.update_layout(
            title=f"Pronósticos vs Entrenamiento - {variable_objetivo}",
            xaxis_title="Índice temporal",
            yaxis_title=variable_objetivo,
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        # Cambiamos use_container_width por width='stretch' (nueva sintaxis)
        st.plotly_chart(fig_main, width='stretch')

    # ---- GRÁFICAS INDIVIDUALES (suavizadas) ----
    st.subheader("📉 Gráficas individuales (suavizadas e interactivas)")
    cols = st.columns(2)
    metodos_individuales = [
        (pred_naive, "Ingenuo", 'green'),
        (pred_drift, "Deriva", 'orange'),
        (pred_ma, f"Media móvil (w={window_ma})", 'red'),
        (pred_mean, "Media", 'purple'),
        (pred_seasonal, f"Ing. Estacional (f={season})", 'brown')
    ]
    for i, (pred, nombre, color) in enumerate(metodos_individuales):
        with cols[i % 2]:
            with st.container(height=350):
                fig = go.Figure()
                # Entrenamiento
                fig.add_trace(go.Scatter(
                    x=tiempo_train, y=train,
                    mode='lines+markers', name='Entrenamiento (real)',
                    line=dict(color='blue', width=1.5),
                    marker=dict(size=2, opacity=0.5),
                    hovertemplate='Periodo: %{x}<br>Valor real: %{y:.2f}<extra></extra>'
                ))
                # Suavizado
                if len(tiempo_test) >= 4:
                    x_new = np.linspace(min(tiempo_test), max(tiempo_test), 100)
                    spl = interp.make_interp_spline(tiempo_test, pred, k=min(3, len(tiempo_test)-1))
                    y_smooth = spl(x_new)
                    fig.add_trace(go.Scatter(
                        x=x_new, y=y_smooth,
                        mode='lines', name=f"{nombre} (suavizado)",
                        line=dict(color=color, width=2, dash='dash'),
                        hovertemplate=f'{nombre}<br>Periodo: %{{x:.1f}}<br>Valor: %{{y:.2f}}<extra></extra>'
                    ))
                    fig.add_trace(go.Scatter(
                        x=tiempo_test, y=pred,
                        mode='markers', name=nombre,
                        marker=dict(color=color, size=8, symbol='circle'),
                        hovertemplate=f'{nombre}<br>Periodo: %{{x}}<br>Valor exacto: %{{y:.2f}}<extra></extra>'
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=tiempo_test, y=pred,
                        mode='lines+markers', name=nombre,
                        line=dict(color=color, width=2),
                        hovertemplate=f'{nombre}<br>Periodo: %{{x}}<br>Valor: %{{y:.2f}}<extra></extra>'
                    ))
                fig.update_layout(title=nombre, xaxis_title="Tiempo", yaxis_title=variable_objetivo, 
                                hovermode="closest", height=350)
                st.plotly_chart(fig, width='stretch')

    # ==================================================
    # CONCLUSIÓN DEL MEJOR MÉTODO
    # ==================================================
    st.subheader("🏆 ¿Cuál es el mejor método para estos datos?")
    mejor_fila = resultados.loc[resultados["MAE"].idxmin()]
    mejor_metodo = mejor_fila["Método"]
    mejor_mae = mejor_fila["MAE"]

    st.success(f"""
    Según el **MAE (Error Absoluto Medio)**, el método con mejor desempeño es **{mejor_metodo}** 
    con un error promedio de {mejor_mae:.2f} unidades de {variable_objetivo}.
    
    **Recomendación:** Utiliza este método como línea base para pronósticos futuros. 
    Si necesitas mayor precisión, considera modelos más avanzados (ARIMA, Suavizamiento exponencial, etc.).
    """)
# ------------------------------------------------------------------------------
# VISTA 2: REGRESIÓN LINEAL (SIMPLE Y MÚLTIPLE)
# ------------------------------------------------------------------------------
with tab2:
    st.header("📊 Regresión Lineal (Simple y Múltiple) - Dinámica")
    st.markdown("Se adapta automáticamente a cualquier archivo CSV/Excel. Pasa el mouse sobre los gráficos para ver valores exactos.")

    # ==================================================
    # INTERPRETACIÓN GENERAL DE REGRESIÓN
    # ==================================================
    with st.expander("📖 ¿Qué es la regresión lineal?", expanded=False):
        st.markdown(r"""
        La **regresión lineal** modela la relación entre una variable dependiente $Y$ (objetivo) y una o más variables independientes $X$ (predictoras).

        ### Regresión Lineal Simple:
        $$
        Y = \beta_0 + \beta_1 X + \varepsilon
        $$
        - $\beta_0$: intercepto (valor de $Y$ cuando $X = 0$)
        - $\beta_1$: coeficiente (cambio en $Y$ por cada unidad de cambio en $X$)
        - $\varepsilon$: error aleatorio

        ### Regresión Lineal Múltiple:
        $$
        Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_k X_k + \varepsilon
        $$

        ### ¿Para qué sirve?
        - Predecir valores futuros de $Y$
        - Identificar qué variables influyen más en $Y$
        - Cuantificar la fuerza de la relación
        """)

    # ==================================================
    # PREPARACIÓN DE DATOS DINÁMICA
    # ==================================================
    columnas_numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    columnas_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(columnas_numericas) == 0:
        st.error("No hay columnas numéricas para realizar regresión.")
        st.stop()

    st.info(f"**Variable objetivo seleccionada:** {variable_objetivo}")

    predictoras_disponibles = [col for col in df.columns if col != variable_objetivo]
    variables_predictoras = st.multiselect(
        "Selecciona variables predictoras (múltiple)",
        predictoras_disponibles,
        default=predictoras_disponibles[:min(3, len(predictoras_disponibles))]
    )
    if len(variables_predictoras) == 0:
        st.error("Selecciona al menos una variable predictora.")
        st.stop()

    datos = df[variables_predictoras + [variable_objetivo]].dropna()
    X = datos[variables_predictoras]
    y = datos[variable_objetivo]

    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    st.caption(f"Datos limpios: {len(datos)} registros | {len(variables_predictoras)} predictoras")

    # ==================================================
    # 1. MATRIZ DE CORRELACIÓN (solo numéricas)
    # ==================================================
    st.subheader("1. Matriz de correlación")
    with st.expander("📖 Interpretación de la matriz de correlación", expanded=False):
        st.markdown(r"""
        La **matriz de correlación** mide la fuerza y dirección de la relación lineal entre pares de variables numéricas.

        - **Coeficiente de correlación de Pearson** $r \in [-1, 1]$:
        - $r > 0$: correlación positiva (una variable aumenta cuando la otra aumenta)
        - $r < 0$: correlación negativa (una aumenta, la otra disminuye)
        - $|r|$ cercano a 1: relación lineal fuerte
        - $|r|$ cercano a 0: relación lineal débil o nula

        **Útil para:** Detectar multicolinealidad (correlaciones altas entre predictoras) y elegir la mejor variable para regresión simple.
        """)

    if len(num_features) >= 2:
        corr_df = datos[num_features + ([variable_objetivo] if variable_objetivo in num_features else [])]
        if len(corr_df.columns) >= 2:
            fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
            ax_corr.set_title("Matriz de correlación (variables numéricas)")
            st.pyplot(fig_corr)
        else:
            st.info("No hay suficientes variables numéricas para correlación.")
    else:
        st.info("Se necesitan al menos 2 variables numéricas para matriz de correlación.")

    # ==================================================
    # 2. REGRESIÓN LINEAL SIMPLE (mejor variable numérica)
    # ==================================================
    st.subheader("2. Regresión Lineal Simple")
    with st.expander("📖 Fórmulas y conceptos de regresión simple", expanded=False):
        st.markdown(r"""
        **Modelo:**
        $$
        \hat{y} = \beta_0 + \beta_1 x
        $$

        **Coeficientes (mínimos cuadrados ordinarios):**
        $$
        \beta_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}, \quad \beta_0 = \bar{y} - \beta_1\bar{x}
        $$

        **R² (coeficiente de determinación):** Proporción de la variabilidad de \( Y \) explicada por el modelo.
        $$
        R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
        $$

        **p-valor:** Probabilidad de que el coeficiente sea cero (no influyente). Generalmente, \( p < 0.05 \) indica significancia estadística.
        """)

    if len(num_features) > 0:
        correlaciones = {}
        for col in num_features:
            if col != variable_objetivo:
                correl = np.corrcoef(datos[col], y)[0,1]
                correlaciones[col] = abs(correl)
        if correlaciones:
            mejor_var = max(correlaciones, key=correlaciones.get)
            st.write(f"**Variable más correlacionada:** {mejor_var} (r = {correlaciones[mejor_var]:.3f})")

            # Gráfico de dispersión interactivo con Plotly
            import plotly.express as px
            fig_disp = px.scatter(datos, x=mejor_var, y=variable_objetivo, 
                                   title=f"Correlación: {mejor_var} vs {variable_objetivo}",
                                   labels={mejor_var: mejor_var, variable_objetivo: variable_objetivo},
                                   trendline="ols")
            fig_disp.update_traces(marker=dict(size=8, opacity=0.6), selector=dict(mode='markers'))
            st.plotly_chart(fig_disp, use_container_width=True)

            # Regresión simple con statsmodels
            try:
                import statsmodels.formula.api as smf
                df_temp = pd.DataFrame({mejor_var: datos[mejor_var], variable_objetivo: y})
                formula = f"{variable_objetivo} ~ {mejor_var}"
                model_simple = smf.ols(formula, data=df_temp).fit()

                st.markdown(f"**Ecuación:** {variable_objetivo} = {model_simple.params[0]:.2f} + {model_simple.params[1]:.2f} * {mejor_var}")
                st.metric("R²", f"{model_simple.rsquared:.3f}")

                coef_df = pd.DataFrame({
                    "Coeficiente": ["Intercepto", mejor_var],
                    "Valor": [model_simple.params[0], model_simple.params[1]],
                    "p-value": [model_simple.pvalues[0], model_simple.pvalues[1]]
                })
                st.dataframe(coef_df, use_container_width=True)

                # Interpretación del p-valor
                if model_simple.pvalues[1] < 0.05:
                    st.success(f"✅ El coeficiente de {mejor_var} es estadísticamente significativo (p < 0.05). La variable influye en {variable_objetivo}.")
                else:
                    st.warning(f"⚠️ El coeficiente de {mejor_var} NO es estadísticamente significativo (p ≥ 0.05). Podría no influir realmente.")
            except Exception as e:
                st.warning(f"No se pudo calcular regresión simple: {e}")
        else:
            st.info("No hay variables numéricas para regresión simple.")
    else:
        st.info("No hay variables numéricas para regresión simple.")

    # ==================================================
    # 3. REGRESIÓN LINEAL MÚLTIPLE
    # ==================================================
    st.subheader("3. Regresión Lineal Múltiple")
    with st.expander("📖 Interpretación de la regresión múltiple", expanded=False):
        st.markdown(r"""
        **Modelo:**
        $$
        \hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_k x_k
        $$

        **Coeficientes $\beta_j$:** Representan el cambio esperado en $y$ cuando $x_j$ aumenta en una unidad, manteniendo las demás variables constantes.

        **R² ajustado:** Penaliza la inclusión de variables innecesarias.  
        **R²:** Proporción de varianza explicada (0 a 1). Valores altos indican mejor ajuste.

        **MSE (Mean Squared Error):**
        $$
        MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2
        $$

        **MAE (Mean Absolute Error):**
        $$
        MAE = \frac{1}{n} \sum |y_i - \hat{y}_i|
        $$

        Ambos miden el error promedio; MSE penaliza más los errores grandes.
        """)

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    preprocesador = ColumnTransformer([
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ])

    modelo = Pipeline([("preprocesador", preprocesador), ("regresion", LinearRegression())])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("R²", f"{r2:.3f}")
    col_r2.metric("MAE", f"{mae:.2f}")
    col_r3.metric("MSE", f"{mse:.2f}")

    # Ecuación real
    reg = modelo.named_steps["regresion"]
    pre = modelo.named_steps["preprocesador"]
    coefs = reg.coef_
    intercepto = reg.intercept_
    vars_modelo = pre.get_feature_names_out()
    ecuacion = f"{variable_objetivo} = {intercepto:.2f}"
    for var, coef in zip(vars_modelo, coefs):
        signo = "+" if coef >= 0 else "-"
        ecuacion += f" {signo} {abs(coef):.2f}({var})"
    with st.expander("Ver ecuación completa del modelo", expanded=False):
        st.code(ecuacion, language="text")
        st.markdown("**Interpretación:** Cada coeficiente indica cuánto cambia el valor estimado de la variable objetivo por cada unidad de aumento en la variable predictora (manteniendo las demás constantes).")

    # Gráfico real vs predicho interactivo (Plotly)
    st.subheader("Real vs Predicho (múltiple)")
    fig_rvp = px.scatter(x=y_test, y=y_pred, 
                         labels={'x': 'Valor real', 'y': 'Valor predicho'},
                         title="Comparación real vs predicción")
    fig_rvp.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                 y=[y_test.min(), y_test.max()],
                                 mode='lines', name='Línea ideal', 
                                 line=dict(color='red', dash='dash')))
    st.plotly_chart(fig_rvp, use_container_width=True)

    # ==================================================
    # 4. MULTICOLINEALIDAD (VIF)
    # ==================================================
    st.subheader("4. Diagnóstico de multicolinealidad (VIF)")
    with st.expander("📖 ¿Qué es el VIF y por qué es importante?", expanded=False):
        st.markdown(r"""
        **VIF (Variance Inflation Factor)** mide cuánto aumenta la varianza de un coeficiente de regresión debido a la correlación con otras variables.

        $$
        VIF = \frac{1}{1 - R_j^2}
        $$

        donde $R_j^2$ es el coeficiente de determinación de la regresión de la variable $j$ sobre las demás predictoras.

        **Interpretación:**
        - VIF = 1 → sin correlación
        - 1 < VIF < 5 → correlación moderada (aceptable)
        - VIF ≥ 5 → correlación alta (preocupante)
        - VIF ≥ 10 → multicolinealidad severa (debe corregirse)

        **Solución:** Eliminar una de las variables correlacionadas o combinarlas.
        """)

    if len(num_features) >= 2:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        X_num = datos[num_features]
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X_num.columns
        vif_data["VIF"] = [variance_inflation_factor(X_num.values, i) for i in range(X_num.shape[1])]
        st.dataframe(vif_data, use_container_width=True)
        
        # Detectar variables con VIF alto
        high_vif = vif_data[vif_data["VIF"] >= 5]
        if not high_vif.empty:
            st.warning(f"⚠️ Variables con posible multicolinealidad (VIF ≥ 5): {', '.join(high_vif['Variable'].tolist())}")
        else:
            st.success("✅ No se detectaron problemas graves de multicolinealidad.")
    else:
        st.info("Se necesitan al menos 2 variables numéricas para calcular VIF.")

    # ==================================================
    # 5. DIAGNÓSTICO DE RESIDUOS
    # ==================================================
    st.subheader("5. Diagnóstico de residuos")
    with st.expander("📖 ¿Por qué analizar los residuos?", expanded=False):
        st.markdown(r"""
        Los **residuos** son las diferencias entre los valores reales y los predichos: 
        $$
        e_i = y_i - \hat{y}_i
        $$

        **Supuestos de la regresión lineal:**
        1. **Normalidad:** Los residuos deben seguir aproximadamente una distribución normal. (Se evalúa con Q-Q plot).
        2. **Homocedasticidad:** La varianza de los residuos debe ser constante a lo largo de los valores predichos. (Se evalúa con gráfico de residuos vs predichos; debe verse una nube sin forma de embudo).
        3. **Independencia:** Los residuos no deben estar correlacionados entre sí (especialmente en series temporales).

        Si estos supuestos se violan, las inferencias (p-valores, intervalos de confianza) pueden no ser fiables.
        """)

    residuos = y_test - y_pred
    fig_res, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Q-Q plot
    import scipy.stats as stats
    stats.probplot(residuos, dist="norm", plot=axes[0])
    axes[0].set_title("Q-Q plot (normalidad)")
    # Residuos vs predichos
    axes[1].scatter(y_pred, residuos, alpha=0.6)
    axes[1].axhline(y=0, color='red', linestyle='--')
    axes[1].set_xlabel("Valores predichos")
    axes[1].set_ylabel("Residuos")
    axes[1].set_title("Homocedasticidad")
    plt.tight_layout()
    st.pyplot(fig_res)

    # Interpretación automática
    if len(residuos) > 0:
        # Prueba de normalidad Shapiro-Wilk (solo si hay suficientes datos)
        if len(residuos) >= 3 and len(residuos) <= 5000:
            _, p_shapiro = stats.shapiro(residuos)
            if p_shapiro > 0.05:
                st.success(f"✅ Los residuos parecen normales (p-valor Shapiro-Wilk = {p_shapiro:.3f} > 0.05).")
            else:
                st.warning(f"⚠️ Los residuos no siguen una distribución normal (p-valor = {p_shapiro:.3f} < 0.05). Considera transformaciones.")
        else:
            st.info("No se realizó prueba de normalidad por tamaño de muestra inadecuado.")
    else:
        st.info("No hay suficientes residuos para diagnóstico.")

    # ==================================================
    # 6. GRÁFICO DE PARES (pairplot)
    # ==================================================
    st.subheader("6. Gráfico de pares (pairplot)")
    with st.expander("📖 ¿Qué muestra un pairplot?", expanded=False):
        st.markdown("""
        El **pairplot** muestra la relación entre cada par de variables numéricas en un solo gráfico matricial.
        - En la diagonal: histogramas o densidad de cada variable.
        - Fuera de la diagonal: diagramas de dispersión (scatter plots) entre dos variables.
        
        **Utilidad:** Identificar relaciones lineales, no lineales, outliers y posibles clusters.
        """)
    if len(num_features) >= 2:
        pair_df = datos[num_features + ([variable_objetivo] if variable_objetivo in num_features else [])]
        fig_pair = sns.pairplot(pair_df, diag_kind='kde', height=2.5)
        st.pyplot(fig_pair)
    else:
        st.info("Se necesitan al menos 2 variables numéricas para pairplot.")

    # ==================================================
    # 7. RELACIÓN ENTRE DOS COLUMNAS (dinámico)
    # ==================================================
    st.subheader("7. Relación entre dos columnas")
    with st.expander("📖 ¿Cómo interpretar esta relación?", expanded=False):
        st.markdown("""
        Este gráfico te permite explorar libremente la relación entre cualquier par de columnas de tu archivo.
        - Si la columna Y es numérica, se muestra un **diagrama de dispersión** (scatter plot).
        - Si la columna Y es categórica, se muestra un **boxplot** para comparar distribuciones.
        
        Pasa el mouse sobre los puntos para ver valores exactos.
        """)
    if len(df.columns) >= 2:
        col_x = st.selectbox("Selecciona columna X (eje horizontal)", df.columns, key="col_x")
        col_y = st.selectbox("Selecciona columna Y (eje vertical)", df.columns, key="col_y")
        if col_x and col_y:
            # Usar plotly para interactividad
            if df[col_y].dtype in ['int64', 'float64']:
                fig_rel = px.scatter(df, x=col_x, y=col_y, title=f"{col_y} vs {col_x}",
                                      labels={col_x: col_x, col_y: col_y})
                st.plotly_chart(fig_rel, use_container_width=True)
            else:
                # Boxplot con plotly
                fig_rel = px.box(df, x=col_x, y=col_y, title=f"{col_y} por {col_x}")
                st.plotly_chart(fig_rel, use_container_width=True)
    else:
        st.info("No hay suficientes columnas para mostrar relación.")

    # ==================================================
    # 8. MATRIZ DE CONFUSIÓN BINARIA (ALTO vs BAJO)
    # ==================================================
    st.subheader("8. Matriz de confusión y métricas de clasificación (binaria)")
    st.markdown("Clasificación binaria usando la mediana como punto de corte: **Alto** (positivo) y **Bajo** (negativo).")

    # Calcular mediana de la variable objetivo (sobre todos los datos de entrenamiento+prueba)
    mediana = y.median()
    
    def convertir_binario(valor):
        return "Alto (Positivo)" if valor > mediana else "Bajo (Negativo)"
    
    # Aplicar a valores reales y predichos del conjunto de prueba
    y_test_bin = y_test.apply(convertir_binario)
    y_pred_bin = pd.Series(y_pred).apply(convertir_binario)
    
    # Calcular matriz de confusión
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
    cm = confusion_matrix(y_test_bin, y_pred_bin, labels=["Alto (Positivo)", "Bajo (Negativo)"])
    
    # Extraer valores
    VP = cm[0,0]  # Verdaderos Positivos (Alto predicho como Alto)
    FP = cm[0,1]  # Falsos Positivos (Bajo predicho como Alto)
    FN = cm[1,0]  # Falsos Negativos (Alto predicho como Bajo)
    VN = cm[1,1]  # Verdaderos Negativos (Bajo predicho como Bajo)
    
    # Métricas
    accuracy = (VP + VN) / (VP + FP + FN + VN)
    precision = VP / (VP + FP) if (VP + FP) > 0 else 0
    recall = VP / (VP + FN) if (VP + FN) > 0 else 0          # Sensibilidad
    specificity = VN / (VN + FP) if (VN + FP) > 0 else 0      # Tasa verdaderos negativos
    vpp = VP / (VP + FP) if (VP + FP) > 0 else 0              # Valor predictivo positivo (igual a precisión)
    vpn = VN / (VN + FN) if (VN + FN) > 0 else 0              # Valor predictivo negativo
    tasa_falsos_positivos = FP / (FP + VN) if (FP + VN) > 0 else 0
    tasa_falsos_negativos = FN / (FN + VP) if (FN + VP) > 0 else 0
    
    # Mostrar matriz de confusión con heatmap
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Alto (Positivo)", "Bajo (Negativo)"],
                yticklabels=["Alto (Positivo)", "Bajo (Negativo)"], ax=ax_cm)
    ax_cm.set_xlabel("Predicción")
    ax_cm.set_ylabel("Valor real")
    ax_cm.set_title("Matriz de confusión")
    st.pyplot(fig_cm)
    
    # Mostrar métricas en formato tabla interpretativa (como la imagen)
    st.markdown("### Métricas de rendimiento")
    
    # Crear un DataFrame con las métricas y su interpretación
    metricas_df = pd.DataFrame({
        "Métrica": [
            "Exactitud (Accuracy)", 
            "Precisión (Precision)", 
            "Sensibilidad (Recall)", 
            "Especificidad (Specificity)", 
            "Valor predictivo positivo (VPP)", 
            "Valor predictivo negativo (VPN)", 
            "Tasa de falsos positivos", 
            "Tasa de falsos negativos"
        ],
        "Valor": [
            f"{accuracy:.2%}", 
            f"{precision:.2%}", 
            f"{recall:.2%}", 
            f"{specificity:.2%}", 
            f"{vpp:.2%}", 
            f"{vpn:.2%}", 
            f"{tasa_falsos_positivos:.2%}", 
            f"{tasa_falsos_negativos:.2%}"
        ],
        "Interpretación": [
            f"Proporción de aciertos totales: {(VP+VN)} / {VP+FP+FN+VN} = {accuracy:.2%}",
            f"De las predicciones 'Alto', cuántas fueron correctas: VP / (VP+FP) = {precision:.2%}",
            f"De los valores 'Alto' reales, cuántos fueron detectados: VP / (VP+FN) = {recall:.2%}",
            f"De los valores 'Bajo' reales, cuántos fueron detectados: VN / (VN+FP) = {specificity:.2%}",
            f"Probabilidad de que un 'Alto' predicho sea realmente 'Alto': {precision:.2%}",
            f"Probabilidad de que un 'Bajo' predicho sea realmente 'Bajo': {vpn:.2%}",
            f"Proporción de 'Bajo' reales clasificados como 'Alto': FP / (FP+VN) = {tasa_falsos_positivos:.2%}",
            f"Proporción de 'Alto' reales clasificados como 'Bajo': FN / (FN+VP) = {tasa_falsos_negativos:.2%}"
        ]
    })
    
    st.dataframe(metricas_df, use_container_width=True)
    
    # Interpretación adicional (como la imagen)
    st.markdown("#### 📌 Interpretación de la matriz")
    st.info(f"""
    - **Punto de corte**: Se usó la mediana de {variable_objetivo} = {mediana:.2f}.
    - **Clase positiva (Alto)**: > {mediana:.2f}
    - **Clase negativa (Bajo)**: ≤ {mediana:.2f}
    
    **Resumen**:
    - El modelo acierta en el **{accuracy:.1%}** de los casos (Exactitud).
    - Cuando predice "Alto", acierta el **{precision:.1%}** de las veces (Precisión).
    - Detecta el **{recall:.1%}** de los "Alto" reales (Sensibilidad).
    - Detecta el **{specificity:.1%}** de los "Bajo" reales (Especificidad).
    """)
    
    # ==================================================
    # 9. PREDICCIÓN MANUAL CON NUEVOS DATOS
    # ==================================================
    st.subheader("9. Predicción manual con nuevos datos")
    st.markdown("Ingresa valores para las variables predictoras y obtén una estimación de la variable objetivo.")

    # Determinar número de columnas (por ejemplo, 3 columnas)
    num_columnas = 3
    cols = st.columns(num_columnas)
    
    with st.form(key="prediccion_form"):
        entrada_usuario = {}
        
        # Asignar cada variable predictora a una columna de forma cíclica
        for i, col in enumerate(variables_predictoras):
            with cols[i % num_columnas]:
                if col in num_features:
                    # Variable numérica
                    valor_default = float(datos[col].mean())
                    entrada_usuario[col] = st.number_input(
                        f"**{col}**", 
                        value=valor_default, 
                        format="%.2f",
                        key=f"num_{col}"
                    )
                elif col in cat_features:
                    # Variable categórica
                    valor_default = datos[col].mode()[0]
                    opciones = datos[col].dropna().unique().tolist()
                    entrada_usuario[col] = st.selectbox(
                        f"**{col}**", 
                        opciones,
                        index=opciones.index(valor_default) if valor_default in opciones else 0,
                        key=f"cat_{col}"
                    )
        
        # Botón de envío (centrado)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            submitted = st.form_submit_button("🔮 Realizar predicción", use_container_width=True)
        
        if submitted:
            nuevo_dato = pd.DataFrame([entrada_usuario])
            prediccion = modelo.predict(nuevo_dato)[0]
            st.success(f"### 📈 Estimación de **{variable_objetivo}**: {prediccion:,.2f}")
            with st.expander("Ver valores ingresados"):
                st.dataframe(nuevo_dato, use_container_width=True)
                
     # ==================================================
    # CONCLUSIONES GENERALES DE REGRESIÓN
    # ==================================================
    st.subheader("📌 Conclusiones del análisis de regresión")

    # Preparar valores con manejo seguro de posibles variables no definidas
    mejor_var_val = mejor_var if 'mejor_var' in locals() else "N/A"
    correl_val = correlaciones[mejor_var] if ('mejor_var' in locals() and mejor_var in correlaciones) else 0.0
    r2_simple_val = model_simple.rsquared if ('model_simple' in locals()) else None
    r2_simple_str = f"{r2_simple_val:.3f}" if r2_simple_val is not None else "N/A"

    # Multicolinealidad
    if 'high_vif' in locals() and not high_vif.empty:
        multicol_msg = "Se detectaron variables con VIF alto"
    else:
        multicol_msg = "No se detectaron problemas graves"

    conclusion = f"""
    - **Regresión simple:** La variable más correlacionada fue **{mejor_var_val}** con un coeficiente de correlación de {correl_val:.3f} y R² = {r2_simple_str}.
    - **Regresión múltiple:** El modelo explica el {r2*100:.1f}% de la variabilidad de {variable_objetivo} (R² = {r2:.3f}). El error promedio (MAE) es de {mae:.2f} unidades.
    - **Multicolinealidad:** {multicol_msg}.
    - **Clasificación:** El modelo acierta en el {accuracy*100:.1f}% de los casos al predecir la categoría (Bajo/Medio/Alto) de {variable_objetivo}.
    """
    st.info(conclusion)
    
    # Exportar PDF (opcional, igual que antes)
    st.subheader("Descargar reporte PDF")
    def generar_pdf():
        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(50, 750, "Reporte de Regresión Lineal")
        pdf.setFont("Helvetica", 11)
        pdf.drawString(50, 720, f"Variable objetivo: {variable_objetivo}")
        pdf.drawString(50, 700, f"R2 del modelo múltiple: {r2:.3f}")
        pdf.drawString(50, 680, f"MAE: {mae:.2f}")
        pdf.drawString(50, 660, f"Accuracy clasificación: {accuracy:.2f}")
        pdf.save()
        buffer.seek(0)
        return buffer
    st.download_button("📄 Descargar PDF", data=generar_pdf(), file_name="reporte_regresion.pdf", mime="application/pdf")