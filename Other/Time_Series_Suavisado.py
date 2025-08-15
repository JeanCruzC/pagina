import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import datetime as dt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

# ─────────── Configuración de página ───────────
st.set_page_config(layout="wide")
st.title("📈 Análisis de Contactos y Ajustes + KPIs, Errores y Recomendaciones")

# ─────────── 1. Carga de datos ───────────
file = st.file_uploader("📂 Carga tu archivo histórico (CSV o Excel)", type=["csv", "xlsx"])
if not file:
    st.stop()
df = pd.read_excel(file) if file.name.endswith("xlsx") else pd.read_csv(file)

# ─────────── 2. Preprocesamiento ───────────
df.columns = df.columns.str.strip().str.lower()
df = df.rename(columns={
    'fecha': 'fecha',
    'tramo': 'intervalo',
    'planif. contactos': 'planificados',
    'contactos': 'reales'
})
df['fecha']      = pd.to_datetime(df['fecha'])
df['intervalo']  = pd.to_datetime(df['intervalo'], format="%H:%M:%S").dt.time
df['dia_semana'] = df['fecha'].dt.day_name()
df['semana_iso'] = df['fecha'].dt.isocalendar().week
df['mes']        = df['fecha'].dt.month
df['nombre_mes'] = df['fecha'].dt.strftime('%B')
df['desvio']     = df['reales'] - df['planificados']
df['desvio_%']   = df['desvio'] / df['planificados'].replace(0, np.nan) * 100

dias_orden = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
df['dia_semana'] = pd.Categorical(df['dia_semana'], categories=dias_orden, ordered=True)

# ─────────── 2.1 Serie continua ───────────
df['_dt'] = df.apply(lambda r: dt.datetime.combine(r['fecha'], r['intervalo']), axis=1)
serie_continua = df.groupby('_dt')[['planificados','reales']].sum().sort_index()

# ─────────── 2.2 Última semana ───────────
ultima_sem = int(df['semana_iso'].max())
_df_last   = df[df['semana_iso']==ultima_sem].copy()
_df_last['_dt'] = _df_last.apply(lambda r: dt.datetime.combine(r['fecha'], r['intervalo']), axis=1)
serie_last = _df_last.groupby('_dt')[['planificados','reales']].sum().sort_index()

# ─────────── 3. Ajustes sugeridos (combinación ponderada) ───────────
N           = 3
proxima_sem = ultima_sem + 1

st.subheader("⚙️ Parámetros de Ponderación")
weight_last = st.slider("Peso de la última semana", 0.0, 1.0, 0.7, 0.05)
weight_prev = st.slider(f"Peso promedio de las últimas {N} semanas", 0.0, 1.0, 0.3, 0.05)

cur = (_df_last
       .groupby(['dia_semana','intervalo'])['desvio_%']
       .mean().reset_index()
       .rename(columns={'desvio_%':'desvio_cur'}))

prev_weeks = sorted(w for w in df['semana_iso'].unique() if w<ultima_sem)[-N:]
df_prev    = df[df['semana_iso'].isin(prev_weeks)]
prev = (df_prev
        .groupby(['dia_semana','intervalo'])['desvio_%']
        .mean().reset_index()
        .rename(columns={'desvio_%':'desvio_prev'}))

aj = pd.merge(cur, prev, on=['dia_semana','intervalo'], how='left')
aj['desvio_prev']     = aj['desvio_prev'].fillna(0)
aj['desvio_comb']     = weight_last*aj['desvio_cur'] + weight_prev*aj['desvio_prev']
aj['ajuste_sugerido'] = (1 + aj['desvio_comb']/100).round(4).map(lambda x: f"{x*100:.0f}%")

st.subheader(f"📆 Ajustes sugeridos para Semana ISO {proxima_sem}")
st.markdown(
    f"**Combinación ponderada:** {int(weight_last*100)}% última semana (ISO {ultima_sem}) + "
    f"{int(weight_prev*100)}% promedio semanas {prev_weeks}"
)
st.dataframe(
    aj[['dia_semana','intervalo','desvio_cur','desvio_prev','desvio_comb','ajuste_sugerido']],
    use_container_width=True
)
st.download_button(
    "📥 Descargar ajustes (.csv)",
    data=aj.to_csv(index=False).encode(),
    file_name=f"ajustes_sem_{proxima_sem}.csv"
)

# ─────────── 4. KPIs de Error ───────────
st.subheader("🔢 KPIs de Planificación vs. Realidad")
y_t_all, y_p_all = serie_continua['reales'], serie_continua['planificados']
mae_all  = mean_absolute_error(y_t_all, y_p_all)
rmse_all = np.sqrt(mean_squared_error(y_t_all, y_p_all))
mape_all = np.mean(np.abs((y_t_all-y_p_all)/y_t_all.replace(0,np.nan))) * 100

y_t_w, y_p_w = serie_last['reales'], serie_last['planificados']
mae_w  = mean_absolute_error(y_t_w, y_p_w)
rmse_w = np.sqrt(mean_squared_error(y_t_w, y_p_w))
mape_w = np.mean(np.abs((y_t_w-y_p_w)/y_t_w.replace(0,np.nan))) * 100

st.markdown(
    f"- **MAE:** Total={mae_all:.0f}, Semana={mae_w:.0f}  \n"
    f"- **RMSE:** Total={rmse_all:.0f}, Semana={rmse_w:.0f}  \n"
    f"- **MAPE:** Total={mape_all:.2f}%, Semana={mape_w:.2f}%"
)

st.subheader("💡 Recomendaciones")
if mape_all>20:
    st.warning("MAPE global >20%: revisar intervalos con mayor desviación.")
elif mape_w>mape_all:
    st.info(f"MAPE semana ({mape_w:.2f}%) > global ({mape_all:.2f}%). Investigar cambios.")
else:
    st.success("Buen alineamiento planificado vs real.")

# ─────────── 4.5 Resumen semanal de Desviación ───────────
st.subheader("🗓️ Resumen semanal de Desviación")
weekly = df.groupby('semana_iso')[['planificados','reales']].sum().reset_index()
weekly['desvío_abs'] = weekly['reales'] - weekly['planificados']
weekly['desvío_%']    = weekly['desvío_abs']/weekly['planificados'].replace(0,np.nan)*100
weekly['MAPE_%']      = weekly['desvío_abs'].abs()/weekly['planificados'].replace(0,np.nan)*100
weekly = weekly.rename(columns={'semana_iso':'Semana ISO'})
weekly[['desvío_%','MAPE_%']] = weekly[['desvío_%','MAPE_%']].applymap(lambda x:f"{x:.2f}%")
st.dataframe(weekly[['Semana ISO','planificados','reales','desvío_abs','desvío_%','MAPE_%']],
             use_container_width=True)

# ─────────── 5. Intervalos con mayor error ───────────
st.subheader("📋 Intervalos con mayor error")
opt = st.selectbox("Mostrar errores de:", ["Total","Última Semana"])
errors = (serie_continua if opt=="Total" else serie_last).copy()
errors['error_abs'] = (errors['reales']-errors['planificados']).abs()
errors['MAPE']      = errors['error_abs']/errors['planificados'].replace(0,np.nan)*100
tab = (errors.reset_index()[['_dt','planificados','reales','error_abs','MAPE']]
       .assign(error_abs=lambda d:d['error_abs'].astype(int),
               MAPE=lambda d:d['MAPE'].map(lambda x:f"{x:.2f}%")))
st.markdown("**MAPE** = |reales − planificados| / planificados × 100")
st.dataframe(tab.sort_values('MAPE',ascending=False).head(10), use_container_width=True)

# ─────────── 6. Heatmap animado ───────────
st.subheader("🔥 Heatmap animado: Desvío % por Semana ISO")
low, high = df['desvio_%'].quantile([0.05,0.95])
fig_heat_anim = px.density_heatmap(
    data_frame=df,
    x='dia_semana',
    y='intervalo',
    z='desvio_%',
    animation_frame='semana_iso',
    category_orders={'dia_semana':dias_orden},
    color_continuous_scale='RdBu_r',
    range_color=(low,high),
    labels={
        'desvio_%':'Desvío %',
        'dia_semana':'Día',
        'intervalo':'Hora',
        'semana_iso':'Semana ISO'
    },
    text_auto='.1f'
)
fig_heat_anim.update_layout(
    yaxis={'categoryorder':'array','categoryarray':sorted(df['intervalo'].astype(str).unique())},
    xaxis={'categoryorder':'array','categoryarray':dias_orden},
    title=(
        f"Desvío % por franja horaria y día (animado)<br>"
        f"(escala limitada a [{low:.1f}%, {high:.1f}%]; valores extremos saturan)"
    )
)
st.plotly_chart(fig_heat_anim, use_container_width=True)

# ─────────── 7. Vista interactiva con anomalías ───────────
st.subheader("🔎 Vista interactiva: Día / Semana / Mes")
vista = st.selectbox("Ver por:", ['Día','Semana','Mes'])
if vista=='Día':
    fig = px.line(
        serie_continua.reset_index(), x='_dt', y=['planificados','reales'],
        title='📅 Contactos Día',
        labels={'_dt':'Fecha y Hora','value':'Volumen','variable':'Tipo'},
        color_discrete_map={'planificados':'red','reales':'blue'}
    ).update_layout(hovermode="x unified", dragmode="zoom")  
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

    decomp = seasonal_decompose(serie_continua['planificados'], model='additive', period=48)
    resid = decomp.resid.dropna(); anoms = resid[np.abs(resid)>3*resid.std()]
    fig_anom = px.line(
        serie_continua.reset_index(), x='_dt', y='planificados',
        title='🔴 Anomalías Día', color_discrete_map={'planificados':'red'}
    )
    fig_anom.add_scatter(
        x=anoms.index, y=serie_continua.loc[anoms.index,'planificados'],
        mode='markers', marker=dict(color='red'), name='Anomalías'
    )
    st.plotly_chart(fig_anom, use_container_width=True)

elif vista=='Semana':
    weekly = df.groupby(['semana_iso','intervalo'])[['planificados','reales']].mean().reset_index()
    melt   = weekly.melt(id_vars=['semana_iso','intervalo'], value_vars=['planificados','reales'],
                         var_name='Tipo', value_name='Volumen')
    fig_week = px.line(
        melt, x='intervalo', y='Volumen', color='Tipo',
        animation_frame='semana_iso', animation_group='Tipo',
        labels={'intervalo':'Hora','semana_iso':'Semana ISO','Volumen':'Contactos','Tipo':'Tipo'},
        title="📆 Curvas horarias por Semana (promedio)"
    ).update_layout(hovermode="x unified")
    st.plotly_chart(fig_week, use_container_width=True)

else:  # Mes
    daily_m    = df.assign(dia=df['fecha'].dt.date)
    monthly_avg = (daily_m
                   .groupby(['nombre_mes','intervalo'])[['planificados','reales']]
                   .mean().reset_index())
    fig = px.line(
        monthly_avg, x='intervalo', y=['planificados','reales'],
        facet_col='nombre_mes', facet_col_wrap=3,
        title='📊 Curva horaria promedio diario por Mes',
        labels={'intervalo':'Hora','value':'Volumen','variable':'Tipo'},
        color_discrete_map={'planificados':'red','reales':'blue'}
    ).update_layout(hovermode="x unified", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
