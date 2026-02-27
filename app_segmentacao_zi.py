# -*- coding: utf-8 -*-
"""
Pipeline Completo de Segmentacao Rodoviaria
Segmentacao (CDA · SHS · MCV) → Agregacao → Clustering (Ward · HDBSCAN)

Dashboard interativo em Streamlit com Plotly.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.sparse import diags
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from datetime import datetime

from homogeneous_segmentation import (
    segment_ids_to_maximize_spatial_heterogeneity,
    segment_ids_to_minimize_coefficient_of_variation,
)

# ======================================================================
#  CONFIGURACAO DA PAGINA
# ======================================================================
st.set_page_config(
    page_title="Pipeline Segmentacao Rodoviaria",
    page_icon="\U0001f6e3\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- CSS --
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 18px; border-radius: 14px; text-align: center;
        color: white; box-shadow: 0 4px 15px rgba(102,126,234,0.3);
        transition: transform 0.2s; margin-bottom: 8px;
    }
    .kpi-card:hover { transform: translateY(-3px); }
    .kpi-value { font-size: 1.8rem; font-weight: 800; line-height: 1.2; }
    .kpi-label { font-size: 0.76rem; opacity: 0.85; margin-top: 4px; }
    .kpi-green { background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
                 box-shadow: 0 4px 15px rgba(46,204,113,0.3); }
    .kpi-red   { background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
                 box-shadow: 0 4px 15px rgba(231,76,60,0.3); }
    .kpi-blue  { background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
                 box-shadow: 0 4px 15px rgba(52,152,219,0.3); }
    .kpi-purple{ background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
                 box-shadow: 0 4px 15px rgba(155,89,182,0.3); }
    .kpi-orange{ background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
                 box-shadow: 0 4px 15px rgba(243,156,18,0.3); }
    .kpi-teal  { background: linear-gradient(135deg, #1abc9c 0%, #16a085 100%);
                 box-shadow: 0 4px 15px rgba(26,188,156,0.3); }
    .story-box {
        background: #f8f9fa; border-left: 5px solid #667eea;
        padding: 15px 20px; border-radius: 0 10px 10px 0;
        margin: 12px 0; font-size: 0.92rem; color: #333;
    }
    .pipeline-step {
        background: linear-gradient(135deg, #f8f9fa 0%, #e8eaf6 100%);
        border-left: 5px solid #667eea; padding: 15px 20px;
        border-radius: 0 12px 12px 0; margin: 10px 0;
    }
    [data-testid="stSidebar"] { background: #f0f2f6; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    .dataframe { font-size: 0.85rem !important; }
    hr { border-top: 2px solid #667eea30; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ======================================================================
#  FUNCOES AUXILIARES — UI
# ======================================================================

def kpi_card(value, label, css_class=""):
    return f"""<div class="kpi-card {css_class}">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>"""


def story(text):
    st.markdown(f'<div class="story-box">{text}</div>', unsafe_allow_html=True)


def pipeline_step(title, text):
    st.markdown(f"""<div class="pipeline-step">
        <b>{title}</b><br><span style="font-size:0.88rem;">{text}</span>
    </div>""", unsafe_allow_html=True)


# ======================================================================
#  FUNCOES — SEGMENTACAO
# ======================================================================

def merge_short_segments(limites, df_local, col_est, min_m):
    """Funde segmentos CDA menores que min_m ao vizinho mais adequado."""
    limites = list(limites)
    while True:
        lengths = [
            df_local.loc[limites[i + 1], col_est] - df_local.loc[limites[i], col_est]
            for i in range(len(limites) - 1)
        ]
        if min(lengths) >= min_m or len(limites) <= 2:
            break
        short_idx = int(np.argmin(lengths))
        if short_idx == 0:
            limites.pop(1)
        elif short_idx == len(lengths) - 1:
            limites.pop(-2)
        else:
            left_combo = lengths[short_idx - 1] + lengths[short_idx]
            right_combo = lengths[short_idx] + lengths[short_idx + 1]
            limites.pop(short_idx if left_combo <= right_combo else short_idx + 1)
    return np.array(limites)


def split_long_segments(limites, df_local, col_est, max_m):
    """Subdivide segmentos CDA maiores que max_m."""
    new_limites = [limites[0]]
    for i in range(len(limites) - 1):
        idx_ini, idx_fim = limites[i], limites[i + 1]
        est_ini = df_local.loc[idx_ini, col_est]
        est_fim = df_local.loc[idx_fim, col_est]
        seg_len = est_fim - est_ini
        if seg_len > max_m:
            n_parts = int(np.ceil(seg_len / max_m))
            target_len = seg_len / n_parts
            subset = df_local.loc[idx_ini:idx_fim]
            for p in range(1, n_parts):
                target_est = est_ini + p * target_len
                closest_idx = (subset[col_est] - target_est).abs().idxmin()
                if closest_idx not in new_limites:
                    new_limites.append(closest_idx)
        new_limites.append(idx_fim)
    return np.array(sorted(set(new_limites)))


def segmentacao_cda(df, col_est, col_defl, media_defl, fp_kwargs, seg_min_m, seg_max_m):
    """Executa segmentacao CDA (Zi / CUSUM).

    Args:
        df: DataFrame com dados brutos.
        col_est: Nome da coluna de estacao.
        col_defl: Nome da coluna de deflexao.
        media_defl: Media global de deflexao.
        fp_kwargs: Parametros para scipy.signal.find_peaks.
        seg_min_m: Comprimento minimo do segmento (m).
        seg_max_m: Comprimento maximo do segmento (m).

    Returns:
        seg_df: DataFrame de segmentos agregados.
        zi: Array da serie Zi.
        picos_raw, vales_raw: Indices brutos de picos e vales.
        picos, vales: Indices finais retidos apos merge/split.
        limites_idx: Array de indices limites dos segmentos.
    """
    col_diff = 'Diferenca'
    col_zi = 'Diferenca Acumulada (Zi)'
    df[col_diff] = df[col_defl] - media_defl
    df[col_zi] = df[col_diff].cumsum()
    zi = df[col_zi].values

    picos_raw, _ = find_peaks(zi, **fp_kwargs)
    vales_raw, _ = find_peaks(-zi, **fp_kwargs)
    limites_ini = np.sort(np.concatenate(([0], picos_raw, vales_raw, [len(df) - 1])))

    limites_merged = merge_short_segments(limites_ini, df, col_est, seg_min_m)
    limites_idx = split_long_segments(limites_merged, df, col_est, seg_max_m)

    picos = np.array(sorted(set(picos_raw) & set(limites_idx)))
    vales = np.array(sorted(set(vales_raw) & set(limites_idx)))

    segmentos = []
    for i in range(len(limites_idx) - 1):
        ii, fi = limites_idx[i], limites_idx[i + 1]
        est_i, est_f = float(df.loc[ii, col_est]), float(df.loc[fi, col_est])
        defl_seg = df.loc[ii:fi, col_defl]
        dm = defl_seg.mean()
        ds = defl_seg.std(ddof=1) if len(defl_seg) > 1 else 0.0
        cv = (ds / dm * 100) if dm != 0 else 0.0
        segmentos.append({
            'SH': i + 1, 'Inicio (m)': est_i, 'Fim (m)': est_f,
            'Comprimento (m)': est_f - est_i,
            'Zi_fim': round(float(df.loc[fi, col_zi]), 1),
            'Defl. Media': round(dm, 1), 'Desvio Padrao': round(ds, 1),
            'CV (%)': round(cv, 1), 'Defl. Caract.': round(dm + ds, 1),
            'Defl. Max': round(float(defl_seg.max()), 1),
            'N_pontos': len(defl_seg),
        })

    return (pd.DataFrame(segmentos), zi, picos_raw, vales_raw,
            picos, vales, limites_idx, col_zi)


def segmentacao_shs_mcv(df, col_est, col_defl, seg_min_m, seg_max_m, method='shs'):
    """Executa segmentacao SHS ou MCV.

    Args:
        df: DataFrame com dados brutos (deve conter slk_from/slk_to).
        col_est: Nome da coluna de estacao.
        col_defl: Nome da coluna de deflexao.
        seg_min_m: Comprimento minimo (m).
        seg_max_m: Comprimento maximo (m).
        method: 'shs' ou 'mcv'.

    Returns:
        seg_df: DataFrame de segmentos agregados.
    """
    seg_range = (float(seg_min_m), float(seg_max_m))
    col_seg = f'seg_{method}'

    func = (segment_ids_to_maximize_spatial_heterogeneity if method == 'shs'
            else segment_ids_to_minimize_coefficient_of_variation)

    df[col_seg] = func(
        data=df, measure=("slk_from", "slk_to"),
        variable_column_names=[col_defl],
        allowed_segment_length_range=seg_range,
    )

    seg_groups = df.groupby(col_seg)
    segmentos = []
    for seg_id, grp in seg_groups:
        est_ini, est_fim = grp[col_est].iloc[0], grp[col_est].iloc[-1]
        dm = grp[col_defl].mean()
        ds = grp[col_defl].std(ddof=1) if len(grp) > 1 else 0.0
        cv = (ds / dm * 100) if dm != 0 else 0.0
        segmentos.append({
            'SH': int(seg_id), 'Inicio (m)': est_ini, 'Fim (m)': est_fim,
            'Comprimento (m)': est_fim - est_ini,
            'Defl. Media': round(dm, 1), 'Desvio Padrao': round(ds, 1),
            'CV (%)': round(cv, 1), 'Defl. Caract.': round(dm + ds, 1),
            'Defl. Max': round(float(grp[col_defl].max()), 1),
            'N_pontos': len(grp),
        })
    return pd.DataFrame(segmentos)


# ======================================================================
#  FUNCOES — AGREGACAO + CLUSTERING
# ======================================================================

def agregar_segmentos(seg_df):
    """Garante que o DataFrame de segmentos tem as colunas necessarias
    para a etapa de clustering. Retorna copia com colunas padronizadas.

    Colunas garantidas: Defl. Media, Desvio Padrao, Defl. Max,
                        Comprimento (m), N_pontos.
    """
    required = ['Defl. Media', 'Desvio Padrao', 'Defl. Max', 'Comprimento (m)', 'N_pontos']
    out = seg_df.copy()
    for col in required:
        if col not in out.columns:
            out[col] = 0.0
    return out


def aplicar_clustering(seg_df, distance_threshold=2.0,
                       hdbscan_min_cluster=3, hdbscan_min_samples=2):
    """Aplica Ward (com restricao espacial) e HDBSCAN sobre os segmentos.

    Args:
        seg_df: DataFrame de segmentos (uma linha = um SH).
        distance_threshold: Limiar de distancia para Ward.
        hdbscan_min_cluster: Tamanho minimo de cluster HDBSCAN.
        hdbscan_min_samples: Amostras minimas HDBSCAN.

    Returns:
        seg_df com colunas adicionais: cluster_ward, cluster_hdbscan, is_outlier.
    """
    out = seg_df.copy()
    feature_cols = ['Defl. Media', 'Desvio Padrao', 'Defl. Max', 'Comprimento (m)']
    available = [c for c in feature_cols if c in out.columns]
    if len(available) < 2 or len(out) < 3:
        out['cluster_ward'] = 0
        out['cluster_hdbscan'] = 0
        out['is_outlier'] = False
        return out

    X = out[available].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Ward com restricao de contiguidade espacial ──
    n = len(X_scaled)
    conn = diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1],
                 shape=(n, n), format='csr')

    ward = AgglomerativeClustering(
        n_clusters=None, linkage='ward', connectivity=conn,
        distance_threshold=distance_threshold,
    )
    out['cluster_ward'] = ward.fit_predict(X_scaled)

    # ── HDBSCAN ──
    min_cs = min(hdbscan_min_cluster, max(2, n // 3))
    min_s = min(hdbscan_min_samples, min_cs)
    hdb = HDBSCAN(min_cluster_size=min_cs, min_samples=min_s)
    labels = hdb.fit_predict(X_scaled)
    out['cluster_hdbscan'] = labels
    out['is_outlier'] = labels == -1

    return out


# ======================================================================
#  FUNCOES — VISUALIZACAO
# ======================================================================

SEG_COLORS_HEX = [
    '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854',
    '#ffd92f', '#e5c494', '#b3b3b3', '#1b9e77', '#d95f02',
    '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d',
]

COLOR_PRIMARY = '#667eea'
COLOR_SEC = '#764ba2'
COLOR_RED = '#e74c3c'
COLOR_GREEN = '#27ae60'
COLOR_YELLOW = '#f39c12'
COLOR_TEAL = '#1abc9c'


def rgba_from_hex(hex_color, alpha=0.3):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def render_segmentation_charts(df, seg_df, col_est, col_defl, media_defl, method_key):
    """Graficos comuns para qualquer metodo de segmentacao."""

    # KPIs
    cols_kpi = st.columns(5)
    with cols_kpi[0]:
        st.markdown(kpi_card(f"{len(seg_df)}", "Segmentos"), unsafe_allow_html=True)
    with cols_kpi[1]:
        st.markdown(kpi_card(f"{seg_df['Comprimento (m)'].sum():,.0f} m",
                             "Extensao", "kpi-green"), unsafe_allow_html=True)
    with cols_kpi[2]:
        st.markdown(kpi_card(f"{seg_df['Comprimento (m)'].mean():,.0f} m",
                             "Compr. Medio", "kpi-blue"), unsafe_allow_html=True)
    with cols_kpi[3]:
        st.markdown(kpi_card(f"{seg_df['CV (%)'].mean():.1f}%",
                             "CV Medio", "kpi-purple"), unsafe_allow_html=True)
    with cols_kpi[4]:
        st.markdown(kpi_card(f"{media_defl:.1f}",
                             "Defl. Media Global", "kpi-orange"), unsafe_allow_html=True)

    # Deflexao colorida por segmento
    st.markdown(f"#### Deflexao por Segmento — {method_key}")
    fig = go.Figure()
    for i, row in seg_df.iterrows():
        mask = (df[col_est] >= row['Inicio (m)']) & (df[col_est] <= row['Fim (m)'])
        c = SEG_COLORS_HEX[i % len(SEG_COLORS_HEX)]
        fig.add_trace(go.Bar(
            x=df.loc[mask, col_est], y=df.loc[mask, col_defl],
            marker_color=c, opacity=0.7,
            name=f"SH {row['SH']} ({row['Comprimento (m)']:,.0f}m)",
        ))
    fig.add_hline(y=media_defl, line_dash='dash', line_color=COLOR_RED, line_width=2,
                  annotation_text=f'Media = {media_defl:.1f}')
    fig.update_layout(xaxis_title='Estacao (m)', yaxis_title='Deflexao (0,01mm)',
                      template='plotly_white', height=400, barmode='stack',
                      legend=dict(orientation='h', y=-0.18, font_size=9),
                      margin=dict(t=30, b=60))
    st.plotly_chart(fig, use_container_width=True)

    # CV por segmento
    st.markdown(f"#### CV (%) por Segmento — {method_key}")
    cv_colors = [COLOR_GREEN if cv <= 25 else COLOR_YELLOW if cv <= 40 else COLOR_RED
                 for cv in seg_df['CV (%)']]
    fig_cv = go.Figure(go.Bar(
        x=seg_df['SH'].apply(lambda x: f'SH {x}'), y=seg_df['CV (%)'],
        marker_color=cv_colors, text=seg_df['CV (%)'].apply(lambda v: f'{v:.1f}%'),
        textposition='outside', showlegend=False))
    fig_cv.add_hline(y=25, line_dash='dash', line_color=COLOR_GREEN, line_width=2,
                     annotation_text="Austroads 25%")
    fig_cv.update_layout(template='plotly_white', height=350,
                         xaxis_title='Segmento', yaxis_title='CV (%)',
                         margin=dict(t=20, b=40))
    st.plotly_chart(fig_cv, use_container_width=True)

    # Tabela
    st.markdown(f"#### Tabela de Segmentos — {method_key}")
    fmt = {'Inicio (m)': '{:,.0f}', 'Fim (m)': '{:,.0f}',
           'Comprimento (m)': '{:,.0f}', 'CV (%)': '{:.1f}',
           'Defl. Media': '{:.1f}', 'Desvio Padrao': '{:.1f}',
           'Defl. Caract.': '{:.1f}', 'Defl. Max': '{:.1f}'}
    if 'Zi_fim' in seg_df.columns:
        fmt['Zi_fim'] = '{:.1f}'
    st.dataframe(
        seg_df.style
            .background_gradient(subset=['Comprimento (m)'], cmap='Blues')
            .background_gradient(subset=['Defl. Media'], cmap='RdYlGn_r')
            .background_gradient(subset=['CV (%)'], cmap='YlOrRd')
            .format(fmt),
        height=380)


def render_clustering_charts(seg_df, method_key):
    """Graficos e KPIs para a etapa de clustering."""

    st.markdown(f"### Clustering — {method_key}")

    n_ward = seg_df['cluster_ward'].nunique()
    n_hdb = seg_df.loc[seg_df['cluster_hdbscan'] >= 0, 'cluster_hdbscan'].nunique()
    n_outliers = int(seg_df['is_outlier'].sum())

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(kpi_card(str(n_ward), "Clusters Ward", "kpi-blue"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card(str(n_hdb), "Clusters HDBSCAN", "kpi-purple"), unsafe_allow_html=True)
    with c3:
        css = 'kpi-red' if n_outliers > 0 else 'kpi-green'
        st.markdown(kpi_card(str(n_outliers), "Outliers (HDBSCAN)", css), unsafe_allow_html=True)

    # ── Ward: deflexao media colorida por cluster ──
    st.markdown("#### Deflexao Media por Cluster — Ward")
    story("Ward Agglomerative Clustering com restricao de contiguidade espacial 1D. "
          "Segmentos adjacentes sao agrupados em clusters de comportamento similar.")
    fig_w = go.Figure()
    for cid in sorted(seg_df['cluster_ward'].unique()):
        mask = seg_df['cluster_ward'] == cid
        sub = seg_df[mask]
        fig_w.add_trace(go.Bar(
            x=sub['SH'].apply(lambda x: f'SH {x}'), y=sub['Defl. Media'],
            name=f'Cluster {cid}',
            marker_color=SEG_COLORS_HEX[int(cid) % len(SEG_COLORS_HEX)],
            text=sub['Defl. Media'].apply(lambda v: f'{v:.0f}'), textposition='outside',
        ))
    fig_w.update_layout(template='plotly_white', height=380, barmode='group',
                        xaxis_title='Segmento', yaxis_title='Defl. Media (0,01mm)',
                        legend=dict(orientation='h', y=-0.15),
                        margin=dict(t=20, b=50))
    st.plotly_chart(fig_w, use_container_width=True)

    # ── HDBSCAN: scatter com outliers ──
    st.markdown("#### Perfilamento HDBSCAN — Outliers")
    story("HDBSCAN identifica clusters de densidade e marca segmentos anomalos (cluster = -1) "
          "como outliers. Uteis para priorizar inspecoes.")

    fig_h = go.Figure()
    for cid in sorted(seg_df['cluster_hdbscan'].unique()):
        mask = seg_df['cluster_hdbscan'] == cid
        sub = seg_df[mask]
        name = f'Cluster {cid}' if cid >= 0 else 'Outlier'
        color = COLOR_RED if cid < 0 else SEG_COLORS_HEX[int(cid) % len(SEG_COLORS_HEX)]
        symbol = 'x' if cid < 0 else 'circle'
        fig_h.add_trace(go.Scatter(
            x=sub['Defl. Media'], y=sub['CV (%)'],
            mode='markers+text', name=name,
            marker=dict(size=14 if cid < 0 else 10, color=color, symbol=symbol,
                        line=dict(width=1.5, color='white')),
            text=sub['SH'].apply(lambda x: f'SH{x}'),
            textposition='top center', textfont=dict(size=8),
        ))
    fig_h.update_layout(template='plotly_white', height=420,
                        xaxis_title='Deflexao Media (0,01mm)', yaxis_title='CV (%)',
                        legend=dict(orientation='h', y=-0.12),
                        margin=dict(t=20, b=50))
    st.plotly_chart(fig_h, use_container_width=True)

    # ── Ward: mapa de segmentos ──
    st.markdown("#### Mapa de Segmentos por Cluster Ward")
    fig_map = go.Figure()
    for _, row in seg_df.iterrows():
        cid = int(row['cluster_ward'])
        c = SEG_COLORS_HEX[cid % len(SEG_COLORS_HEX)]
        fig_map.add_shape(type='rect',
                          x0=row['Inicio (m)'], x1=row['Fim (m)'],
                          y0=0, y1=row['Defl. Media'],
                          fillcolor=c, opacity=0.5,
                          line=dict(width=1, color=c))
        fig_map.add_annotation(x=(row['Inicio (m)'] + row['Fim (m)']) / 2,
                               y=row['Defl. Media'] + 2,
                               text=f"SH{int(row['SH'])}<br>C{cid}",
                               showarrow=False, font=dict(size=8))
    # Outlier markers
    outliers = seg_df[seg_df['is_outlier']]
    if len(outliers):
        fig_map.add_trace(go.Scatter(
            x=(outliers['Inicio (m)'] + outliers['Fim (m)']) / 2,
            y=outliers['Defl. Media'],
            mode='markers', name='Outlier HDBSCAN',
            marker=dict(symbol='x', size=16, color=COLOR_RED,
                        line=dict(width=2, color='white')),
        ))
    fig_map.update_layout(template='plotly_white', height=350,
                          xaxis_title='Estacao (m)', yaxis_title='Defl. Media',
                          showlegend=True, margin=dict(t=20, b=40))
    st.plotly_chart(fig_map, use_container_width=True)

    # ── Tabela de features enriquecidos ──
    st.markdown("#### Features Enriquecidos")
    show_cols = ['SH', 'Inicio (m)', 'Fim (m)', 'Comprimento (m)',
                 'Defl. Media', 'Desvio Padrao', 'CV (%)', 'Defl. Max',
                 'N_pontos', 'cluster_ward', 'cluster_hdbscan', 'is_outlier']
    show_cols = [c for c in show_cols if c in seg_df.columns]
    st.dataframe(seg_df[show_cols].style
                 .background_gradient(subset=['Defl. Media'], cmap='RdYlGn_r')
                 .background_gradient(subset=['cluster_ward'], cmap='tab10')
                 .applymap(lambda v: 'background-color: #fdcece' if v else '',
                           subset=['is_outlier']),
                 height=380)


# ======================================================================
#  FUNCOES — EXPORTACAO EXCEL
# ======================================================================

def gerar_excel_pipeline(df_data, results_dict, col_est, col_defl,
                         zi_data=None):
    """Gera Excel formatado com todas as abas do pipeline."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.chart import BarChart, Reference
    from openpyxl.chart.label import DataLabelList
    from openpyxl.utils import get_column_letter
    from openpyxl.formatting.rule import DataBarRule

    COR_H = 'FF4A6FA5'; COR_H2 = 'FF667EEA'; COR_AC = 'FF764BA2'
    COR_LB = 'FFF2F4F8'; COR_W = 'FFFFFFFF'

    ft = Font(name='Calibri', bold=True, size=16, color='FFFFFF')
    fh = Font(name='Calibri', bold=True, size=11, color='FFFFFF')
    fb = Font(name='Calibri', size=11)
    fH = PatternFill('solid', fgColor=COR_H)
    fH2 = PatternFill('solid', fgColor=COR_H2)
    fAC = PatternFill('solid', fgColor=COR_AC)
    fL = PatternFill('solid', fgColor=COR_LB)
    fW = PatternFill('solid', fgColor=COR_W)
    ac = Alignment(horizontal='center', vertical='center', wrap_text=True)
    bd = Border(*[Side(style='thin', color='D0D0D0')] * 4)

    wb = Workbook()

    def write_sheet(ws, title, seg_df, fill_h):
        n_cols = len(seg_df.columns)
        end_let = get_column_letter(min(n_cols, 12))
        ws.merge_cells(f'A1:{end_let}2')
        ws['A1'].value = title; ws['A1'].font = ft; ws['A1'].fill = fH; ws['A1'].alignment = ac
        cols = list(seg_df.columns)
        for j, cn in enumerate(cols, 1):
            c = ws.cell(row=4, column=j, value=cn)
            c.font = fh; c.fill = fill_h; c.alignment = ac; c.border = bd
            ws.column_dimensions[get_column_letter(j)].width = 18
        for i, rd in seg_df.iterrows():
            for j, cn in enumerate(cols, 1):
                v = rd[cn]
                c = ws.cell(row=5 + i, column=j,
                            value=float(v) if isinstance(v, (int, float, np.integer, np.floating, bool)) else str(v))
                c.font = fb; c.alignment = ac; c.border = bd
                c.fill = fL if i % 2 == 0 else fW
                if cn in ['Inicio (m)', 'Fim (m)', 'Comprimento (m)']:
                    c.number_format = '#,##0'
                elif cn in ['Defl. Media', 'Desvio Padrao', 'CV (%)', 'Defl. Caract.',
                            'Zi_fim', 'Defl. Max']:
                    c.number_format = '0.0'
        lr = 5 + len(seg_df) - 1
        if 'Defl. Media' in cols and lr >= 5:
            defl_col = cols.index('Defl. Media') + 1
            ch = BarChart(); ch.type = 'col'; ch.style = 10
            ch.title = 'Deflexao Media por SH'; ch.y_axis.title = 'Deflexao'
            ch.width = 22; ch.height = 12
            ch.add_data(Reference(ws, min_col=defl_col, min_row=4, max_row=lr),
                        titles_from_data=True)
            ch.set_categories(Reference(ws, min_col=1, min_row=5, max_row=lr))
            ch.series[0].graphicalProperties.solidFill = COR_H2[2:]
            ch.series[0].dLbls = DataLabelList()
            ch.series[0].dLbls.showVal = True; ch.series[0].dLbls.numFmt = '0.0'
            ch.legend = None
            ws.add_chart(ch, f'A{lr + 3}')

    first = True
    for mkey, seg_df in results_dict.items():
        if seg_df is None or len(seg_df) == 0:
            continue
        if first:
            ws = wb.active; ws.title = f'Seg {mkey}'; first = False
        else:
            ws = wb.create_sheet(f'Seg {mkey}')
        fill = fH2 if mkey == 'CDA' else fAC
        write_sheet(ws, f'PIPELINE — Segmentacao + Clustering ({mkey})', seg_df, fill)

    # Dados brutos
    ws_raw = wb.create_sheet('Dados Brutos')
    cols_export = [c for c in df_data.columns
                   if c not in ['slk_from', 'slk_to', 'seg_shs', 'seg_mcv',
                                'Diferenca', 'Diferenca Acumulada (Zi)']]
    for j, cn in enumerate(cols_export, 1):
        c = ws_raw.cell(row=1, column=j, value=cn)
        c.font = fh; c.fill = fH2; c.alignment = ac; c.border = bd
        ws_raw.column_dimensions[get_column_letter(j)].width = 20
    for i, rd in df_data.iterrows():
        for j, cn in enumerate(cols_export, 1):
            v = rd[cn]
            ws_raw.cell(row=2 + i, column=j,
                        value=float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)

    buf = BytesIO()
    wb.save(buf); buf.seek(0)
    return buf.getvalue()


# ======================================================================
#  SIDEBAR
# ======================================================================
with st.sidebar:
    st.markdown("## \U0001f6e3\ufe0f Pipeline de Segmentacao")
    st.caption("Segmentacao → Agregacao → Clustering")
    st.divider()

    uploaded_file = st.file_uploader(
        "**Carregar planilha**", type=["xlsx", "xls", "csv"],
        help="Arquivo com colunas Estacao (m) e Deflexao (0,01mm)")

    st.divider()
    st.markdown("### Colunas")
    col_estacao_name = st.text_input("Coluna **Estacao**", value="Estação (m)")
    col_deflexao_name = st.text_input("Coluna **Deflexao**", value="Deflexão (0,01mm)")

    st.divider()
    st.markdown("### 1. Segmentacao")
    metodos = st.multiselect(
        "Metodo(s)",
        options=["CDA (CUSUM/Zi)", "SHS (Spatial Heterogeneity)", "MCV (Minimize CV)"],
        default=["CDA (CUSUM/Zi)", "SHS (Spatial Heterogeneity)", "MCV (Minimize CV)"],
    )

    seg_min_m = st.number_input("Comprimento MINIMO (m)", value=800, min_value=50, step=50)
    seg_max_m = st.number_input("Comprimento MAXIMO (m)", value=5000, min_value=500, step=100)

    run_cda = "CDA (CUSUM/Zi)" in metodos
    if run_cda:
        st.divider()
        st.markdown("#### Params CDA (`find_peaks`)")
        fp_distance = st.slider("distance", 1, 50, 5, 1)
        fp_height = st.number_input("height", value=0.0, step=1.0, format="%.1f")
        fp_prominence = st.number_input("prominence", value=0.0, min_value=0.0, step=1.0, format="%.1f")
        fp_width = st.number_input("width", value=0.0, min_value=0.0, step=1.0, format="%.1f")
        fp_threshold = st.number_input("threshold", value=0.0, min_value=0.0, step=0.5, format="%.1f")
        fp_plateau_size = st.number_input("plateau_size", value=0, min_value=0, step=1)

    run_shs = "SHS (Spatial Heterogeneity)" in metodos
    run_mcv = "MCV (Minimize CV)" in metodos

    st.divider()
    st.markdown("### 2. Clustering")
    ward_distance = st.slider("Ward distance_threshold", 0.5, 10.0, 2.0, 0.1,
                              help="Limiar de distancia para corte da dendrograma Ward.")
    hdb_min_cluster = st.slider("HDBSCAN min_cluster_size", 2, 20, 3, 1)
    hdb_min_samples = st.slider("HDBSCAN min_samples", 1, 10, 2, 1)

    st.divider()
    st.caption(f"Gerado em {datetime.now():%d/%m/%Y %H:%M}")


# ======================================================================
#  HERO HEADER
# ======================================================================
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 28px 36px; border-radius: 16px; margin-bottom: 20px;
            box-shadow: 0 8px 30px rgba(102,126,234,0.35);">
    <h1 style="color: white; margin: 0; font-size: 1.9rem;">
        Pipeline de Segmentacao Rodoviaria
    </h1>
    <p style="color: rgba(255,255,255,0.85); font-size: 1rem; margin-top: 6px;">
        <b>Entrada</b> → <b>Segmentacao</b> (CDA · SHS · MCV) →
        <b>Agregacao</b> → <b>Clustering</b> (Ward · HDBSCAN) →
        <b>Features Enriquecidos</b>
    </p>
</div>
""", unsafe_allow_html=True)


# ======================================================================
#  PROCESSAMENTO
# ======================================================================
if uploaded_file is None:
    st.info("Carregue um arquivo Excel ou CSV na barra lateral para iniciar.")

    pipeline_step("1. Entrada de Dados",
                  "Upload de planilha Excel/CSV com colunas Estacao (m) e Deflexao (0,01mm).")
    pipeline_step("2. Segmentacao",
                  "Tres metodos paralelos: CDA (CUSUM/Zi), SHS e MCV. Geram seg_id por linha.")
    pipeline_step("3. Agregacao",
                  "Calculo de Defl. Media, Desvio Padrao, Max, CV, N_pontos por segmento.")
    pipeline_step("4. Clustering",
                  "Ward (contiguidade espacial) + HDBSCAN (deteccao de outliers).")
    pipeline_step("5. Features Enriquecidos",
                  "DataFrame final com cluster_ward, cluster_hdbscan, is_outlier.")
    pipeline_step("6. Modelo Preditivo (Futuro)",
                  "Placeholder para XGBoost/RF: previsao de IRI, necessidade de intervencao.")
    st.stop()

if not metodos:
    st.warning("Selecione ao menos um metodo.")
    st.stop()

# ── Leitura ──
fname = uploaded_file.name
try:
    df = pd.read_csv(uploaded_file, sep=None, engine='python') if fname.endswith('.csv') \
        else pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Erro ao ler o arquivo: {e}"); st.stop()

col_est, col_defl = col_estacao_name, col_deflexao_name
if col_est not in df.columns or col_defl not in df.columns:
    st.error(f"Colunas nao encontradas. Disponiveis: {list(df.columns)}"); st.stop()

for c in [col_est, col_defl]:
    if df[c].dtype == 'O':
        df[c] = df[c].astype(str).str.replace(',', '.').astype(float)
    df[c] = pd.to_numeric(df[c], errors='coerce')
df.dropna(subset=[col_est, col_defl], inplace=True)
df.sort_values(col_est, inplace=True)
df.reset_index(drop=True, inplace=True)

# Filtro de estacao
with st.sidebar:
    st.divider()
    st.markdown("#### Filtro de Estacao")
    est_min_d, est_max_d = float(df[col_est].min()), float(df[col_est].max())
    est_range = st.slider("Intervalo (m)", est_min_d, est_max_d,
                          (est_min_d, est_max_d), step=10.0, format="%.0f")
df = df[(df[col_est] >= est_range[0]) & (df[col_est] <= est_range[1])].copy()
df.reset_index(drop=True, inplace=True)
if len(df) < 3:
    st.error("Poucos registros apos o filtro."); st.stop()

media_deflexao = df[col_defl].mean()


# ======================================================================
#  EXECUTAR PIPELINE
# ======================================================================
seg_results = {}  # method_key → seg_df (com clusters)
zi_data = {}      # extras CDA

# ── Preparar slk para SHS/MCV ──
if run_shs or run_mcv:
    step = df[col_est].diff().median()
    df['slk_from'] = df[col_est]
    df['slk_to'] = df[col_est].shift(-1).fillna(df[col_est].iloc[-1] + step)

# ── CDA ──
if run_cda:
    fp_kwargs = {'distance': max(1, fp_distance)}
    if fp_height > 0: fp_kwargs['height'] = fp_height
    if fp_prominence > 0: fp_kwargs['prominence'] = fp_prominence
    if fp_width > 0: fp_kwargs['width'] = fp_width
    if fp_threshold > 0: fp_kwargs['threshold'] = fp_threshold
    if fp_plateau_size > 0: fp_kwargs['plateau_size'] = fp_plateau_size

    (seg_cda, zi, picos_raw, vales_raw, picos, vales,
     limites_idx, col_zi) = segmentacao_cda(
        df, col_est, col_defl, media_deflexao, fp_kwargs, seg_min_m, seg_max_m)

    seg_cda = agregar_segmentos(seg_cda)
    seg_cda = aplicar_clustering(seg_cda, ward_distance, hdb_min_cluster, hdb_min_samples)
    seg_results['CDA'] = seg_cda
    zi_data = {'zi': zi, 'picos': picos, 'vales': vales,
               'limites_idx': limites_idx, 'col_zi': col_zi,
               'picos_raw': picos_raw, 'vales_raw': vales_raw}

# ── SHS ──
if run_shs:
    try:
        seg_shs = segmentacao_shs_mcv(df, col_est, col_defl, seg_min_m, seg_max_m, 'shs')
        seg_shs = agregar_segmentos(seg_shs)
        seg_shs = aplicar_clustering(seg_shs, ward_distance, hdb_min_cluster, hdb_min_samples)
        seg_results['SHS'] = seg_shs
    except Exception as e:
        st.error(f"Erro SHS: {e}")

# ── MCV ──
if run_mcv:
    try:
        seg_mcv = segmentacao_shs_mcv(df, col_est, col_defl, seg_min_m, seg_max_m, 'mcv')
        seg_mcv = agregar_segmentos(seg_mcv)
        seg_mcv = aplicar_clustering(seg_mcv, ward_distance, hdb_min_cluster, hdb_min_samples)
        seg_results['MCV'] = seg_mcv
    except Exception as e:
        st.error(f"Erro MCV: {e}")

if not seg_results:
    st.error("Nenhum metodo retornou resultados."); st.stop()


# ======================================================================
#  ABAS
# ======================================================================
tab_labels = list(seg_results.keys())
if run_cda and 'CDA' in seg_results:
    tab_labels.append("Serie Zi")
if len(seg_results) >= 2:
    tab_labels.append("Comparacao")
tab_labels += ["Dados", "Exportar"]

tabs = st.tabs(tab_labels)
tab_idx = 0

# ── Abas por metodo ──
for mkey, seg_df in seg_results.items():
    with tabs[tab_idx]:
        st.markdown(f"## {mkey} — Segmentacao + Clustering")
        pipeline_step("Pipeline Executado",
                      f"Segmentacao {mkey} → Agregacao → Ward + HDBSCAN")

        render_segmentation_charts(df, seg_df, col_est, col_defl, media_deflexao, mkey)
        st.divider()
        render_clustering_charts(seg_df, mkey)
    tab_idx += 1

# ── Serie Zi ──
if run_cda and 'CDA' in seg_results and zi_data:
    with tabs[tab_idx]:
        st.markdown("## Serie Zi — CDA")
        zi = zi_data['zi']
        picos = zi_data['picos']; vales = zi_data['vales']
        limites_idx = zi_data['limites_idx']
        seg_cda_df = seg_results['CDA']

        fig_zi = go.Figure()
        for i, row in seg_cda_df.iterrows():
            mask = (df[col_est] >= row['Inicio (m)']) & (df[col_est] <= row['Fim (m)'])
            c = SEG_COLORS_HEX[i % len(SEG_COLORS_HEX)]
            fig_zi.add_trace(go.Scatter(
                x=df.loc[mask, col_est], y=zi[mask],
                fill='tozeroy', mode='lines', line=dict(width=0),
                fillcolor=rgba_from_hex(c, 0.2),
                name=f"SH {row['SH']}", showlegend=False, hoverinfo='skip'))
        fig_zi.add_trace(go.Scatter(
            x=df[col_est], y=zi, mode='lines', name='Zi',
            line=dict(color='#2c3e50', width=2.5)))
        if len(picos) > 0:
            fig_zi.add_trace(go.Scatter(
                x=df.loc[picos, col_est].values, y=zi[picos],
                mode='markers', name=f'Picos ({len(picos)})',
                marker=dict(symbol='triangle-up', size=14, color=COLOR_RED,
                            line=dict(color='white', width=2))))
        if len(vales) > 0:
            fig_zi.add_trace(go.Scatter(
                x=df.loc[vales, col_est].values, y=zi[vales],
                mode='markers', name=f'Vales ({len(vales)})',
                marker=dict(symbol='triangle-down', size=14, color=COLOR_GREEN,
                            line=dict(color='white', width=2))))
        for idx in limites_idx[1:-1]:
            fig_zi.add_vline(x=float(df.loc[idx, col_est]), line_dash='dot',
                             line_color='gray', line_width=0.8, opacity=0.5)
        fig_zi.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1, opacity=0.4)
        fig_zi.update_layout(xaxis_title='Estacao (m)', yaxis_title='Zi',
                             template='plotly_white', height=500,
                             legend=dict(orientation='h', y=-0.12),
                             hovermode='x unified', margin=dict(t=30, b=40))
        st.plotly_chart(fig_zi, use_container_width=True)

        # Mapa colorido
        st.markdown("#### Mapa Segmentos — Zi Colorida")
        fig_map = go.Figure()
        for i, row in seg_cda_df.iterrows():
            mask = (df[col_est] >= row['Inicio (m)']) & (df[col_est] <= row['Fim (m)'])
            c = SEG_COLORS_HEX[i % len(SEG_COLORS_HEX)]
            fig_map.add_trace(go.Scatter(
                x=df.loc[mask, col_est], y=zi[mask],
                fill='tozeroy', mode='lines', line=dict(width=0.5, color=c),
                fillcolor=rgba_from_hex(c, 0.4),
                name=f"SH {row['SH']} ({row['Comprimento (m)']:,.0f}m)"))
        fig_map.add_trace(go.Scatter(
            x=df[col_est], y=zi, mode='lines', name='Zi',
            line=dict(color='#2c3e50', width=2), showlegend=False))
        fig_map.update_layout(template='plotly_white', height=420,
                              xaxis_title='Estacao (m)', yaxis_title='Zi',
                              legend=dict(orientation='h', y=-0.18, font_size=9),
                              margin=dict(t=20, b=60))
        st.plotly_chart(fig_map, use_container_width=True)
    tab_idx += 1

# ── Comparacao ──
if len(seg_results) >= 2:
    with tabs[tab_idx]:
        st.markdown("## Comparacao entre Metodos")
        story("Comparacao lado a lado: segmentacao + clustering.")

        # KPIs
        comp_cols = st.columns(len(seg_results))
        for col_ui, (mkey, mdf) in zip(comp_cols, seg_results.items()):
            with col_ui:
                st.markdown(f"##### {mkey}")
                st.markdown(kpi_card(f"{len(mdf)}", "Segmentos",
                                     'kpi-blue' if mkey == 'CDA' else 'kpi-purple' if mkey == 'SHS' else 'kpi-teal'),
                            unsafe_allow_html=True)
                n_out = int(mdf['is_outlier'].sum()) if 'is_outlier' in mdf.columns else 0
                st.markdown(kpi_card(f"{n_out}", "Outliers",
                                     'kpi-red' if n_out else 'kpi-green'),
                            unsafe_allow_html=True)

        st.divider()

        # Resumo
        resumo = []
        for mkey, mdf in seg_results.items():
            n_out = int(mdf['is_outlier'].sum()) if 'is_outlier' in mdf.columns else 0
            n_w = mdf['cluster_ward'].nunique() if 'cluster_ward' in mdf.columns else '-'
            resumo.append({
                'Metodo': mkey, 'Segmentos': len(mdf),
                'CV Medio (%)': f"{mdf['CV (%)'].mean():.1f}",
                'Clusters Ward': n_w,
                'Outliers HDBSCAN': n_out,
                'Homogeneos (CV<=25%)': f"{(mdf['CV (%)'] <= 25).sum()}/{len(mdf)}",
            })
        st.dataframe(pd.DataFrame(resumo), hide_index=True)

        # Grafico comparativo
        st.markdown("##### Deflexao Media — Comparacao")
        method_plotcolors = {'CDA': COLOR_PRIMARY, 'SHS': COLOR_SEC, 'MCV': COLOR_TEAL}
        fig_comp = go.Figure()
        for mkey, mdf in seg_results.items():
            fig_comp.add_trace(go.Bar(
                x=mdf['SH'].apply(lambda x: f'SH {x}'), y=mdf['Defl. Media'],
                name=mkey, marker_color=method_plotcolors.get(mkey, COLOR_PRIMARY),
                opacity=0.8))
        fig_comp.update_layout(barmode='group', template='plotly_white', height=380,
                               xaxis_title='Segmento', yaxis_title='Defl. Media',
                               legend=dict(orientation='h', y=-0.15),
                               margin=dict(t=20, b=50))
        st.plotly_chart(fig_comp, use_container_width=True)
    tab_idx += 1

# ── Dados ──
with tabs[tab_idx]:
    st.markdown("#### Dados Brutos Processados")
    cols_show = [c for c in df.columns if c not in ['slk_from', 'slk_to']]
    st.caption(f"{len(df)} registros")
    st.dataframe(df[cols_show], height=450)
tab_idx += 1

# ── Exportar ──
with tabs[tab_idx]:
    st.markdown("#### Exportar Resultados do Pipeline")
    story("Baixe os resultados com <b>segmentos + clusters + outliers</b>.")

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("""<div style="background:#f0f2f6;padding:20px;border-radius:12px;text-align:center;">
            <div style="font-size:3rem;">&#128215;</div>
            <div style="font-weight:bold;margin:8px 0;">Excel Pipeline</div>
            <div style="font-size:0.85rem;color:#666;">
                Abas por metodo + Dados Brutos<br>Inclui clusters e outliers
            </div></div>""", unsafe_allow_html=True)

        excel_bytes = gerar_excel_pipeline(df, seg_results, col_est, col_defl, zi_data)
        st.download_button("Baixar Excel", data=excel_bytes,
                           file_name=f"pipeline_segmentacao_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with col_e2:
        st.markdown("""<div style="background:#f0f2f6;padding:20px;border-radius:12px;text-align:center;">
            <div style="font-size:3rem;">&#128196;</div>
            <div style="font-weight:bold;margin:8px 0;">CSV Pipeline</div>
            <div style="font-size:0.85rem;color:#666;">
                Features enriquecidos (clusters + outliers)<br>Separador: ponto e virgula
            </div></div>""", unsafe_allow_html=True)
        csv_parts = []
        for mkey, mdf in seg_results.items():
            csv_parts.append(f"# METODO: {mkey}\n" + mdf.to_csv(index=False, sep=';'))
        csv_data = "\n\n".join(csv_parts).encode('utf-8-sig')
        st.download_button("Baixar CSV", data=csv_data,
                           file_name=f"pipeline_segmentacao_{datetime.now():%Y%m%d_%H%M%S}.csv",
                           mime="text/csv")

    st.divider()
    st.markdown("#### Parametros do Pipeline")
    params = [
        {'Parametro': 'Metodo(s)', 'Valor': ', '.join(seg_results.keys())},
        {'Parametro': 'seg_min_m', 'Valor': f'{seg_min_m} m'},
        {'Parametro': 'seg_max_m', 'Valor': f'{seg_max_m} m'},
        {'Parametro': 'Ward distance_threshold', 'Valor': str(ward_distance)},
        {'Parametro': 'HDBSCAN min_cluster_size', 'Valor': str(hdb_min_cluster)},
        {'Parametro': 'HDBSCAN min_samples', 'Valor': str(hdb_min_samples)},
        {'Parametro': 'Media deflexao', 'Valor': f'{media_deflexao:.2f}'},
        {'Parametro': 'Registros', 'Valor': str(len(df))},
    ]
    for mkey in seg_results:
        sdf = seg_results[mkey]
        params.append({'Parametro': f'Segmentos {mkey}', 'Valor': str(len(sdf))})
        if 'cluster_ward' in sdf.columns:
            params.append({'Parametro': f'Clusters Ward ({mkey})',
                           'Valor': str(sdf['cluster_ward'].nunique())})
        if 'is_outlier' in sdf.columns:
            params.append({'Parametro': f'Outliers HDBSCAN ({mkey})',
                           'Valor': str(int(sdf['is_outlier'].sum()))})
    st.dataframe(pd.DataFrame(params), hide_index=True)

    # Placeholder futuro
    st.divider()
    st.markdown("#### Modelo Preditivo (Futuro — Fase 2)")
    pipeline_step("Proximos Passos",
                  "Os features enriquecidos (cluster_ward, cluster_hdbscan, is_outlier) "
                  "servem como entrada para modelos de ML (XGBoost, Random Forest). "
                  "Alvos possiveis: IRI futuro, necessidade de intervencao, custo de manutencao.")
    st.markdown("""
    ```python
    # Exemplo de uso futuro:
    from xgboost import XGBRegressor
    
    feature_cols = ['Defl. Media', 'Desvio Padrao', 'Defl. Max',
                    'Comprimento (m)', 'CV (%)', 'cluster_ward']
    X = seg_df[feature_cols]
    y = seg_df['IRI_futuro']  # alvo a ser definido
    
    model = XGBRegressor(n_estimators=100, max_depth=5)
    model.fit(X, y)
    ```
    """)


# ── Footer ──
st.divider()
st.markdown("""
<div style="text-align:center;color:#999;font-size:0.8rem;padding:10px;">
    Pipeline de Segmentacao Rodoviaria — CDA · SHS · MCV · Ward · HDBSCAN |
    Streamlit + Plotly + scikit-learn
</div>""", unsafe_allow_html=True)
