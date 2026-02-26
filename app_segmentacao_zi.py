# -*- coding: utf-8 -*-
"""
Segmentacao Homogenea de Rodovias - Metodo CUSUM (Zi)
Dashboard interativo em Streamlit com Plotly.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from io import BytesIO
from datetime import datetime

# ======================================================================
#  CONFIGURACAO DA PAGINA
# ======================================================================
st.set_page_config(
    page_title="Segmentacao Zi - CUSUM",
    page_icon="\U0001f6e3\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- CSS customizado --
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 14px; text-align: center;
        color: white; box-shadow: 0 4px 15px rgba(102,126,234,0.3);
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-3px); }
    .kpi-value { font-size: 2rem; font-weight: 800; line-height: 1.2; }
    .kpi-label { font-size: 0.78rem; opacity: 0.85; margin-top: 4px; }
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
    .story-box {
        background: #f8f9fa; border-left: 5px solid #667eea;
        padding: 15px 20px; border-radius: 0 10px 10px 0;
        margin: 12px 0; font-size: 0.92rem; color: #333;
    }
    [data-testid="stSidebar"] { background: #f0f2f6; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    .dataframe { font-size: 0.85rem !important; }
    hr { border-top: 2px solid #667eea30; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ======================================================================
#  FUNCOES AUXILIARES
# ======================================================================

def kpi_card(value, label, css_class=""):
    return f"""
    <div class="kpi-card {css_class}">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>"""


def story(text):
    st.markdown(f'<div class="story-box">{text}</div>', unsafe_allow_html=True)


def merge_short_segments(limites, df_local, col_est, min_m):
    limites = list(limites)
    while True:
        lengths = [
            df_local.loc[limites[i+1], col_est] - df_local.loc[limites[i], col_est]
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
            left_combo  = lengths[short_idx - 1] + lengths[short_idx]
            right_combo = lengths[short_idx] + lengths[short_idx + 1]
            if left_combo <= right_combo:
                limites.pop(short_idx)
            else:
                limites.pop(short_idx + 1)
    return np.array(limites)


def split_long_segments(limites, df_local, col_est, max_m):
    new_limites = [limites[0]]
    for i in range(len(limites) - 1):
        idx_ini = limites[i]; idx_fim = limites[i + 1]
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


# Paleta hex para segmentos
SEG_COLORS_HEX = [
    '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854',
    '#ffd92f', '#e5c494', '#b3b3b3', '#1b9e77', '#d95f02',
    '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d',
]

def rgba_from_hex(hex_color, alpha=0.3):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def gerar_excel(df_data, segmentos_df, picos_arr, vales_arr, limites, col_est, col_defl, col_d, col_z):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.chart import BarChart, Reference, LineChart
    from openpyxl.chart.marker import Marker
    from openpyxl.chart.label import DataLabelList
    from openpyxl.chart.shapes import GraphicalProperties
    from openpyxl.utils import get_column_letter
    from openpyxl.formatting.rule import DataBarRule

    COR_H = 'FF4A6FA5'; COR_H2 = 'FF667EEA'; COR_AC = 'FF764BA2'
    COR_LB = 'FFF2F4F8'; COR_W = 'FFFFFFFF'; COR_R = 'FFE74C3C'; COR_G = 'FF27AE60'

    ft = Font(name='Calibri', bold=True, size=16, color='FFFFFF')
    fh = Font(name='Calibri', bold=True, size=11, color='FFFFFF')
    fb = Font(name='Calibri', size=11)
    fH = PatternFill('solid', fgColor=COR_H)
    fH2 = PatternFill('solid', fgColor=COR_H2)
    fL = PatternFill('solid', fgColor=COR_LB)
    fW = PatternFill('solid', fgColor=COR_W)
    ac = Alignment(horizontal='center', vertical='center', wrap_text=True)
    bd = Border(*[Side(style='thin', color='D0D0D0')]*4)

    wb = Workbook()

    # -- Resumo --
    ws = wb.active; ws.title = 'Resumo'
    ws.merge_cells('A1:F2')
    ws['A1'].value = 'SEGMENTACAO HOMOGENEA - CUSUM (Zi)'; ws['A1'].font = ft; ws['A1'].fill = fH; ws['A1'].alignment = ac
    cols_r = ['SH', 'Inicio (m)', 'Fim (m)', 'Comprimento (m)', 'Zi_fim', 'Defl. Media']
    for j, cn in enumerate(cols_r, 1):
        c = ws.cell(row=4, column=j, value=cn); c.font = fh; c.fill = fH2; c.alignment = ac; c.border = bd
        ws.column_dimensions[get_column_letter(j)].width = 18
    for i, rd in segmentos_df.iterrows():
        for j, cn in enumerate(cols_r, 1):
            c = ws.cell(row=5+i, column=j, value=rd[cn]); c.font = fb; c.alignment = ac; c.border = bd
            c.fill = fL if i % 2 == 0 else fW
            if cn in ['Inicio (m)', 'Fim (m)', 'Comprimento (m)']: c.number_format = '#,##0'
            elif cn in ['Zi_fim', 'Defl. Media']: c.number_format = '0.0'
    lr = 5 + len(segmentos_df) - 1
    ws.conditional_formatting.add(f'D5:D{lr}', DataBarRule(start_type='min', end_type='max', color=COR_H2[2:]))
    ch = BarChart(); ch.type='col'; ch.style=10; ch.title='Deflexao Media por SH'
    ch.y_axis.title='Deflexao (0,01mm)'; ch.width=22; ch.height=12
    ch.add_data(Reference(ws,min_col=6,min_row=4,max_row=lr), titles_from_data=True)
    ch.set_categories(Reference(ws,min_col=1,min_row=5,max_row=lr))
    ch.series[0].graphicalProperties.solidFill=COR_H2[2:]
    ch.series[0].dLbls=DataLabelList(); ch.series[0].dLbls.showVal=True; ch.series[0].dLbls.numFmt='0.0'
    ch.legend=None; ws.add_chart(ch, f'A{lr+3}')

    # -- Serie Zi --
    ws2 = wb.create_sheet('Serie Zi')
    hdrs = ['Estacao (m)', 'Zi', 'Zi (Picos)', 'Zi (Vales)']
    for j, h in enumerate(hdrs, 1):
        c = ws2.cell(row=1, column=j, value=h); c.font=fh; c.fill=fH2; c.alignment=ac; c.border=bd
    p_set = set(picos_arr); v_set = set(vales_arr)
    for i in range(len(df_data)):
        ws2.cell(row=2+i, column=1, value=float(df_data.loc[i, col_est])).number_format='#,##0'
        ws2.cell(row=2+i, column=2, value=float(df_data.loc[i, col_z])).number_format='0.00'
        if i in p_set: ws2.cell(row=2+i, column=3, value=float(df_data.loc[i, col_z]))
        if i in v_set: ws2.cell(row=2+i, column=4, value=float(df_data.loc[i, col_z]))
    lz = 2+len(df_data)-1
    cz = LineChart(); cz.title='Serie Zi - Picos e Vales'; cz.width=32; cz.height=16; cz.style=10
    cz.y_axis.title='Zi'; cz.x_axis.title='Estacao (m)'
    cz.add_data(Reference(ws2,min_col=2,min_row=1,max_row=lz), titles_from_data=True)
    cz.add_data(Reference(ws2,min_col=3,min_row=1,max_row=lz), titles_from_data=True)
    cz.add_data(Reference(ws2,min_col=4,min_row=1,max_row=lz), titles_from_data=True)
    cz.set_categories(Reference(ws2,min_col=1,min_row=2,max_row=lz))
    cz.series[0].graphicalProperties.line.solidFill=COR_H[2:]; cz.series[0].graphicalProperties.line.width=22000; cz.series[0].marker=Marker(symbol='none')
    cz.series[1].graphicalProperties.line.noFill=True; cz.series[1].marker=Marker(symbol='triangle',size=10)
    cz.series[1].marker.graphicalProperties=GraphicalProperties(); cz.series[1].marker.graphicalProperties.solidFill=COR_R[2:]
    cz.series[2].graphicalProperties.line.noFill=True; cz.series[2].marker=Marker(symbol='triangle',size=10)
    cz.series[2].marker.graphicalProperties=GraphicalProperties(); cz.series[2].marker.graphicalProperties.solidFill=COR_G[2:]
    if len(df_data)>20: cz.x_axis.tickLblSkip=max(1,len(df_data)//15)
    ws2.add_chart(cz, 'F1')

    # -- Dados Brutos --
    ws3 = wb.create_sheet('Dados Brutos')
    for j, cn in enumerate(df_data.columns, 1):
        c = ws3.cell(row=1, column=j, value=cn); c.font=fh; c.fill=fH2; c.alignment=ac; c.border=bd
        ws3.column_dimensions[get_column_letter(j)].width = 20
    for i, rd in df_data.iterrows():
        for j, cn in enumerate(df_data.columns, 1):
            v = rd[cn]
            ws3.cell(row=2+i, column=j, value=float(v) if isinstance(v,(int,float,np.integer,np.floating)) else v)

    buf = BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()


# ======================================================================
#  SIDEBAR
# ======================================================================
with st.sidebar:
    st.markdown("## \U0001f6e3\ufe0f Segmentacao Zi")
    st.caption("Metodo CUSUM - Diferenca Acumulada")
    st.divider()

    uploaded_file = st.file_uploader(
        "**Carregar planilha**",
        type=["xlsx", "xls", "csv"],
        help="Arquivo com colunas Estacao (m) e Deflexao (0,01mm)"
    )

    st.divider()
    st.markdown("### Parametros")

    col_estacao_name  = st.text_input("Coluna **Estação**", value="Estação (m)")
    col_deflexao_name = st.text_input("Coluna **Deflexão**", value="Deflexão (0,01mm)")

    st.divider()

    st.markdown("#### `find_peaks` (scipy)")

    fp_distance = st.slider(
        "distance (pontos)",
        min_value=1, max_value=50, value=5, step=1,
        help="Distancia minima entre picos consecutivos (em pontos). Menor = mais candidatos."
    )
    fp_height = st.number_input(
        "height (Zi)", value=0.0, step=1.0, format="%.1f",
        help="Altura minima dos picos. 0 = desativado."
    )
    fp_prominence = st.number_input(
        "prominence (Zi)", value=0.0, min_value=0.0, step=1.0, format="%.1f",
        help="Proeminencia minima do pico em relacao aos vizinhos. 0 = desativado."
    )
    fp_width = st.number_input(
        "width (pontos)", value=0.0, min_value=0.0, step=1.0, format="%.1f",
        help="Largura minima do pico (em amostras). 0 = desativado."
    )
    fp_threshold = st.number_input(
        "threshold (Zi)", value=0.0, min_value=0.0, step=0.5, format="%.1f",
        help="Diferenca vertical minima entre o pico e seus vizinhos imediatos. 0 = desativado."
    )
    fp_plateau_size = st.number_input(
        "plateau_size (pontos)", value=0, min_value=0, step=1,
        help="Tamanho minimo do plato (flat top). 0 = desativado."
    )

    st.divider()
    st.markdown("#### Restricoes de Comprimento")

    seg_min_m = st.number_input(
        "Comprimento MINIMO de SH (m)", value=800, min_value=50, step=50,
        help="Segmentos menores serao fundidos com vizinhos."
    )
    seg_max_m = st.number_input(
        "Comprimento MAXIMO de SH (m)", value=5000, min_value=500, step=100,
        help="Segmentos maiores serao subdivididos."
    )

# ======================================================================
#  HERO HEADER
# ======================================================================
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px 40px; border-radius: 16px; margin-bottom: 24px;
            box-shadow: 0 8px 30px rgba(102,126,234,0.35);">
    <h1 style="color: white; margin: 0; font-size: 2rem;">
        Segmentacao Homogenea de Rodovias
    </h1>
    <p style="color: rgba(255,255,255,0.85); font-size: 1.05rem; margin-top: 8px;">
        Identificacao automatica de Segmentos Homogeneos via <b>CUSUM</b> -
        Cumulative Sum Control Chart (Diferenca Acumulada Zi)
    </p>
</div>
""", unsafe_allow_html=True)

# ======================================================================
#  PROCESSAMENTO
# ======================================================================
if uploaded_file is None:
    st.info("Carregue um arquivo Excel ou CSV na barra lateral para iniciar a analise.")
    st.markdown("""
    ### O que este app faz?
    1. **Le** uma planilha com colunas **Estacao** e **Deflexao**
    2. **Calcula** a Diferenca (Deflexao - Media) e a Diferenca Acumulada (Zi)
    3. **Detecta** picos e vales na curva Zi via `scipy.signal.find_peaks`
    4. **Aplica** restricoes de comprimento minimo e maximo nos segmentos
    5. **Gera** visualizacoes interativas e exporta planilha Excel formatada

    ---
    **Colunas de entrada necessarias:**
    | Coluna | Descricao |
    |--------|-----------|
    | Estacao (m) | Posicao quilometrica |
    | Deflexao (0,01mm) | Valor medido |
    """)
    st.stop()

# -- Leitura dos dados --
fname = uploaded_file.name
try:
    if fname.endswith('.csv'):
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Erro ao ler o arquivo: {e}")
    st.stop()

col_est  = col_estacao_name
col_defl = col_deflexao_name

if col_est not in df.columns or col_defl not in df.columns:
    st.error(f"Colunas esperadas nao encontradas. Disponiveis: {list(df.columns)}")
    st.stop()

for c in [col_est, col_defl]:
    if df[c].dtype == 'O':
        df[c] = df[c].astype(str).str.replace(',', '.').astype(float)
    df[c] = pd.to_numeric(df[c], errors='coerce')
df.dropna(subset=[col_est, col_defl], inplace=True)
df.reset_index(drop=True, inplace=True)

# -- Filtro de Estacao na sidebar --
with st.sidebar:
    st.divider()
    st.markdown("#### Filtro de Estacao")
    est_min_data = float(df[col_est].min())
    est_max_data = float(df[col_est].max())
    est_range = st.slider(
        "Intervalo de Estacao (m)",
        min_value=est_min_data,
        max_value=est_max_data,
        value=(est_min_data, est_max_data),
        step=10.0,
        format="%.0f",
        help="Selecione o trecho da rodovia a ser analisado."
    )
    est_filter_min, est_filter_max = est_range

df = df[(df[col_est] >= est_filter_min) & (df[col_est] <= est_filter_max)].copy()
df.reset_index(drop=True, inplace=True)

if len(df) < 3:
    st.error("Poucos registros apos o filtro de Estacao. Amplie o intervalo.")
    st.stop()

col_diff = 'Diferenca'
col_zi   = 'Diferenca Acumulada (Zi)'
media_deflexao = df[col_defl].mean()
df[col_diff] = df[col_defl] - media_deflexao
df[col_zi]   = df[col_diff].cumsum()

zi = df[col_zi].values

# -- Deteccao --
# Montar dict de kwargs para find_peaks (somente parametros > 0)
fp_kwargs = {'distance': max(1, fp_distance)}
if fp_height > 0:
    fp_kwargs['height'] = fp_height
if fp_prominence > 0:
    fp_kwargs['prominence'] = fp_prominence
if fp_width > 0:
    fp_kwargs['width'] = fp_width
if fp_threshold > 0:
    fp_kwargs['threshold'] = fp_threshold
if fp_plateau_size > 0:
    fp_kwargs['plateau_size'] = fp_plateau_size

picos_raw, _ = find_peaks(zi, **fp_kwargs)
vales_raw, _ = find_peaks(-zi, **fp_kwargs)
limites_ini = np.sort(np.concatenate(([0], picos_raw, vales_raw, [len(df)-1])))

limites_merged = merge_short_segments(limites_ini, df, col_est, seg_min_m)
limites_idx    = split_long_segments(limites_merged, df, col_est, seg_max_m)

picos = np.array(sorted(set(picos_raw) & set(limites_idx)))
vales = np.array(sorted(set(vales_raw) & set(limites_idx)))

# -- Tabela de segmentos --
segmentos = []
for i in range(len(limites_idx) - 1):
    ii, fi = limites_idx[i], limites_idx[i+1]
    est_i = float(df.loc[ii, col_est]); est_f = float(df.loc[fi, col_est])
    defl_seg = df.loc[ii:fi, col_defl]
    defl_media = defl_seg.mean()
    defl_std   = defl_seg.std(ddof=1) if len(defl_seg) > 1 else 0.0
    segmentos.append({
        'SH': i+1,
        'Inicio (m)': est_i,
        'Fim (m)': est_f,
        'Comprimento (m)': est_f - est_i,
        'Zi_fim': round(float(df.loc[fi, col_zi]), 1),
        'Defl. Media': round(defl_media, 1),
        'Desvio Padrao': round(defl_std, 1),
        'Defl. Caract.': round(defl_media + defl_std, 1),
    })
seg_df = pd.DataFrame(segmentos)

# ======================================================================
#  KPIs
# ======================================================================
st.markdown("### Indicadores-Chave")

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: st.markdown(kpi_card(f"{len(seg_df)}", "Segmentos"), unsafe_allow_html=True)
with c2: st.markdown(kpi_card(f"{seg_df['Comprimento (m)'].sum():,.0f} m", "Extensao Total", "kpi-green"), unsafe_allow_html=True)
with c3: st.markdown(kpi_card(f"{seg_df['Comprimento (m)'].mean():,.0f} m", "Compr. Medio", "kpi-blue"), unsafe_allow_html=True)
with c4: st.markdown(kpi_card(f"{len(picos)}", "Picos", "kpi-red"), unsafe_allow_html=True)
with c5: st.markdown(kpi_card(f"{len(vales)}", "Vales", "kpi-green"), unsafe_allow_html=True)
with c6: st.markdown(kpi_card(f"{media_deflexao:.1f}", "Defl. Media Global", "kpi-orange"), unsafe_allow_html=True)

st.markdown("")

story(
    f"Foram identificados <b>{len(seg_df)} segmentos homogeneos</b> ao longo de "
    f"<b>{seg_df['Comprimento (m)'].sum():,.0f} m</b> de rodovia. "
    f"A deflexao media global e <b>{media_deflexao:.2f}</b> (0,01mm). "
    f"O metodo CUSUM detectou <b>{len(picos_raw)} picos</b> e <b>{len(vales_raw)} vales</b> brutos; "
    f"apos fusao (min {seg_min_m}m) e subdivisao (max {seg_max_m}m), restaram "
    f"<b>{len(limites_idx)} pontos de limite</b>."
)

st.divider()

COLOR_PRIMARY = '#667eea'
COLOR_SEC     = '#764ba2'
COLOR_RED     = '#e74c3c'
COLOR_GREEN   = '#27ae60'
COLOR_YELLOW  = '#f39c12'

# ======================================================================
#  ABAS PRINCIPAIS
# ======================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Serie Zi", "Segmentos", "Analise", "Dados", "Exportar"
])

# -- TAB 1: Serie Zi --
with tab1:
    st.markdown("#### Deflexao ao Longo da Rodovia")
    story("O grafico abaixo mostra os valores de deflexao medidos. "
          "A <span style='color:#e74c3c;font-weight:bold;'>linha tracejada vermelha</span> "
          "indica a media global - pontos acima representam trechos com deflexao superior.")

    fig_defl = go.Figure()
    fig_defl.add_trace(go.Bar(
        x=df[col_est], y=df[col_defl], name='Deflexao',
        marker_color=COLOR_PRIMARY, opacity=0.6,
    ))
    fig_defl.add_hline(y=media_deflexao, line_dash='dash', line_color=COLOR_RED, line_width=2,
                       annotation_text=f'Media = {media_deflexao:.2f}',
                       annotation_position='top right')
    fig_defl.update_layout(
        xaxis_title='Estacao (m)', yaxis_title='Deflexao (0,01mm)',
        template='plotly_white', height=380, margin=dict(t=30, b=40),
        legend=dict(orientation='h', y=-0.15)
    )
    st.plotly_chart(fig_defl)

    st.markdown("#### Serie de Diferencas Acumuladas (Zi)")
    story("A curva Zi e a <b>soma cumulativa</b> das diferencas (Deflexao - Media). "
          "Os <span style='color:#e74c3c;font-weight:bold;'>picos</span> e "
          "<span style='color:#27ae60;font-weight:bold;'>vales</span> marcam as mudancas de comportamento - "
          "cada transicao pico/vale define um <b>Segmento Homogeneo</b>.")

    fig_zi = go.Figure()

    for i, row in seg_df.iterrows():
        mask = (df[col_est] >= row['Inicio (m)']) & (df[col_est] <= row['Fim (m)'])
        hex_c = SEG_COLORS_HEX[i % len(SEG_COLORS_HEX)]
        fig_zi.add_trace(go.Scatter(
            x=df.loc[mask, col_est], y=zi[mask],
            fill='tozeroy', mode='lines', line=dict(width=0),
            fillcolor=rgba_from_hex(hex_c, 0.2),
            name=f"SH {row['SH']}",
            showlegend=False, hoverinfo='skip'
        ))

    fig_zi.add_trace(go.Scatter(
        x=df[col_est], y=zi, mode='lines', name='Zi',
        line=dict(color='#2c3e50', width=2.5),
    ))

    if len(picos) > 0:
        fig_zi.add_trace(go.Scatter(
            x=df.loc[picos, col_est].values, y=zi[picos],
            mode='markers+text', name=f'Picos ({len(picos)})',
            marker=dict(symbol='triangle-up', size=14, color=COLOR_RED, line=dict(color='white', width=2)),
            text=[f'{v:.0f}' for v in zi[picos]],
            textposition='top center', textfont=dict(size=9, color=COLOR_RED),
        ))

    if len(vales) > 0:
        fig_zi.add_trace(go.Scatter(
            x=df.loc[vales, col_est].values, y=zi[vales],
            mode='markers+text', name=f'Vales ({len(vales)})',
            marker=dict(symbol='triangle-down', size=14, color=COLOR_GREEN, line=dict(color='white', width=2)),
            text=[f'{v:.0f}' for v in zi[vales]],
            textposition='bottom center', textfont=dict(size=9, color=COLOR_GREEN),
        ))

    for idx in limites_idx[1:-1]:
        fig_zi.add_vline(x=float(df.loc[idx, col_est]), line_dash='dot',
                         line_color='gray', line_width=0.8, opacity=0.5)

    fig_zi.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1, opacity=0.5)
    fig_zi.update_layout(
        xaxis_title='Estacao (m)', yaxis_title='Diferenca Acumulada (Zi)',
        template='plotly_white', height=500, margin=dict(t=30, b=40),
        legend=dict(orientation='h', y=-0.12),
        hovermode='x unified',
    )
    st.plotly_chart(fig_zi)


# -- TAB 2: Segmentos --
with tab2:
    st.markdown("#### Mapa de Segmentos Homogeneos")
    story(
        f"Os <b>{len(seg_df)} segmentos</b> estao representados com cores distintas. "
        f"O segmento mais longo tem <b>{seg_df['Comprimento (m)'].max():,.0f} m</b> "
        f"e o mais curto <b>{seg_df['Comprimento (m)'].min():,.0f} m</b>."
    )

    # --- Plot 1: Serie Zi com segmentos coloridos ---
    st.markdown("##### Serie Zi - Segmentos Coloridos")
    fig_zi_seg = go.Figure()
    for i, row in seg_df.iterrows():
        mask = (df[col_est] >= row['Inicio (m)']) & (df[col_est] <= row['Fim (m)'])
        hex_c = SEG_COLORS_HEX[i % len(SEG_COLORS_HEX)]
        fig_zi_seg.add_trace(go.Scatter(
            x=df.loc[mask, col_est], y=zi[mask],
            fill='tozeroy', mode='lines', line=dict(width=0.5, color=hex_c),
            fillcolor=rgba_from_hex(hex_c, 0.45),
            name=f"SH {row['SH']} ({row['Comprimento (m)']:,.0f}m)",
            legendgroup=f"sh{row['SH']}", showlegend=True,
        ))
    fig_zi_seg.add_trace(go.Scatter(
        x=df[col_est], y=zi, mode='lines', name='Zi',
        line=dict(color='#2c3e50', width=2), showlegend=False,
    ))
    fig_zi_seg.update_layout(
        template='plotly_white', height=450,
        margin=dict(t=20, b=60),
        xaxis_title='Estacao (m)', yaxis_title='Zi',
        legend=dict(orientation='h', y=-0.18, font_size=10,
                    xanchor='center', x=0.5),
        hovermode='x unified',
    )
    st.plotly_chart(fig_zi_seg)

    st.markdown("")  # espacamento entre os graficos

    # --- Plot 2: Deflexao media por segmento ---
    st.markdown("##### Deflexao Media por Segmento")
    fig_bar_seg = go.Figure()
    bar_colors = [COLOR_RED if d > seg_df['Defl. Media'].mean() * 1.2
                  else COLOR_YELLOW if d > seg_df['Defl. Media'].mean()
                  else COLOR_GREEN
                  for d in seg_df['Defl. Media']]
    fig_bar_seg.add_trace(go.Bar(
        x=seg_df['SH'].apply(lambda x: f'SH {x}'), y=seg_df['Defl. Media'],
        marker_color=bar_colors, name='Defl. Media', showlegend=False,
        text=seg_df['Defl. Media'].apply(lambda x: f'{x:.1f}'),
        textposition='outside',
    ))
    fig_bar_seg.add_hline(y=seg_df['Defl. Media'].mean(),
                          line_dash='dash', line_color=COLOR_PRIMARY, line_width=2,
                          annotation_text=f"Media = {seg_df['Defl. Media'].mean():.1f}")
    fig_bar_seg.update_layout(
        template='plotly_white', height=380,
        margin=dict(t=20, b=40),
        xaxis_title='Segmento', yaxis_title='Defl. Media (0,01mm)',
    )
    st.plotly_chart(fig_bar_seg)


# -- TAB 3: Analise --
with tab3:
    st.markdown("#### Analise Detalhada dos Segmentos")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("##### Distribuicao de Comprimentos")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=seg_df['Comprimento (m)'], nbinsx=max(5, len(seg_df)//2),
            marker_color=COLOR_PRIMARY, opacity=0.75, name='Comprimento',
        ))
        fig_hist.add_vline(x=seg_df['Comprimento (m)'].mean(), line_dash='dash',
                           line_color=COLOR_RED, annotation_text=f"Media = {seg_df['Comprimento (m)'].mean():,.0f}m")
        fig_hist.add_vline(x=seg_min_m, line_dash='dot', line_color=COLOR_YELLOW,
                           annotation_text=f"Min = {seg_min_m}m")
        fig_hist.add_vline(x=seg_max_m, line_dash='dot', line_color=COLOR_YELLOW,
                           annotation_text=f"Max = {seg_max_m}m")
        fig_hist.update_layout(template='plotly_white', height=350,
                               xaxis_title='Comprimento (m)', yaxis_title='Frequencia',
                               margin=dict(t=30))
        st.plotly_chart(fig_hist)

    with col_b:
        st.markdown("##### Comprimento por Segmento")
        fig_bar = go.Figure()
        bar_c = [COLOR_GREEN if seg_min_m <= x <= seg_max_m else COLOR_RED
                 for x in seg_df['Comprimento (m)']]
        fig_bar.add_trace(go.Bar(
            x=seg_df['SH'], y=seg_df['Comprimento (m)'],
            marker_color=bar_c, name='Comprimento',
            text=seg_df['Comprimento (m)'].apply(lambda x: f'{x:,.0f}'),
            textposition='outside',
        ))
        fig_bar.add_hline(y=seg_min_m, line_dash='dot', line_color=COLOR_YELLOW, line_width=2,
                          annotation_text=f"Min = {seg_min_m}m")
        fig_bar.add_hline(y=seg_max_m, line_dash='dot', line_color=COLOR_RED, line_width=2,
                          annotation_text=f"Max = {seg_max_m}m")
        fig_bar.update_layout(template='plotly_white', height=350,
                              xaxis_title='Segmento (SH)', yaxis_title='Comprimento (m)',
                              margin=dict(t=30))
        st.plotly_chart(fig_bar)

    # Boxplot
    st.markdown("##### Variabilidade da Deflexao por Segmento")
    story("O boxplot revela a <b>dispersao interna</b> de cada segmento. "
          "Segmentos com alta variabilidade podem indicar heterogeneidade remanescente.")

    box_data = []
    for _, row in seg_df.iterrows():
        mask = (df[col_est] >= row['Inicio (m)']) & (df[col_est] <= row['Fim (m)'])
        for v in df.loc[mask, col_defl].values:
            box_data.append({'Segmento': f"SH {int(row['SH'])}", 'Deflexao': v})
    df_box = pd.DataFrame(box_data)

    fig_box = px.box(df_box, x='Segmento', y='Deflexao', color='Segmento',
                     color_discrete_sequence=px.colors.qualitative.Set2)
    fig_box.add_hline(y=media_deflexao, line_dash='dash', line_color=COLOR_RED,
                      annotation_text=f'Media global = {media_deflexao:.1f}')
    fig_box.update_layout(template='plotly_white', height=400,
                          yaxis_title='Deflexao (0,01mm)', showlegend=False,
                          margin=dict(t=30))
    st.plotly_chart(fig_box)

    # Resumo numerico
    st.markdown("##### Resumo Estatistico dos Segmentos")
    desc = seg_df[['Comprimento (m)', 'Defl. Media', 'Zi_fim']].describe().round(2)
    st.dataframe(desc)


# -- TAB 4: Dados --
with tab4:
    st.markdown("#### Tabela de Segmentos Homogeneos")

    n_curtos = (seg_df['Comprimento (m)'] < seg_min_m).sum()
    n_longos = (seg_df['Comprimento (m)'] > seg_max_m).sum()
    if n_curtos == 0 and n_longos == 0:
        st.success(f"Todos os {len(seg_df)} segmentos atendem as restricoes "
                   f"({seg_min_m}m <= SH <= {seg_max_m}m)")
    else:
        if n_curtos: st.warning(f"{n_curtos} segmento(s) abaixo de {seg_min_m}m")
        if n_longos: st.warning(f"{n_longos} segmento(s) acima de {seg_max_m}m")

    st.dataframe(
        seg_df.style
            .background_gradient(subset=['Comprimento (m)'], cmap='Blues')
            .background_gradient(subset=['Defl. Media'], cmap='RdYlGn_r')
            .background_gradient(subset=['Defl. Caract.'], cmap='OrRd')
            .format({'Inicio (m)': '{:,.0f}', 'Fim (m)': '{:,.0f}',
                     'Comprimento (m)': '{:,.0f}', 'Zi_fim': '{:.1f}',
                     'Defl. Media': '{:.1f}', 'Desvio Padrao': '{:.1f}',
                     'Defl. Caract.': '{:.1f}'}),
        height=400
    )

    st.divider()
    st.markdown("#### Dados Brutos Processados")
    st.caption(f"{len(df)} registros - Colunas calculadas: Diferenca, Zi")
    st.dataframe(df, height=400)


# -- TAB 5: Exportar --
with tab5:
    st.markdown("#### Exportar Resultados")
    story("Baixe os resultados em <b>Excel</b> (com graficos e formatacao) ou <b>CSV</b> (tabela simples).")

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("""
        <div style="background:#f0f2f6; padding:20px; border-radius:12px; text-align:center;">
            <div style="font-size:3rem;">&#128215;</div>
            <div style="font-weight:bold; margin:8px 0;">Planilha Excel</div>
            <div style="font-size:0.85rem; color:#666;">
                3 abas: Resumo, Serie Zi, Dados Brutos<br>
                Graficos com picos/vales marcados
            </div>
        </div>
        """, unsafe_allow_html=True)
        excel_bytes = gerar_excel(df, seg_df, picos, vales, limites_idx,
                                  col_est, col_defl, col_diff, col_zi)
        st.download_button(
            "Baixar Excel",
            data=excel_bytes,
            file_name=f"segmentos_Zi_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with col_e2:
        st.markdown("""
        <div style="background:#f0f2f6; padding:20px; border-radius:12px; text-align:center;">
            <div style="font-size:3rem;">&#128196;</div>
            <div style="font-weight:bold; margin:8px 0;">CSV (Segmentos)</div>
            <div style="font-size:0.85rem; color:#666;">
                Tabela de segmentos homogeneos<br>
                Separador: ponto e virgula (;)
            </div>
        </div>
        """, unsafe_allow_html=True)
        csv_data = seg_df.to_csv(index=False, sep=';').encode('utf-8-sig')
        st.download_button(
            "Baixar CSV",
            data=csv_data,
            file_name=f"segmentos_Zi_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv",
        )

    st.divider()
    st.markdown("#### Parametros Utilizados")
    fp_desc = [f'distance={fp_kwargs["distance"]}']
    for k in ['height','prominence','width','threshold','plateau_size']:
        if k in fp_kwargs:
            fp_desc.append(f'{k}={fp_kwargs[k]}')
    fp_str = ', '.join(fp_desc)

    params_df = pd.DataFrame({
        'Parametro': ['find_peaks kwargs', 'seg_min_m', 'seg_max_m', 'Media deflexao',
                      'Picos (brutos)', 'Vales (brutos)', 'Limites finais', 'Segmentos'],
        'Valor': [fp_str, f'{seg_min_m} m', f'{seg_max_m} m', f'{media_deflexao:.2f}',
                  len(picos_raw), len(vales_raw), len(limites_idx), len(seg_df)]
    })
    st.dataframe(params_df, hide_index=True)

# ======================================================================
#  FOOTER
# ======================================================================
st.divider()
st.markdown("""
<div style="text-align: center; color: #999; font-size: 0.8rem; padding: 10px;">
    Segmentacao Homogenea Zi - Metodo CUSUM | Streamlit + Plotly
</div>
""", unsafe_allow_html=True)


