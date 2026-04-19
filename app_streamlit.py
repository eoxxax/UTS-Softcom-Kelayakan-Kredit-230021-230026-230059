import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle, warnings, os
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Credit Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=IBM+Plex+Mono:wght@300;400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

:root {
    --bg:       #1a1410;
    --surface:  #221c16;
    --surface2: #2a221a;
    --border:   #3a2e22;
    --border2:  #4a3c2e;
    --br-dim:   #5c4a38;
    --br-mid:   #8a6e52;
    --br-base:  #b8926a;
    --br-light: #d4b48a;
    --br-pale:  #e8d4b8;
    --text:     #d4c4ae;
    --muted:    #6a5642;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background: var(--bg) !important;
    color: var(--text);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

/* HERO */
.hero {
    background: linear-gradient(150deg, var(--surface) 0%, #1e1710 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 40px 44px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    top: -80px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, #b8926a16 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-eye {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--br-mid);
    margin-bottom: 10px;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--br-pale);
    line-height: 1.05;
    margin: 0 0 8px;
}
.hero-title span { color: var(--br-base); }
.hero-desc { font-size: 0.82rem; color: var(--muted); font-weight: 300; }
.hero-tags { margin-top: 16px; display: flex; gap: 8px; flex-wrap: wrap; }
.hero-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border: 1px solid var(--border2);
    border-radius: 4px;
    padding: 3px 10px;
    color: var(--br-dim);
}

/* SECTION LABEL */
.slabel {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 20px 0 10px;
}

/* RESULT CARDS */
.rcard {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 26px 16px;
    text-align: center;
}
.rcard-method {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--br-mid);
    margin-bottom: 8px;
}
.rcard-score {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.2rem;
    font-weight: 700;
    line-height: 1;
    color: var(--br-light);
}
.rcard-unit {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin: 4px 0 12px;
}
.rcard-verdict {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 4px;
    display: inline-block;
    margin-top: 6px;
}

/* GAUGE */
.gauge { margin: 10px 0; }
.gauge-meta {
    display: flex;
    justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: var(--muted);
    margin-bottom: 5px;
}
.gauge-track { background: var(--border); border-radius: 3px; height: 5px; overflow: hidden; }
.gauge-fill  { height: 100%; border-radius: 3px; }

/* CONSENSUS */
.consensus {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: 14px;
    padding: 28px;
    text-align: center;
    margin-top: 24px;
}
.consensus-hdr {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
}
.consensus-verdict {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--br-light);
}
.consensus-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    color: var(--muted);
    margin-top: 6px;
}

/* MROW */
.mrow {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 9px 13px;
    margin: 4px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: var(--text);
}
.sdot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    margin-right: 7px;
    vertical-align: middle;
}

/* BEST PILL */
.best-pill {
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    margin-top: 16px;
}

/* ZONE BAR */
.zone-bar {
    display: flex;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.56rem;
    letter-spacing: 0.04em;
    margin-top: 6px;
    color: var(--muted);
}

.stButton > button:hover {
    background: var(--border) !important;
    color: var(--br-light) !important;
    border-color: var(--br-dim) !important;
}

/* BUTTON — predict*/
button[kind="primary"] {
    background: linear-gradient(135deg, #b8926a, #8a6e52) !important;
    border: none !important;
    padding: 0.65rem 1rem !important;
    transition: all 0.3s ease !important;
}

button[kind="primary"] p {
    color: #1a1410 !important;
    font-weight: 600 !important;
    margin: 0 !important;
}

button[kind="primary"]:hover {
    background: #d4c4ae !important; /* Pakai abu muda/krem dari palet warna kamu biar senada */
    border: none !important;
}

button[kind="primary"]:hover p {
    color: #1a1410 !important;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 4px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    color: var(--muted) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.2rem !important;
    margin-bottom: -1px !important;
}
.stTabs [aria-selected="true"] {
    color: var(--br-light) !important;
    border-bottom-color: var(--br-base) !important;
}

/* MISC */
hr { border-color: var(--border) !important; margin: 16px 0 !important; }
.stNumberInput input, .stTextInput input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 8px !important; }
.stAlert { background: var(--surface2) !important; border-color: var(--border2) !important; }
</style>
""", unsafe_allow_html=True)

# CONSTANTS
BG          = '#1a1410'
CARD        = '#221c16'
BR_SHADES   = ['#b8926a', '#8a6e52', '#6e5540']

def verdict_colors(s):
    if s >= 65: return '#d4b48a', '#2e2418', 'LAYAK'
    if s >= 40: return '#9a7c52', '#261f16', 'PERTIMBANGKAN'
    return             '#6e4e34', '#1e1612', 'TIDAK LAYAK'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD,
    'axes.edgecolor': '#3a2e22', 'axes.labelcolor': '#6a5642',
    'xtick.color': '#6a5642', 'ytick.color': '#6a5642',
    'text.color': '#d4c4ae', 'grid.color': '#2a221a',
    'font.family': 'monospace',
})

# FIS & ANN
def trimf(x, a, b, c):
    x=np.asarray(x,float); y=np.zeros_like(x)
    m=(x>=a)&(x<=b)&(b!=a); y[m]=(x[m]-a)/(b-a)
    m=(x>b)&(x<=c)&(c!=b);  y[m]=(c-x[m])/(c-b)
    y[x==b]=1.0; return y

def trapmf(x, a, b, c, d):
    x=np.asarray(x,float); y=np.zeros_like(x)
    m=(x>=a)&(x<b)&(b!=a); y[m]=(x[m]-a)/(b-a)
    # FIX: Pakai >= dan <= agar nilai tidak blank
    y[(x>=b)&(x<=c)]=1.0  
    m=(x>c)&(x<=d)&(d!=c); y[m]=(d-x[m])/(d-c)
    return y

def mf_eval(mf,x):
    t,*a=mf; return trimf(x,*a) if t=='tri' else trapmf(x,*a)

def fuzzify(v,D):
    return{l:float(mf_eval(m,np.array([v]))[0]) for l,m in D.items()}

def defuzz(out_mfs,strengths,lo=0,hi=100):
    x=np.linspace(lo,hi,600); agg=np.zeros_like(x)
    for l,s in strengths.items():
        if s>0: agg=np.maximum(agg,np.minimum(s,mf_eval(out_mfs[l],x)))
    return float(np.sum(x*agg)/np.sum(agg)) if agg.sum()>0 else 50.0

def run_fis(row, P, norm):
    pend  = np.clip((row['pendapatan_bulanan'] - norm['PEND_MIN']) / (norm['PEND_MAX'] - norm['PEND_MIN']) * 100, 0, 100)
    pinj  = np.clip((1 - (row['jumlah_pinjaman'] - norm['PINJ_MIN']) / (norm['PINJ_MAX'] - norm['PINJ_MIN'])) * 100, 0, 100)
    lama  = np.clip(row['lama_bekerja_tahun'] / norm['LAMA_MAX'] * 100, 0, 100)
    
    riw   = np.clip(row['riwayat_kredit_skor'], 0, 100) 
    
    rasio = np.clip((1 - row['rasio_hutang'] / norm['RASIO_MAX']) * 100, 0, 100)
    
    fp=fuzzify(pend, P['pendapatan']); fj=fuzzify(pinj, P['pinjaman'])
    fl=fuzzify(lama, P['lama_kerja']); fr=fuzzify(riw, P['riwayat'])
    fd=fuzzify(rasio, P['rasio_hutang'])
    
    st2={}
    for rule in P['rules']:
        p,j,l,r,d,o=rule
        s=min(fp.get(p,0),fj.get(j,0),fl.get(l,0),fr.get(r,0),fd.get(d,0))
        st2[o]=max(st2.get(o,0),s)
    return defuzz(P['output'], st2)

def relu(x):    return np.maximum(0,x)
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-500,500)))

def predict_ann(w,row,norm):
    x=np.array([[
        np.clip((row['pendapatan_bulanan']-norm['PEND_MIN'])/(norm['PEND_MAX']-norm['PEND_MIN']),0,1),
        np.clip(1-(row['jumlah_pinjaman']-norm['PINJ_MIN'])/(norm['PINJ_MAX']-norm['PINJ_MIN']),0,1),
        np.clip(row['lama_bekerja_tahun']/norm['LAMA_MAX'],0,1),
        np.clip(row['riwayat_kredit_skor']/100,0,1),
        np.clip(1-row['rasio_hutang']/norm['RASIO_MAX'],0,1),
    ]])
    a1=relu(x  @w['W1']+w['b1'])
    a2=relu(a1 @w['W2']+w['b2'])
    return float(sigmoid(a2@w['W3']+w['b3']).squeeze())*100


# LOAD MODELS
DEFAULT_NORM={
    'PEND_MIN':2_000_000,'PEND_MAX':40_000_000,
    'PINJ_MIN':5_000_000,'PINJ_MAX':180_000_000,
    'LAMA_MAX':30.0,'RASIO_MAX':0.90,
}

@st.cache_resource
def load_models():
    out={}
    for key,fname in [('fis','fis_manual.pkl'),('ga','ga_params.pkl'),
                       ('ann','ann_model.pkl'),('metrics','metrics.pkl')]:
        if os.path.exists(fname):
            with open(fname,'rb') as f: out[key]=pickle.load(f)
        else: out[key]=None
    return out

models=load_models()


# SIDEBAR
with st.sidebar:
    st.markdown(
        '<p style="font-family:\'Cormorant Garamond\',serif;font-size:1.3rem;'
        'color:#d4b48a;font-weight:700;margin:0 0 2px">Kelayakan Kredit</p>'
        '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.56rem;'
        'color:#3a2e22;letter-spacing:0.2em;text-transform:uppercase;margin:0">Kelayakan Kredit</p>',
        unsafe_allow_html=True
    )
    st.divider()

    st.markdown(
        '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.6rem;'
        'letter-spacing:0.16em;text-transform:uppercase;color:#6a5642;margin-bottom:8px">Status Model</p>',
        unsafe_allow_html=True
    )
    for lbl,key in [("FIS Manual",'fis'),("GA-FIS",'ga'),("ANN",'ann')]:
        ok=models.get(key) is not None
        dot='#b8926a' if ok else '#3a2e22'
        st.markdown(
            f'<div class="mrow">'
            f'<span><span class="sdot" style="background:{dot}"></span>{lbl}</span>'
            f'<span style="font-size:0.65rem;color:{"#8a6e52" if ok else "#3a2e22"}">{"Siap" if ok else "—"}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    if models.get('metrics'):
        st.divider()
        mv=models['metrics']
        st.markdown(
            '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.6rem;'
            'letter-spacing:0.16em;text-transform:uppercase;color:#6a5642;margin-bottom:8px">Hasil Training</p>',
            unsafe_allow_html=True
        )
        for n,k in [("FIS",'fis'),("GA",'ga'),("ANN",'ann')]:
            st.markdown(
                f'<div class="mrow"><span>{n}</span>'
                f'<span style="color:#6a5642;font-size:0.65rem">'
                f'RMSE {mv[k]["rmse"]:.3f}&nbsp;&nbsp;R² {mv[k]["r2"]:.3f}</span></div>',
                unsafe_allow_html=True
            )
        st.caption(f"Train: {mv['n_train']} · Test: {mv['n_test']} data")


# HERO
st.markdown("""
<div class="hero">
  <div class="hero-eye">Sistem Analisis Kredit</div>
  <div class="hero-title">Kelayakan <span>Kredit</span></div>
  <div class="hero-desc">Penilaian kelayakan kredit menggunakan tiga pendekatan komputasional.</div>
  <div class="hero-tags">
    <div class="hero-tag">FIS Mamdani</div>
    <div class="hero-tag">GA-FIS</div>
    <div class="hero-tag">ANN Backpropagation</div>
  </div>
</div>
""", unsafe_allow_html=True)

any_model=any(models.get(k) is not None for k in ['fis','ga','ann'])
if not any_model:
    st.warning("Model tidak ditemukan. Pastikan file .pkl berada di folder yang sama dengan app_streamlit.py.")
    st.stop()

tab1,tab2=st.tabs(["Prediksi Debitur","Performa Model"])

# TAB 1
with tab1:
    dv = dict(pend=8_000_000, pinj=50_000_000, lama=5.0, riw=65.0, rasio=0.30)

    st.markdown('<div style="margin-top:12px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="slabel">Data Calon Debitur</div>', unsafe_allow_html=True)

    c1,c2=st.columns(2, gap="large")
    with c1:
        pend =st.number_input("Pendapatan Bulanan (Rp)",2_000_000,40_000_000,value=dv['pend'],step=500_000,format="%d")
        pinj =st.number_input("Jumlah Pinjaman (Rp)",5_000_000,180_000_000,value=dv['pinj'],step=5_000_000,format="%d")
        lama =st.slider("Lama Bekerja (tahun)",0.5,30.0,value=float(dv['lama']),step=0.5)
    with c2:
        riw  =st.slider("Riwayat Kredit (0-100)",5.0,100.0,value=float(dv['riw']),step=1.0)
        rasio=st.slider("Rasio Hutang",0.05,0.90,value=float(dv['rasio']),step=0.01)
        
        if rasio<0.35:   r_lbl,r_col="Rendah — Baik",'#b8926a'
        elif rasio<0.60: r_lbl,r_col="Sedang",'#8a6e52'
        else:            r_lbl,r_col="Tinggi — Berisiko",'#5c4a38'
        
        st.markdown(
            f'<div style="padding:8px 14px;border-radius:7px;background:#221c16;'
            f'border:1px solid {r_col}55;font-family:IBM Plex Mono,monospace;'
            f'font-size:0.72rem;color:{r_col};margin-top:4px">'
            f'Rasio: <b>{r_lbl}</b></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div style="padding:10px 14px;border-radius:7px;background:#1a1410;'
            f'border:1px solid #3a2e22;font-family:IBM Plex Mono,monospace;'
            f'font-size:0.66rem;color:#6a5642;margin-top:8px;line-height:1.7">'
            f'Pendapatan &nbsp;Rp {pend:,.0f}<br>'
            f'Pinjaman &nbsp;&nbsp;&nbsp;Rp {pinj:,.0f}<br>'
            f'Lama kerja &nbsp;{lama} thn &nbsp;·&nbsp; Kredit {riw}/100'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div style="margin-top:10px"></div>', unsafe_allow_html=True)
    do_predict = st.button("Analisis Kelayakan Kredit", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    if do_predict:
        row=pd.Series({
            'pendapatan_bulanan':pend,'jumlah_pinjaman':pinj,
            'lama_bekerja_tahun':lama,'riwayat_kredit_skor':riw,'rasio_hutang':rasio,
        })
        norm={k:models['ann'][k] for k in DEFAULT_NORM} if models.get('ann') else DEFAULT_NORM

        preds={}
        if models.get('fis'): preds['FIS Manual']=run_fis(row,models['fis'],norm)
        if models.get('ga'):  preds['GA-FIS']    =run_fis(row,models['ga'], norm)
        if models.get('ann'): preds['ANN']        =predict_ann(models['ann'],row,norm)

        st.markdown('<div class="slabel" style="margin-top:24px">Hasil Analisis</div>', unsafe_allow_html=True)

        cols=st.columns(len(preds), gap="medium")
        scores={}
        for col,(name,score) in zip(cols,preds.items()):
            tc,bg,lbl=verdict_colors(score)
            scores[name]=score
            col.markdown(
                f'<div class="rcard" style="border-color:{tc}30">'
                f'<div class="rcard-method">{name}</div>'
                f'<div class="rcard-score">{score:.1f}</div>'
                f'<div class="rcard-unit">skor / 100</div>'
                f'<div class="rcard-verdict" style="color:{tc};background:{bg}">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown('<div style="margin-top:20px"></div>', unsafe_allow_html=True)
        for name,score in preds.items():
            tc,_,lbl=verdict_colors(score)
            st.markdown(
                f'<div class="gauge">'
                f'<div class="gauge-meta"><span>{name}</span>'
                f'<span style="color:{tc}">{score:.1f} — {lbl}</span></div>'
                f'<div class="gauge-track"><div class="gauge-fill" '
                f'style="width:{score:.1f}%;background:{tc}"></div></div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown(
            '<div class="zone-bar" style="margin-top:8px">'
            '<div style="flex:40;border-top:1px solid #5c4a38;padding-top:4px">0 — Tidak Layak</div>'
            '<div style="flex:25;border-top:1px solid #8a6e52;padding-top:4px;text-align:center">40 — Pertimbangkan</div>'
            '<div style="flex:35;border-top:1px solid #b8926a;padding-top:4px;text-align:right">65 — Layak — 100</div>'
            '</div>',
            unsafe_allow_html=True
        )

        if len(scores)>1:
            avg=np.mean(list(scores.values()))
            tc_avg,bg_avg,lbl_avg=verdict_colors(avg)
            votes=[verdict_colors(s)[2] for s in scores.values()]
            majority=max(set(votes),key=votes.count)
            st.markdown(
                f'<div class="consensus">'
                f'<div class="consensus-hdr">Keputusan Konsensus</div>'
                f'<div class="consensus-verdict">{majority}</div>'
                f'<div class="consensus-sub">Rata-rata tiga model: '
                f'<b style="color:{tc_avg}">{avg:.1f}</b></div>'
                f'</div>',
                unsafe_allow_html=True
            )

# TAB 2
with tab2:
    if not models.get('metrics'):
        st.info("File `metrics.pkl` tidak ditemukan di folder yang sama.")
    else:
        mv=models['metrics']
        st.markdown(
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;'
            f'color:#6a5642;margin-bottom:16px">'
            f'{mv["n_test"]} data test · total {mv["n_train"]+mv["n_test"]} data</div>',
            unsafe_allow_html=True
        )

        met_df=pd.DataFrame({
            'Metode':['FIS Manual','GA-FIS','ANN'],
            'RMSE':  [mv['fis']['rmse'],mv['ga']['rmse'],mv['ann']['rmse']],
            'MAE':   [mv['fis']['mae'], mv['ga']['mae'], mv['ann']['mae']],
            'R2':    [mv['fis']['r2'],  mv['ga']['r2'],  mv['ann']['r2']],
        }).set_index('Metode')
        st.dataframe(met_df.style.format('{:.4f}'), use_container_width=True)

        fig,(a1,a2)=plt.subplots(1,2,figsize=(9,3.5))
        names=['FIS Manual','GA-FIS','ANN']
        rmses=[mv['fis']['rmse'],mv['ga']['rmse'],mv['ann']['rmse']]
        r2s  =[mv['fis']['r2'],  mv['ga']['r2'],  mv['ann']['r2']]

        for ax in(a1,a2):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        bars=a1.bar(names,rmses,color=BR_SHADES,width=0.42,edgecolor='none')
        a1.set_title('RMSE — lebih rendah lebih baik',fontsize=8.5,color='#6a5642',pad=10)
        for b,v in zip(bars,rmses):
            a1.text(b.get_x()+b.get_width()/2,v+0.04,f'{v:.3f}',ha='center',fontsize=8,color='#d4c4ae')

        bars2=a2.bar(names,r2s,color=BR_SHADES,width=0.42,edgecolor='none')
        a2.set_title('R² — lebih tinggi lebih baik',fontsize=8.5,color='#6a5642',pad=10)
        a2.set_ylim(0,1.15)
        for b,v in zip(bars2,r2s):
            a2.text(b.get_x()+b.get_width()/2,v+0.01,f'{v:.3f}',ha='center',fontsize=8,color='#d4c4ae')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        best=min(['fis','ga','ann'],key=lambda k:mv[k]['rmse'])
        best_name={'fis':'FIS Manual','ga':'GA-FIS','ann':'ANN'}[best]
        st.markdown(
            f'<div class="best-pill">'
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.58rem;'
            f'letter-spacing:0.14em;text-transform:uppercase;color:#6a5642;margin-bottom:6px">Model Terbaik</div>'
            f'<span style="font-family:Cormorant Garamond,serif;font-size:1.5rem;'
            f'font-weight:700;color:#d4b48a">{best_name}</span>'
            f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.74rem;'
            f'color:#6a5642;margin-left:14px">'
            f'RMSE {mv[best]["rmse"]:.4f} &nbsp;·&nbsp; R² {mv[best]["r2"]:.4f}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

st.divider()
st.markdown(
    '<div style="text-align:center;font-family:IBM Plex Mono,monospace;font-size:0.56rem;'
    'color:#3a2e22;padding:6px"> Kelayakan Kredit · 230021-230026-230059</div>',
    unsafe_allow_html=True
)