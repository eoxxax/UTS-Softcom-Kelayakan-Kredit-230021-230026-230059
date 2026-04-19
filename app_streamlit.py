import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, os, copy
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Credit Intelligence Battle", page_icon="💳",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:#080b0f;color:#d4c9b8;}
.hero-wrap{background:linear-gradient(135deg,#0d1117 0%,#111820 100%);border:1px solid #1e2a38;border-radius:20px;padding:36px 40px;margin-bottom:28px;position:relative;overflow:hidden;}
.hero-wrap::before{content:'';position:absolute;top:-60px;right:-60px;width:220px;height:220px;background:radial-gradient(circle,#c9a86c22 0%,transparent 70%);border-radius:50%;}
.hero-title{font-family:'Playfair Display',serif;font-size:2.6rem;font-weight:800;color:#c9a86c;line-height:1.1;margin:0 0 6px 0;}
.hero-sub{font-family:'DM Mono',monospace;font-size:0.75rem;color:#556;letter-spacing:0.18em;text-transform:uppercase;}
.stage-chip{display:inline-block;padding:3px 12px;border-radius:20px;margin-bottom:10px;font-family:'DM Mono',monospace;font-size:0.68rem;letter-spacing:0.12em;text-transform:uppercase;font-weight:500;}
.chip-human{background:#1a1508;color:#c9a86c;border:1px solid #c9a86c55;}
.chip-ga{background:#081a12;color:#3ecf8e;border:1px solid #3ecf8e55;}
.chip-ann{background:#150818;color:#a78bfa;border:1px solid #a78bfa55;}
.kpi-card{background:#0d1117;border:1px solid #1e2a38;border-radius:14px;padding:20px 16px;text-align:center;}
.kpi-val{font-family:'Playfair Display',serif;font-size:2rem;font-weight:800;}
.kpi-lbl{font-family:'DM Mono',monospace;font-size:0.65rem;color:#445;letter-spacing:0.12em;text-transform:uppercase;margin-top:4px;}
.winner-box{background:linear-gradient(135deg,#0d1117,#111820);border:2px solid #c9a86c;border-radius:16px;padding:22px;text-align:center;font-family:'Playfair Display',serif;font-size:1.5rem;color:#c9a86c;margin-top:20px;}
.info-strip{background:#0a1628;border:1px solid #1e3a5f;border-radius:10px;padding:10px 16px;font-family:'DM Mono',monospace;font-size:0.72rem;color:#4a90d9;margin-bottom:16px;}
.stButton>button{background:linear-gradient(135deg,#c9a86c,#a07840)!important;color:#080b0f!important;border:none!important;border-radius:8px!important;width:100%!important;font-family:'DM Mono',monospace!important;font-weight:500!important;letter-spacing:0.06em!important;padding:0.6rem 1.2rem!important;}
div[data-testid="stExpander"]{background:#0d1117;border:1px solid #1e2a38!important;border-radius:12px;}
</style>
""", unsafe_allow_html=True)

BG='#080b0f'; CARD='#0d1117'; GOLD='#c9a86c'; TEAL='#3ecf8e'; PURP='#a78bfa'
plt.rcParams.update({'figure.facecolor':BG,'axes.facecolor':CARD,'axes.edgecolor':'#1e2a38',
    'axes.labelcolor':'#778','xtick.color':'#556','ytick.color':'#556',
    'text.color':'#d4c9b8','grid.color':'#1e2a38','font.family':'monospace'})

# ──────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file=None):
    if file is not None:
        df = pd.read_csv(file)
        return df, "upload"
    local_path = "dataset_kredit.csv"
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        return df, "local"
    # Fallback sintetis
    np.random.seed(99); n = 200
    pend  = np.random.uniform(1e6,  20e6,  n)
    pinj  = np.random.uniform(5e6,  200e6, n)
    lama  = np.random.uniform(0,    30,    n)
    riw   = np.random.uniform(0,    100,   n)
    rasio = np.random.uniform(0,    0.9,   n)
    skor  = (0.25*(pend/20e6*100)+0.20*riw+0.20*(lama/30*100)
             +0.20*((1-rasio)*100)+0.15*((1-pinj/200e6)*100)
             +np.random.normal(0,4,n)).clip(0,100)
    lbl = lambda s: "Layak" if s>=70 else ("Pertimbangkan" if s>=45 else "Tidak Layak")
    df = pd.DataFrame({
        'pendapatan_bulanan':  pend.round(0).astype(int),
        'jumlah_pinjaman':     pinj.round(0).astype(int),
        'lama_bekerja_tahun':  lama.round(1),
        'riwayat_kredit_skor': riw.round(1),
        'rasio_hutang':        rasio.round(3),
        'skor_kelayakan':      skor.round(2),
        'label_kelayakan':     [lbl(s) for s in skor],
    })
    return df, "synthetic"

# ──────────────────────────────────────────────────────────────────
# TRAIN-TEST SPLIT
# ──────────────────────────────────────────────────────────────────
@st.cache_data
def split_data(df, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state,
        stratify=df['label_kelayakan'],
    )
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)

# ──────────────────────────────────────────────────────────────────
# FUZZY
# ──────────────────────────────────────────────────────────────────
def trimf(x,a,b,c):
    x=np.asarray(x,float);y=np.zeros_like(x)
    m=(x>=a)&(x<=b)&(b!=a);y[m]=(x[m]-a)/(b-a)
    m=(x>b)&(x<=c)&(c!=b);y[m]=(c-x[m])/(c-b)
    y[x==b]=1.0;return y

def trapmf(x,a,b,c,d):
    x=np.asarray(x,float);y=np.zeros_like(x)
    m=(x>=a)&(x<=b)&(b!=a);y[m]=(x[m]-a)/(b-a)
    y[(x>b)&(x<c)]=1.0
    m=(x>=c)&(x<=d)&(d!=c);y[m]=(d-x[m])/(d-c);return y

def mf_eval(mf,x):
    t,*a=mf;return trimf(x,*a) if t=='tri' else trapmf(x,*a)

def fuzzify(v,D):
    return {l:float(mf_eval(m,np.array([v]))[0]) for l,m in D.items()}

def defuzz(out_mfs,strengths,lo=0,hi=100):
    x=np.linspace(lo,hi,600);agg=np.zeros_like(x)
    for l,s in strengths.items():
        if s>0:agg=np.maximum(agg,np.minimum(s,mf_eval(out_mfs[l],x)))
    return float(np.sum(x*agg)/np.sum(agg)) if agg.sum()>0 else 50.0

def run_fis(row,P):
    pend =(row['pendapatan_bulanan']/20e6)*100
    pinj =(1-row['jumlah_pinjaman']/200e6)*100
    lama =(row['lama_bekerja_tahun']/30)*100
    riw  = row['riwayat_kredit_skor']
    rasio=(1-row['rasio_hutang'])*100
    fp=fuzzify(pend,P['pendapatan']);fj=fuzzify(pinj,P['pinjaman'])
    fl=fuzzify(lama,P['lama_kerja']);fr=fuzzify(riw,P['riwayat'])
    fd=fuzzify(rasio,P['rasio_hutang'])
    st2={}
    for rule in P['rules']:
        p,j,l,r,d,o=rule
        s=min(fp.get(p,0),fj.get(j,0),fl.get(l,0),fr.get(r,0),fd.get(d,0))
        st2[o]=max(st2.get(o,0),s)
    return defuzz(P['output'],st2)

def calc_metrics(actual,preds):
    actual,preds=np.array(actual),np.array(preds)
    rmse=float(np.sqrt(np.mean((preds-actual)**2)))
    mae=float(np.mean(np.abs(preds-actual)))
    ss_res=np.sum((actual-preds)**2);ss_tot=np.sum((actual-actual.mean())**2)
    r2=float(1-ss_res/ss_tot) if ss_tot else 0.0
    return {'rmse':rmse,'mae':mae,'r2':r2,'preds':preds}

def evaluate_fis(P,df):
    preds=df.apply(lambda r:run_fis(r,P),axis=1).values
    return calc_metrics(df['skor_kelayakan'].values,preds)

# ──────────────────────────────────────────────────────────────────
# MANUAL PARAMS
# ──────────────────────────────────────────────────────────────────
MANUAL={
    'pendapatan':{'rendah':('trap',0,0,25,40),'sedang':('tri',30,50,70),'tinggi':('trap',60,80,100,100)},
    'pinjaman':{'besar':('trap',0,0,30,45),'sedang':('tri',35,55,75),'kecil':('trap',65,80,100,100)},
    'lama_kerja':{'baru':('trap',0,0,20,35),'sedang':('tri',25,45,65),'veteran':('trap',55,75,100,100)},
    'riwayat':{'buruk':('trap',0,0,30,50),'sedang':('tri',40,60,80),'baik':('trap',70,85,100,100)},
    'rasio_hutang':{'tinggi_hutang':('trap',0,0,30,50),'sedang_hutang':('tri',40,60,75),'rendah_hutang':('trap',65,80,100,100)},
    'output':{'tidak_layak':('trap',0,0,30,50),'pertimbangkan':('tri',40,55,70),'layak':('trap',62,80,100,100)},
    'rules':[
        ('rendah','besar','baru','buruk','tinggi_hutang','tidak_layak'),
        ('rendah','besar','baru','sedang','tinggi_hutang','tidak_layak'),
        ('rendah','besar','sedang','buruk','tinggi_hutang','tidak_layak'),
        ('rendah','sedang','baru','buruk','tinggi_hutang','tidak_layak'),
        ('rendah','sedang','sedang','sedang','sedang_hutang','pertimbangkan'),
        ('sedang','besar','sedang','sedang','sedang_hutang','pertimbangkan'),
        ('sedang','sedang','sedang','sedang','sedang_hutang','pertimbangkan'),
        ('sedang','sedang','veteran','sedang','rendah_hutang','pertimbangkan'),
        ('sedang','kecil','sedang','baik','rendah_hutang','layak'),
        ('tinggi','sedang','sedang','baik','rendah_hutang','layak'),
        ('tinggi','kecil','veteran','baik','rendah_hutang','layak'),
        ('tinggi','kecil','sedang','baik','sedang_hutang','layak'),
        ('sedang','kecil','veteran','baik','rendah_hutang','layak'),
        ('rendah','kecil','veteran','baik','rendah_hutang','pertimbangkan'),
        ('tinggi','besar','veteran','buruk','tinggi_hutang','pertimbangkan'),
        ('tinggi','sedang','veteran','sedang','sedang_hutang','layak'),
        ('sedang','sedang','baru','buruk','rendah_hutang','pertimbangkan'),
        ('rendah','sedang','sedang','baik','rendah_hutang','pertimbangkan'),
        ('tinggi','kecil','veteran','sedang','rendah_hutang','layak'),
        ('sedang','kecil','baru','baik','rendah_hutang','layak'),
    ],
}

# ──────────────────────────────────────────────────────────────────
# GENETIC ALGORITHM
# ──────────────────────────────────────────────────────────────────
VARS=['pendapatan','pinjaman','lama_kerja','riwayat','rasio_hutang']
LBLS={'pendapatan':['rendah','sedang','tinggi'],'pinjaman':['besar','sedang','kecil'],
      'lama_kerja':['baru','sedang','veteran'],'riwayat':['buruk','sedang','baik'],
      'rasio_hutang':['tinggi_hutang','sedang_hutang','rendah_hutang']}
N_GENES=sum(len(v) for v in LBLS.values())

def decode_chrom(chrom,base):
    P=copy.deepcopy(base);idx=0
    for var in VARS:
        for lbl in LBLS[var]:
            shift=(chrom[idx]-0.5)*10
            mf=list(P[var][lbl])
            if mf[0]=='tri':
                a,b,c=mf[1],mf[2],mf[3]
                P[var][lbl]=('tri',a,float(np.clip(b+shift,a+1,c-1)),c)
            else:
                a,b,c,d=mf[1],mf[2],mf[3],mf[4]
                nb=float(np.clip(b+shift,a,c));nc=float(np.clip(c+shift,nb,d))
                P[var][lbl]=('trap',a,nb,nc,d)
            idx+=1
    return P

def run_ga(base, df_train, df_test, pop_size=30, n_gen=40, mut=0.15, cx=0.8, cb=None):
    pop=np.random.rand(pop_size,N_GENES);best_c,best_f,hist=None,-np.inf,[]
    for gen in range(n_gen):
        fit=np.array([-evaluate_fis(decode_chrom(c,base),df_train)['rmse'] for c in pop])
        ib=np.argmax(fit)
        if fit[ib]>best_f:best_f=fit[ib];best_c=pop[ib].copy()
        hist.append(-best_f)
        if cb:cb(gen+1,n_gen,-best_f)
        new=[]
        for _ in range(pop_size):
            t=np.random.choice(pop_size,4,replace=False);new.append(pop[t[np.argmax(fit[t])]].copy())
        new=np.array(new)
        for i in range(0,pop_size-1,2):
            if np.random.rand()<cx:
                pt=np.random.randint(1,N_GENES);new[i,pt:],new[i+1,pt:]=new[i+1,pt:].copy(),new[i,pt:].copy()
        pop=np.clip(new+((np.random.rand(pop_size,N_GENES)<mut)*np.random.randn(pop_size,N_GENES)*0.15),0,1)
    best_params=decode_chrom(best_c,base)
    test_metrics=evaluate_fis(best_params,df_test)       # ← eval di TEST
    return best_params,hist,test_metrics

# ──────────────────────────────────────────────────────────────────
# ANN
# ──────────────────────────────────────────────────────────────────
def relu(x):return np.maximum(0,x)
def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-500,500)))

class ANN:
    def __init__(self,n_in=5,hidden=20,lr=0.005,seed=7):
        np.random.seed(seed);self.lr=lr
        self.W1=np.random.randn(n_in,hidden)*0.1;      self.b1=np.zeros(hidden)
        self.W2=np.random.randn(hidden,hidden//2)*0.1;  self.b2=np.zeros(hidden//2)
        self.W3=np.random.randn(hidden//2,1)*0.1;       self.b3=np.zeros(1)

    def _prep(self,df):
        return np.column_stack([
            df['pendapatan_bulanan'].values/20e6,
            1-df['jumlah_pinjaman'].values/200e6,
            df['lama_bekerja_tahun'].values/30,
            df['riwayat_kredit_skor'].values/100,
            1-df['rasio_hutang'].values,
        ])

    def forward(self,X):
        a1=relu(X@self.W1+self.b1);a2=relu(a1@self.W2+self.b2)
        return sigmoid(a2@self.W3+self.b3).squeeze()

    def train(self,X,y,epochs=200,batch=32,cb=None):
        hist=[]
        for ep in range(epochs):
            idx=np.random.permutation(len(X));ep_loss=0
            for i in range(0,len(X),batch):
                bi=idx[i:i+batch];Xb,yb=X[bi],y[bi]
                z1=Xb@self.W1+self.b1;a1=relu(z1)
                z2=a1@self.W2+self.b2;a2=relu(z2)
                z3=a2@self.W3+self.b3;out=sigmoid(z3).squeeze()
                ep_loss+=np.mean((out-yb)**2)
                do=(2*(out-yb)/len(yb)).reshape(-1,1)
                ds=out.reshape(-1,1)*(1-out.reshape(-1,1))
                dz3=do*ds;dW3=a2.T@dz3;db3=dz3.sum(0)
                dz2=(dz3@self.W3.T)*(a2>0);dW2=a1.T@dz2;db2=dz2.sum(0)
                dz1=(dz2@self.W2.T)*(a1>0);dW1=Xb.T@dz1;db1=dz1.sum(0)
                for w,g in [(self.W1,dW1),(self.b1,db1),(self.W2,dW2),
                            (self.b2,db2),(self.W3,dW3),(self.b3,db3)]:w-=self.lr*g
            hist.append(ep_loss)
            if cb:cb(ep+1,epochs,ep_loss)
        return hist

    def predict(self,X):return self.forward(X)*100

def run_ann(df_train, df_test, hidden=20, lr=0.005, epochs=200, cb=None):
    model=ANN(n_in=5,hidden=hidden,lr=lr)
    X_train=model._prep(df_train);y_train=df_train['skor_kelayakan'].values/100
    X_test=model._prep(df_test);  y_test=df_test['skor_kelayakan'].values
    hist=model.train(X_train,y_train,epochs=epochs,cb=cb)
    preds=model.predict(X_test)
    test_metrics=calc_metrics(y_test,preds)              # ← eval di TEST
    return model,hist,test_metrics

def ann_single(model,row):
    x=model._prep(pd.DataFrame([row]))
    out=model.predict(x)
    return float(out[0] if hasattr(out,'__len__') else out)

# ──────────────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────────────
C4=[GOLD,TEAL,PURP,'#f87171']

def fig_mf(P,var,title):
    fig,ax=plt.subplots(figsize=(4.8,2.5));x=np.linspace(0,100,400)
    for ci,(l,m) in enumerate(P[var].items()):
        y=mf_eval(m,x);ax.plot(x,y,color=C4[ci%4],lw=2,label=l)
        ax.fill_between(x,y,alpha=0.08,color=C4[ci%4])
    ax.set_title(title,fontsize=8.5,color='#a9a090',pad=6)
    ax.legend(fontsize=7,facecolor=CARD,edgecolor='#1e2a38',labelcolor='#d4c9b8',loc='upper right')
    ax.set_ylim(-0.05,1.15);fig.tight_layout(pad=0.5);return fig

def fig_scatter(actual,pred,title,color):
    fig,ax=plt.subplots(figsize=(4,3.5))
    ax.scatter(actual,pred,s=16,alpha=0.5,color=color,edgecolors='none')
    mn,mx=min(actual.min(),pred.min()),max(actual.max(),pred.max())
    ax.plot([mn,mx],[mn,mx],'--',color='#334',lw=1.2)
    ax.set_xlabel('Aktual (Test Set)');ax.set_ylabel('Prediksi')
    ax.set_title(title,fontsize=9,color='#a9a090');fig.tight_layout(pad=0.5);return fig

def fig_curve(hist,title,color,ylabel='RMSE'):
    fig,ax=plt.subplots(figsize=(4.8,2.5))
    ax.plot(hist,color=color,lw=2);ax.fill_between(range(len(hist)),hist,alpha=0.12,color=color)
    ax.set_xlabel('Iterasi');ax.set_ylabel(ylabel)
    ax.set_title(title,fontsize=8.5,color='#a9a090');fig.tight_layout(pad=0.5);return fig

def fig_compare(results):
    labels=list(results.keys());rmses=[results[l]['rmse'] for l in labels];r2s=[results[l]['r2'] for l in labels]
    cols=[GOLD,TEAL,PURP]
    fig,(a1,a2)=plt.subplots(1,2,figsize=(7,3))
    for ax in (a1,a2):ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    bars=a1.bar(labels,rmses,color=cols,width=0.45,edgecolor='none')
    a1.set_title('RMSE  ↓ lebih baik',fontsize=8.5,color='#a9a090')
    for b,v in zip(bars,rmses):a1.text(b.get_x()+b.get_width()/2,v+0.2,f'{v:.2f}',ha='center',fontsize=8,color='#d4c9b8')
    bars2=a2.bar(labels,r2s,color=cols,width=0.45,edgecolor='none')
    a2.set_title('R²  ↑ lebih baik',fontsize=8.5,color='#a9a090');a2.set_ylim(0,1.12)
    for b,v in zip(bars2,r2s):a2.text(b.get_x()+b.get_width()/2,v+0.01,f'{v:.3f}',ha='center',fontsize=8,color='#d4c9b8')
    fig.tight_layout(pad=1);return fig

# ──────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family:\'Playfair Display\',serif;font-size:1.3rem;color:#c9a86c;font-weight:800">💳 Credit<br>Intelligence</p>',unsafe_allow_html=True)
    st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:#445;letter-spacing:0.15em">KELAYAKAN KREDIT · UTS</p>',unsafe_allow_html=True)
    uploaded=st.file_uploader("📂 Upload dataset_kredit.csv",type='csv')
    df_full, data_source = load_data(uploaded)

    if data_source == "synthetic":
        st.warning("⚠️ Data sintetis — upload CSV asli untuk hasil valid.")
    elif data_source == "local":
        st.success("✅ Dataset lokal dimuat.")
    else:
        st.success("✅ Dataset ter-upload dimuat.")

    st.divider()
    st.markdown("**🔀 Train-Test Split**")
    test_sz = st.slider("Test Size", 0.10, 0.40, 0.20, 0.05)
    st.caption(f"Train: {int(len(df_full)*(1-test_sz))}  |  Test: {int(len(df_full)*test_sz)} baris")

    # Split dilakukan sekali di sini
    df_train, df_test = split_data(df_full, test_size=test_sz)

    st.divider()
    st.markdown("**⚙️ Parameter GA**")
    ga_pop=st.slider("Populasi",10,80,30,5);ga_gen=st.slider("Generasi",10,100,40,5)
    ga_mut=st.slider("Mutation Rate",0.05,0.40,0.15,0.05);ga_cx=st.slider("Crossover",0.50,1.00,0.80,0.05)
    st.divider()
    st.markdown("**⚙️ Parameter ANN**")
    ann_h=st.slider("Hidden Neurons",8,64,20,4)
    ann_lr=st.select_slider("Learning Rate",options=[0.001,0.003,0.005,0.01,0.02],value=0.005)
    ann_ep=st.slider("Epochs",50,500,200,25)
    st.divider()
    run_all=st.button("🚀 Jalankan Semua Tahap")

for k in ['fis_r','ga_r','ga_h','ga_p','ann_r','ann_h_hist','ann_m']:
    if k not in st.session_state: st.session_state[k]=None

# ──────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────
st.markdown("""<div class="hero-wrap">
  <div class="hero-title">The Intelligence Battle</div>
  <div style="font-family:'Playfair Display',serif;font-size:1.1rem;color:#7a6840;margin:4px 0 10px">Sistem Penentu Kelayakan Kredit</div>
  <div class="hero-sub">Human Expert (FIS) &nbsp;vs&nbsp; Evolutionary Tuning (GA) &nbsp;vs&nbsp; Neural Network (ANN)</div>
</div>""",unsafe_allow_html=True)

# Info strip split
st.markdown(f'<div class="info-strip">📊 Dataset: <b>{len(df_full)}</b> baris &nbsp;|&nbsp; 🏋️ Train: <b>{len(df_train)}</b> &nbsp;|&nbsp; 🧪 Test: <b>{len(df_test)}</b> &nbsp;|&nbsp; Semua metrik dihitung di <b>TEST SET</b></div>', unsafe_allow_html=True)

with st.expander("📊 Preview Dataset & Split"):
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Total Data",len(df_full))
    c2.metric("Train Set",len(df_train))
    c3.metric("Test Set",len(df_test))
    c4.metric("Layak (test)",int((df_test['label_kelayakan']=='Layak').sum()))
    t1,t2=st.tabs(["Train","Test"])
    with t1: st.dataframe(df_train.head(10),use_container_width=True,hide_index=True)
    with t2: st.dataframe(df_test.head(10),use_container_width=True,hide_index=True)

tab1,tab2,tab3,tab4,tab5=st.tabs([
    "🧑 Tahap 1 · Manual FIS","🧬 Tahap 2 · GA Tuning",
    "🤖 Tahap 3 · ANN","🏆 Perbandingan","🔍 Uji Prediksi Baru"])

# ═══ TAHAP 1 ═══════════════════════════════════
with tab1:
    st.markdown('<div class="stage-chip chip-human">Tahap 1 · Human Intuition — Manual FIS Mamdani</div>',unsafe_allow_html=True)
    st.markdown("FIS Mamdani, 5 variabel input, 20 rules pakar. **Dievaluasi di test set.**")

    c1,c2,c3=st.columns(3)
    with c1:
        st.pyplot(fig_mf(MANUAL,'pendapatan','Pendapatan Bulanan'))
        st.pyplot(fig_mf(MANUAL,'lama_kerja','Lama Bekerja'))
    with c2:
        st.pyplot(fig_mf(MANUAL,'pinjaman','Jumlah Pinjaman (inv)'))
        st.pyplot(fig_mf(MANUAL,'riwayat','Riwayat Kredit'))
    with c3:
        st.pyplot(fig_mf(MANUAL,'rasio_hutang','Rasio Hutang (inv)'))
        st.pyplot(fig_mf(MANUAL,'output','Output: Skor Kelayakan'))

    with st.expander("📋 Lihat 20 Rules Manual"):
        st.dataframe(pd.DataFrame(MANUAL['rules'],
            columns=['Pendapatan','Pinjaman','Lama Kerja','Riwayat','Rasio Hutang','Output']),
            use_container_width=True,hide_index=True)

    # ← evaluasi di df_TEST
    fis_r = evaluate_fis(MANUAL, df_test)
    st.session_state['fis_r'] = fis_r

    c1,c2,c3=st.columns(3)
    c1.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:{GOLD}">{fis_r["rmse"]:.2f}</div><div class="kpi-lbl">RMSE (test)</div></div>',unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:{GOLD}">{fis_r["mae"]:.2f}</div><div class="kpi-lbl">MAE (test)</div></div>',unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:{GOLD}">{fis_r["r2"]:.3f}</div><div class="kpi-lbl">R² (test)</div></div>',unsafe_allow_html=True)
    st.pyplot(fig_scatter(df_test['skor_kelayakan'].values, fis_r['preds'], 'Aktual vs Prediksi – Manual FIS (Test)', GOLD))

# ═══ TAHAP 2 ═══════════════════════════════════
with tab2:
    st.markdown('<div class="stage-chip chip-ga">Tahap 2 · Evolutionary Tuning — Genetic Algorithm</div>',unsafe_allow_html=True)
    st.markdown("GA mengoptimasi 15 parameter MF. **Fitness dihitung di train set. Hasil dievaluasi di test set.**")

    if run_all or st.button("▶ Jalankan GA",key="btn_ga"):
        pb=st.progress(0);info=st.empty()
        def ga_cb(g,T,r):pb.progress(g/T);info.markdown(f"`Gen {g}/{T}` RMSE train: **{r:.4f}**")
        with st.spinner("Evolusi berlangsung…"):
            gp,gh,gr=run_ga(MANUAL,df_train,df_test,      # ← pisah train/test
                            pop_size=ga_pop,n_gen=ga_gen,mut=ga_mut,cx=ga_cx,cb=ga_cb)
        pb.empty();info.empty()
        st.session_state.update({'ga_r':gr,'ga_h':gh,'ga_p':gp});st.success("✅ GA selesai!")

    if st.session_state['ga_r']:
        gr,gh,gp=st.session_state['ga_r'],st.session_state['ga_h'],st.session_state['ga_p']
        c1,c2,c3=st.columns(3)
        c1.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:{TEAL}">{gr["rmse"]:.2f}</div><div class="kpi-lbl">RMSE (test)</div></div>',unsafe_allow_html=True)
        c2.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:{TEAL}">{gr["mae"]:.2f}</div><div class="kpi-lbl">MAE (test)</div></div>',unsafe_allow_html=True)
        c3.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:{TEAL}">{gr["r2"]:.3f}</div><div class="kpi-lbl">R² (test)</div></div>',unsafe_allow_html=True)
        ca,cb2=st.columns(2)
        with ca:st.pyplot(fig_curve(gh,'Konvergensi GA — RMSE Train',TEAL))
        with cb2:st.pyplot(fig_scatter(df_test['skor_kelayakan'].values,gr['preds'],'Aktual vs Prediksi – GA-FIS (Test)',TEAL))
        st.markdown("##### MF Setelah GA Optimasi")
        c1,c2,c3=st.columns(3)
        with c1:st.pyplot(fig_mf(gp,'pendapatan','Pendapatan (GA)'))
        with c2:st.pyplot(fig_mf(gp,'riwayat','Riwayat Kredit (GA)'))
        with c3:st.pyplot(fig_mf(gp,'rasio_hutang','Rasio Hutang (GA)'))
    else:
        st.info("Klik **▶ Jalankan GA** atau **🚀 Jalankan Semua Tahap** di sidebar.")

# ═══ TAHAP 3 ═══════════════════════════════════
with tab3:
    st.markdown('<div class="stage-chip chip-ann">Tahap 3 · ANN Tuning — Backpropagation</div>',unsafe_allow_html=True)
    st.markdown("ANN 3-layer, ReLU + Sigmoid. **Train di train set. Evaluasi di test set.**")

    if run_all or st.button("▶ Jalankan ANN",key="btn_ann"):
        pb=st.progress(0);info=st.empty()
        def ann_cb(e,T,l):pb.progress(e/T);info.markdown(f"`Epoch {e}/{T}` Loss: **{l:.5f}**")
        with st.spinner("Neural network berlatih…"):
            am,ah_hist,ar=run_ann(df_train,df_test,       # ← pisah train/test
                                  hidden=ann_h,lr=ann_lr,epochs=ann_ep,cb=ann_cb)
        pb.empty();info.empty()
        st.session_state.update({'ann_r':ar,'ann_h_hist':ah_hist,'ann_m':am});st.success("✅ ANN selesai!")

    if st.session_state['ann_r']:
        ar,ah_hist=st.session_state['ann_r'],st.session_state['ann_h_hist']
        c1,c2,c3=st.columns(3)
        c1.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:{PURP}">{ar["rmse"]:.2f}</div><div class="kpi-lbl">RMSE (test)</div></div>',unsafe_allow_html=True)
        c2.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:{PURP}">{ar["mae"]:.2f}</div><div class="kpi-lbl">MAE (test)</div></div>',unsafe_allow_html=True)
        c3.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:{PURP}">{ar["r2"]:.3f}</div><div class="kpi-lbl">R² (test)</div></div>',unsafe_allow_html=True)
        ca,cb2=st.columns(2)
        with ca:st.pyplot(fig_curve(ah_hist,'Loss Curve ANN (Train)',PURP,ylabel='Loss'))
        with cb2:st.pyplot(fig_scatter(df_test['skor_kelayakan'].values,ar['preds'],'Aktual vs Prediksi – ANN (Test)',PURP))
        st.markdown(f"**Arsitektur:** `Input(5)` → `ReLU({ann_h})` → `ReLU({ann_h//2})` → `Sigmoid(1)` → ×100")
    else:
        st.info("Klik **▶ Jalankan ANN** atau **🚀 Jalankan Semua Tahap** di sidebar.")

# ═══ PERBANDINGAN ════════════════════════════════
with tab4:
    st.markdown("### 🏆 Perbandingan Ketiga Metode — Test Set")
    fr,gr2,ar2=st.session_state['fis_r'],st.session_state['ga_r'],st.session_state['ann_r']
    if not (gr2 and ar2):
        st.info("Jalankan Tahap 2 (GA) dan Tahap 3 (ANN) terlebih dahulu.")
    else:
        results={'Manual FIS':fr,'GA-FIS':gr2,'ANN':ar2}
        st.dataframe(pd.DataFrame({'Metode':list(results.keys()),
            'RMSE ↓':[f"{v['rmse']:.3f}" for v in results.values()],
            'MAE ↓':[f"{v['mae']:.3f}" for v in results.values()],
            'R² ↑':[f"{v['r2']:.4f}" for v in results.values()],
        }).set_index('Metode'),use_container_width=True)
        st.pyplot(fig_compare(results))
        best_r=min(results,key=lambda k:results[k]['rmse'])
        best_r2=max(results,key=lambda k:results[k]['r2'])
        em={'Manual FIS':'🧑','GA-FIS':'🧬','ANN':'🤖'}
        cm={'Manual FIS':GOLD,'GA-FIS':TEAL,'ANN':PURP}
        st.markdown(f'<div class="winner-box">{em[best_r]} RMSE Terkecil: <span style="color:{cm[best_r]}">{best_r}</span> &nbsp;|&nbsp; {em[best_r2]} R² Tertinggi: <span style="color:{cm[best_r2]}">{best_r2}</span></div>',unsafe_allow_html=True)
        st.markdown("##### Peningkatan vs Manual FIS")
        for name,res in [('GA-FIS',gr2),('ANN',ar2)]:
            d=fr['rmse']-res['rmse'];s="✅ turun" if d>0 else "⚠️ naik"
            st.markdown(f"- **{name}** RMSE {s} **{abs(d):.3f}** poin ({abs(d)/fr['rmse']*100:.1f}%)")
        with st.expander("📝 Template Kesimpulan Laporan"):
            st.markdown(f"""
**Hasil Perbandingan – Sistem Penentu Kelayakan Kredit**

Evaluasi dilakukan pada **test set** ({len(df_test)} data, {int(test_sz*100)}% dari total {len(df_full)} data).
Split dilakukan secara stratified berdasarkan label kelayakan.

| Metode | RMSE | MAE | R² |
|---|---|---|---|
| Manual FIS | {fr['rmse']:.3f} | {fr['mae']:.3f} | {fr['r2']:.4f} |
| GA-FIS | {gr2['rmse']:.3f} | {gr2['mae']:.3f} | {gr2['r2']:.4f} |
| ANN | {ar2['rmse']:.3f} | {ar2['mae']:.3f} | {ar2['r2']:.4f} |

- **Manual FIS** (Mamdani, 5 input, 20 rules): RMSE={fr['rmse']:.3f}.
- **GA-FIS**: optimasi {ga_gen} generasi × {ga_pop} populasi. RMSE={'membaik' if gr2['rmse']<fr['rmse'] else 'belum membaik'} → {gr2['rmse']:.3f}.
- **ANN** (hidden={ann_h}, lr={ann_lr}, epoch={ann_ep}): RMSE={ar2['rmse']:.3f}.
- **Pemenang: {best_r}** (RMSE={results[best_r]['rmse']:.3f}).
            """)

# ═══ UJI PREDIKSI BARU ═══════════════════════════
with tab5:
    st.markdown("### 🔍 Uji Prediksi Data Calon Debitur Baru")
    st.markdown("Geser slider atau isi angka, lalu klik **Prediksi**. Hasil langsung keluar dari ketiga model.")

    # ── Preset contoh ────────────────────────────────────────────
    st.markdown("**💡 Preset Contoh Cepat:**")
    p1,p2,p3,p4 = st.columns(4)
    preset = None
    if p1.button("😊 Profil Ideal"):    preset='ideal'
    if p2.button("😐 Profil Sedang"):   preset='sedang'
    if p3.button("😟 Profil Berisiko"): preset='berisiko'
    if p4.button("🎲 Acak"):            preset='acak'

    # Tentukan default value berdasarkan preset
    if preset == 'ideal':
        dv = dict(pend=25_000_000, pinj=20_000_000, lama=18.0, riw=85.0, rasio=0.15)
    elif preset == 'sedang':
        dv = dict(pend=9_000_000,  pinj=55_000_000, lama=7.0,  riw=55.0, rasio=0.40)
    elif preset == 'berisiko':
        dv = dict(pend=3_500_000,  pinj=120_000_000,lama=1.5,  riw=22.0, rasio=0.75)
    elif preset == 'acak':
        np.random.seed(None)
        dv = dict(pend=int(np.random.uniform(2e6,40e6)),
                  pinj=int(np.random.uniform(5e6,180e6)),
                  lama=round(float(np.random.uniform(0.5,30)),1),
                  riw=round(float(np.random.uniform(5,100)),1),
                  rasio=round(float(np.random.uniform(0.05,0.90)),2))
    else:
        dv = dict(pend=8_000_000, pinj=50_000_000, lama=5.0, riw=65.0, rasio=0.30)

    st.divider()

    # ── Input form ───────────────────────────────────────────────
    col1,col2 = st.columns(2)
    with col1:
        pend_v  = st.number_input("💰 Pendapatan Bulanan (Rp)", 2_000_000, 40_000_000,
                                   value=dv['pend'], step=500_000, format="%d")
        pinj_v  = st.number_input("🏦 Jumlah Pinjaman (Rp)", 5_000_000, 180_000_000,
                                   value=dv['pinj'], step=5_000_000, format="%d")
        lama_v  = st.slider("🕐 Lama Bekerja (tahun)", 0.5, 30.0, value=dv['lama'], step=0.5)
    with col2:
        riw_v   = st.slider("⭐ Skor Riwayat Kredit (0–100)", 5.0, 100.0, value=dv['riw'], step=1.0)
        rasio_v = st.slider("📊 Rasio Hutang (0–0.9)", 0.05, 0.90, value=dv['rasio'], step=0.01)

        # Indikator visual rasio hutang
        rasio_color = TEAL if rasio_v < 0.35 else (GOLD if rasio_v < 0.60 else '#f87171')
        rasio_label = "Rendah ✅" if rasio_v < 0.35 else ("Sedang ⚠️" if rasio_v < 0.60 else "Tinggi ❌")
        st.markdown(f'<div style="margin-top:8px;padding:8px 14px;border-radius:8px;'
                    f'background:#0d1117;border:1px solid {rasio_color}33;'
                    f'font-family:DM Mono,monospace;font-size:0.78rem;color:{rasio_color}">'
                    f'Rasio hutang: <b>{rasio_label}</b></div>', unsafe_allow_html=True)

    # Ringkasan input
    st.markdown(f"""
    <div style="background:#0d1117;border:1px solid #1e2a38;border-radius:10px;
                padding:12px 18px;margin:12px 0;font-family:DM Mono,monospace;font-size:0.75rem;color:#778">
    Pendapatan <b style="color:#d4c9b8">Rp {pend_v:,.0f}</b> &nbsp;·&nbsp;
    Pinjaman <b style="color:#d4c9b8">Rp {pinj_v:,.0f}</b> &nbsp;·&nbsp;
    Lama kerja <b style="color:#d4c9b8">{lama_v} thn</b> &nbsp;·&nbsp;
    Riwayat <b style="color:#d4c9b8">{riw_v}/100</b> &nbsp;·&nbsp;
    Rasio <b style="color:#d4c9b8">{rasio_v}</b>
    </div>""", unsafe_allow_html=True)

    # ── Tombol prediksi ──────────────────────────────────────────
    if st.button("🔎 Prediksi Sekarang", use_container_width=True):
        new_row = pd.Series({'pendapatan_bulanan':pend_v,'jumlah_pinjaman':pinj_v,
                             'lama_bekerja_tahun':lama_v,'riwayat_kredit_skor':riw_v,
                             'rasio_hutang':rasio_v})

        def decode_lbl(s):
            if s>=65: return("✅ LAYAK", TEAL)
            elif s>=40: return("⚠️ PERTIMBANGKAN", GOLD)
            else: return("❌ TIDAK LAYAK","#f87171")

        s_fis = run_fis(new_row, MANUAL)
        lf, cf = decode_lbl(s_fis)

        r1,r2,r3 = st.columns(3)
        # FIS Manual — selalu tersedia
        r1.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-val" style="color:{cf}">{s_fis:.1f}</div>'
            f'<div class="kpi-lbl">Manual FIS</div>'
            f'<div style="font-size:0.9rem;margin-top:10px;color:{cf}">{lf}</div>'
            f'</div>', unsafe_allow_html=True)

        # GA-FIS
        if st.session_state['ga_p']:
            sg = run_fis(new_row, st.session_state['ga_p']); lg,cg = decode_lbl(sg)
            r2.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-val" style="color:{cg}">{sg:.1f}</div>'
                f'<div class="kpi-lbl">GA-FIS</div>'
                f'<div style="font-size:0.9rem;margin-top:10px;color:{cg}">{lg}</div>'
                f'</div>', unsafe_allow_html=True)
        else:
            r2.markdown('<div class="kpi-card" style="opacity:0.4">'
                        '<div class="kpi-lbl" style="margin-top:20px">GA-FIS</div>'
                        '<div style="font-size:0.8rem;color:#556;margin-top:8px">Jalankan Tahap 2 dulu</div>'
                        '</div>', unsafe_allow_html=True)

        # ANN
        if st.session_state['ann_m']:
            sa = ann_single(st.session_state['ann_m'], new_row); la,ca2 = decode_lbl(sa)
            r3.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-val" style="color:{ca2}">{sa:.1f}</div>'
                f'<div class="kpi-lbl">ANN</div>'
                f'<div style="font-size:0.9rem;margin-top:10px;color:{ca2}">{la}</div>'
                f'</div>', unsafe_allow_html=True)
        else:
            r3.markdown('<div class="kpi-card" style="opacity:0.4">'
                        '<div class="kpi-lbl" style="margin-top:20px">ANN</div>'
                        '<div style="font-size:0.8rem;color:#556;margin-top:8px">Jalankan Tahap 3 dulu</div>'
                        '</div>', unsafe_allow_html=True)

        # Gauge bar skor FIS
        st.markdown("##### Skor Kelayakan (Manual FIS)")
        pct = s_fis / 100
        bar_color = TEAL if s_fis>=65 else (GOLD if s_fis>=40 else '#f87171')
        st.markdown(f"""
        <div style="background:#1e2a38;border-radius:8px;height:18px;margin:6px 0 2px;overflow:hidden">
          <div style="width:{pct*100:.1f}%;height:100%;background:{bar_color};
                      border-radius:8px;transition:width 0.4s"></div>
        </div>
        <div style="font-family:DM Mono,monospace;font-size:0.7rem;color:#556">
          0 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
          Tidak Layak &lt;40 &nbsp;·&nbsp; Pertimbangkan 40–65 &nbsp;·&nbsp; Layak ≥65
          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 100
        </div>""", unsafe_allow_html=True)

st.divider()
st.markdown('<div style="text-align:center;font-family:DM Mono,monospace;font-size:0.65rem;color:#223;padding:8px">Intelligence Battle · UTS Sistem Cerdas · Penentu Kelayakan Kredit</div>',unsafe_allow_html=True)
