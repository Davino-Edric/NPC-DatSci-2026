# %% [markdown]
# ## 0. Setup Minimal & Reproducibility
# 
# Sel ini menggabungkan semua import, seed, dan variabel global.
# 
# Yang diatur di sini:
# - seed global: 42,
# - path data,
# - folder cache fitur,
# - ekstensi audio,
# - import library utama.
# 
# Tidak ada parameter model atau training di section ini.

# %%
import pandas as pd
import numpy as np
import os, time, inspect, glob, random, itertools, math
import matplotlib.pyplot as plt
import importlib.metadata as md
import soundfile as sf

import sqlite3
import librosa
from tqdm import tqdm
import birdnet
import librosa.display

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, classification_report, roc_auc_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

path = 'data/schematics-2026'
os.makedirs('submission', exist_ok=True)
path_key = lambda s: os.path.normcase(os.path.abspath(str(s))) 

CACHE = 'features/cache'
CACHE3 = 'features/cache_v3'
CACHE_OUT = CACHE3

EXT = ('.flac', '.wav', '.mp3', '.ogg')

CFG = dict(batch_size=16, n_workers=8, n_producers=2)

# %% [markdown]
# ## 1. EDA & Audit Data
# 
# Section ini memeriksa kondisi data sebelum ekstraksi fitur.
# 
# Yang dicek:
# - metadata train,
# - distribusi label,
# - jumlah file per label,
# - jumlah source_group per label,
# - distribusi file per source_group,
# - keberadaan file audio di disk,
# - sanity check StratifiedGroupKFold.
# 
# Ini penting karena model akan divalidasi berdasarkan source_group, bukan hanya label.

# %%
metas = pd.read_csv(f'{path}/train/metadata.csv')
metas

# %%
metas['label'].value_counts().plot(kind='bar', title='Label Distribution')
plt.show()

# %%
print(metas.groupby('label')['source_group'].nunique())
metas.groupby(['label', 'source_group']).size()

# %%
sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
for k, (tr, va) in enumerate(sgkf.split(metas, metas['label'], groups=metas['source_group'])):
    print(k, metas.iloc[va]['label'].value_counts().to_dict())

# %%
# Cell 2: audit train vs test audio (about 1-2 mins): sample rate + min/median/max seconds
test_files = [os.path.join(d, f) for d, _, fs in os.walk(f'{path}/test')
              for f in fs if f.lower().endswith(('.flac', '.wav', '.mp3', '.ogg'))]

def probe(paths):
    out = []
    for p in paths:
        y, sr = librosa.load(p, sr=None, mono=True)   # sr=None keeps the real rate
        out.append((sr, len(y) / sr))
    return np.array(out)

for lab, g in metas.groupby('label').head(15).groupby('label'):
    a = probe([f'{path}/{p}' for p in g['audio_path']])
    print(lab, set(a[:, 0]), np.percentile(a[:, 1], [0, 50, 100]).round(1))
a = probe(test_files[:45])
print('TEST', len(test_files), set(a[:, 0]), np.percentile(a[:, 1], [0, 50, 100]).round(1))

# %%
metas['have'] = metas['audio_path'].map(lambda p: os.path.exists(f'{path}/{p}'))
print(metas['have'].sum(), len(metas))
print(metas[metas['have']].groupby('label').size())

# %% [markdown]
# ## 2. Ekstraksi Fitur BirdNET v2.4
# 
# Pipeline fitur utama:
# - load BirdNET acoustic 2.4,
# - encode audio menjadi embeddings,
# - reshape menjadi (windows, 1024),
# - buang window NaN,
# - simpan cache per file agar bisa dilanjutkan ulang.
# 
# Cache disimpan di:
# - `features/cache/train`
# - `features/cache/test`
# 
# Embedding akhir per split dikumpulkan dengan fungsi `load_cache`.

# %%
model = birdnet.load("acoustic", "2.4", "tf", library="litert")   # first run downloads weights
print(inspect.signature(model.encode))                            # shows options like overlap, if any

def embed_file(fp):
    """(n_windows, 1024) embeddings for one file; windows with NaN are dropped."""
    res = model.encode(fp)
    e = np.asarray(res.embeddings, dtype=np.float32)
    e = e.reshape(-1, e.shape[-1]) # (inputs, segments, dim) -> (windows, dim)
    return e[~np.isnan(e).any(axis=1)]

fp = next(f'{path}/{x}' for x in metas['audio_path'] if os.path.exists(f'{path}/{x}'))
t = time.time(); e = embed_file(fp)
print(e.shape, round(time.time() - t, 2), 's') # dims 3 detik x 1024 

# %%
def extract(items, split):                          # items: list of (file_id, full_path)
    d = f'{CACHE}/{split}'; os.makedirs(d, exist_ok=True)
    X, ids, bad, new = [], [], [], 0
    for fid, fp in tqdm(items):
        out = f'{d}/{fid}.npy'
        try:
            if os.path.exists(out):                 # skip: this file was embedded before
                e = np.load(out)
            else:
                e = embed_file(fp); new += 1
                np.save(out + '.tmp.npy', e)        # write to a temp name first, so an interrupt
                os.replace(out + '.tmp.npy', out)   # never leaves a half-written cache file
            X.append(e); ids += [fid] * len(e)
        except Exception as ex:
            bad.append((fid, repr(ex)[:80]))        # failed files are not cached, so they retry next run
    X = np.vstack(X)
    np.savez_compressed(f'features/{split}_emb_{len(items)}.npz', X=X, ids=np.array(ids))
    print(split, X.shape, '| new:', new, '| failed:', len(bad), bad[:3])

# %%
for f in glob.glob('features/*_emb*.npz'):
    d = np.load(f); X, ids = d['X'], d['ids']
    split = 'test' if os.path.basename(f).startswith('test') else 'train'
    os.makedirs(f'{CACHE}/{split}', exist_ok=True)
    for fid in np.unique(ids):
        out = f'{CACHE}/{split}/{fid}.npy'
        if not os.path.exists(out):
            np.save(out, X[ids == fid])             # keeps the window order inside each file
    print('seeded from', f)

# %%
train_items = [(r.file_id, f'{path}/{r.audio_path}') for r in metas.itertuples() if os.path.exists(f'{path}/{r.audio_path}')]        # rebuilt, so it sees new downloads
test_items = [(os.path.splitext(f)[0], os.path.join(d, f)) for d, _, fs in os.walk(f'{path}/test') for f in fs if f.lower().endswith(EXT)]
print(len(train_items), len(test_items))
extract(test_items, 'test')
extract(train_items, 'train')

# %%
def load_cache(split, root='features/cache'):
    """Window-level embeddings from the per-file cache: (X, file_ids)."""
    d = f'{root}/{split}'
    fids = sorted(f[:-4] for f in os.listdir(d) if f.endswith('.npy') and not f.endswith('.tmp.npy'))
    X, ids = [], []
    for f in fids: e = np.load(f'{d}/{f}.npy'); X.append(e); ids += [f] * len(e)
    return np.vstack(X), np.array(ids)

def write_submission(pred, out='submission.csv'):
    """pred: pd.Series indexed by file_id, values = label strings. Checks every rule in the Submission section."""
    sample = pd.read_csv(glob.glob(f'{path}/**/sample_submission.csv', recursive=True)[0])
    assert pred.index.is_unique, 'duplicate file_id'
    assert set(pred.index) == set(sample['file_id']), 'file_id mismatch vs sample_submission'
    assert pred.isin(['Bondol Jawa', 'Cendrawasih', 'Kutilang', 'Unknown']).all(), 'bad label'
    sub = pd.DataFrame({'file_id': sample['file_id'], 'label': pred.loc[sample['file_id']].values})
    sub.to_csv(f'submission/{out}', index=False)
    print(out, sub.shape, sub['label'].value_counts().to_dict())

# %% [markdown]
# ## 3. Baseline Window-level Logistic Regression
# 
# Baseline pertama:
# - setiap window 3 detik diklasifikasikan,
# - probabilitas per window dirata-rata menjadi probabilitas file,
# - label file diambil dari argmax probabilitas rata-rata.
# 
# Validasi memakai StratifiedGroupKFold dengan `source_group` sebagai group.

# %%
# Cell 2: dry run on whatever train clips are embedded so far. Tests the format only; the score will be poor.
meta = metas.set_index('file_id')
Xtr, itr = load_cache('train'); ytr = meta.loc[itr, 'label'].to_numpy()
print(Xtr.shape, pd.Series(ytr).value_counts().to_dict())    # which birds exist so far
clf = make_pipeline(StandardScaler(),
                    LogisticRegression(C=0.5, class_weight='balanced', max_iter=2000, random_state=42)).fit(Xtr, ytr)
Xte, ite = load_cache('test')
P = pd.DataFrame(clf.predict_proba(Xte), columns=clf.classes_).assign(file_id=ite).groupby('file_id').mean()
write_submission(P.idxmax(axis=1))                           # asserts fail loudly if the format is wrong

# %%
meta = metas.set_index('file_id')
X, ids = load_cache('train')                                   # reads the per-file cache from Cell 1
y = meta.loc[ids, 'label'].to_numpy(); g = meta.loc[ids, 'source_group'].to_numpy()
classes = np.unique(y)
print('files embedded:', len(set(ids)), '| source groups per class:', pd.Series(g).groupby(y).nunique().to_dict())

def cv_score(C, cw):
    sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    oof = np.zeros((len(y), len(classes))); folds = []
    for tr, va in sgkf.split(X, y, groups=g):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C, class_weight=cw, max_iter=2000, random_state=42)).fit(X[tr], y[tr])
        assert len(clf.classes_) == len(classes), 'a train fold lost a class'
        oof[va] = clf.predict_proba(X[va])                     # predicted by a model that never saw these groups
        fp = pd.DataFrame(oof[va], columns=classes).assign(file_id=ids[va]).groupby('file_id').mean()
        folds.append(f1_score(meta.loc[fp.index, 'label'], fp.idxmax(axis=1), average='macro', zero_division=0))
    fp = pd.DataFrame(oof, columns=classes).assign(file_id=ids).groupby('file_id').mean()
    pooled = f1_score(meta.loc[fp.index, 'label'], fp.idxmax(axis=1), average='macro', zero_division=0)
    return round(pooled, 3), np.round(folds, 2)

for C, cw in itertools.product([0.01, 0.1, 0.5, 2], ['balanced', None]):   # 8 combos, chosen by hand
    print(C, cw, *cv_score(C, cw))    

# %%
# Random forest
from sklearn.ensemble import RandomForestClassifier
clf = make_pipeline(StandardScaler(),
                    RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)).fit(Xtr, ytr)
Xte, ite = load_cache('test')
P = pd.DataFrame(clf.predict_proba(Xte), columns=clf.classes_).assign(file_id=ite).groupby('file_id').mean()
write_submission(P.idxmax(axis=1), out='submissionRF.csv')                           # asserts fail loudly if the format is wrong

# %% [markdown]
# Catatan: dua submisi pertama (submission.csv dan submissionRF.csv) dilakukan dengan kode diatas, namun dengan jumlah data 1044 dengan cache embedding .npz yang hilang

# %% [markdown]
# ## 4. Simulasi Unknown: Leave-One-Species-Out
# 
# Karena tidak ada contoh latih untuk kelas Unknown, kita memakai proxy:
# 
# - satu spesies dikenal disembunyikan,
# - dua spesies lain dipakai sebagai known,
# - spesies yang disembunyikan dianggap Unknown.
# 
# Sinyal yang dibandingkan:
# - `max-prob`: probabilitas maksimum classifier,
# - `knn-dist`: jarak ke tetangga terdekat di train embeddings.
# 
# Metrik:
# - AUROC known vs hidden,
# - catch rate ketika 10% known files ditolak.

# %%
def scores(Xtr, ytr, Xva, C=0.01, k=5):
    """Familiarity of each val window given a train set. Higher = more familiar."""
    sc = StandardScaler().fit(Xtr); A, B = sc.transform(Xtr), sc.transform(Xva)
    lr = LogisticRegression(C=C, class_weight='balanced', max_iter=2000, random_state=42).fit(A, ytr)
    conf = lr.predict_proba(B).max(axis=1)                            # signal 1: how sure the classifier is
    dist = NearestNeighbors(n_neighbors=k, metric='cosine').fit(A).kneighbors(B)[0].mean(axis=1)
    return conf, -dist                                                # signal 2: distance to k closest train windows (negated)

def to_file(s, ids): return pd.Series(np.asarray(s)).groupby(np.asarray(ids)).mean()   # mean score per file

rows = []
for held in classes:
    kn = y != held                                                    # known birds = the other two
    Xk, yk, gk, ik = X[kn], y[kn], g[kn], ids[kn]
    kc, kd = np.zeros(len(yk)), np.zeros(len(yk))
    for tr, va in StratifiedGroupKFold(4, shuffle=True, random_state=42).split(Xk, yk, groups=gk):
        kc[va], kd[va] = scores(Xk[tr], yk[tr], Xk[va])               # known files: scored by models that never saw their groups
    uc, ud = scores(Xk, yk, X[~kn])                                   # held-out bird plays Unknown
    for name, ks, us in [('max-prob', kc, uc), ('knn-dist', kd, ud)]:
        K, U = to_file(ks, ik), to_file(us, ids[~kn])
        auc = roc_auc_score([1] * len(K) + [0] * len(U), np.r_[K.values, U.values])
        caught = (U < np.quantile(K, 0.10)).mean()                    # Unknown caught if we accept losing 10% of known files
        rows.append((held, name, round(auc, 3), round(caught, 2)))
print(pd.DataFrame(rows, columns=['held out (plays Unknown)', 'signal', 'AUROC', 'caught @ 10% known lost']))

# %%
# Cell 2: what each signal says about the real test set (no labels used; automated, so it is rule-safe)
kc, kd = np.zeros(len(y)), np.zeros(len(y))
for tr, va in StratifiedGroupKFold(4, shuffle=True, random_state=42).split(X, y, groups=g):
    kc[va], kd[va] = scores(X[tr], y[tr], X[va])
Xte, ite = load_cache('test')
tc, td = scores(X, y, Xte)
for name, ks, ts in [('max-prob', kc, tc), ('knn-dist', kd, td)]:
    K, T = to_file(ks, ids), to_file(ts, ite)
    for q in [0.05, 0.10, 0.20]:
        print(name, f'known lost {q:.0%} -> test rejected', round((T < np.quantile(K, q)).mean(), 2))

# %% [markdown]
# ## 5. File-level Pooling & SVC RBF
# 
# Submission kompetisi bersifat per file, bukan per window. Karena itu kita juga menguji model file-level:
# 
# - setiap file direpresentasikan oleh rata-rata embedding windows-nya,
# - model dilatih pada level file,
# - evaluasi memakai StratifiedGroupKFold berbasis source_group.
# 
# Model yang dibandingkan:
# - Logistic Regression,
# - SVC RBF dengan calibration.

# %%
# Cell 1: file-level table (1 row per file = mean of its windows), then a small SVC grid with logistic regression as a control

Xf = pd.DataFrame(X).groupby(ids).mean()
fid, Xf = Xf.index.to_numpy(), Xf.to_numpy()
yf = meta.loc[fid, 'label'].to_numpy(); gf = meta.loc[fid, 'source_group'].to_numpy()
print(Xf.shape, pd.Series(yf).value_counts().to_dict())          # expect 1826 files

def make_model(kind, C, gamma=None):
    if kind == 'svc':
        base = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=C, gamma=gamma, class_weight='balanced'))
        return CalibratedClassifierCV(base, method='sigmoid', cv=3)
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, class_weight='balanced', max_iter=2000, random_state=SEED))

def cv_file(kind, C, gamma=None):
    oof = np.zeros((len(yf), len(classes)))
    for tr, va in StratifiedGroupKFold(4, shuffle=True, random_state=SEED).split(Xf, yf, groups=gf):
        oof[va] = make_model(kind, C, gamma).fit(Xf[tr], yf[tr]).predict_proba(Xf[va])   # each file predicted by a model that never saw its group
    return f1_score(yf, classes[oof.argmax(axis=1)], average='macro'), oof

grid = [('lr', 0.01, None), ('lr', 0.1, None)] + [('svc', C, gm) for C in [1, 4] for gm in [0.0003, 0.001, 0.003]]
oofs, res = {}, []
for cfg in grid:
    s, oofs[cfg] = cv_file(*cfg); res.append((cfg, round(s, 3))); print(cfg, round(s, 3))

best_svc = max([r for r in res if r[0][0] == 'svc'], key=lambda r: r[1])[0]
best_lr = max([r for r in res if r[0][0] == 'lr'], key=lambda r: r[1])[0]
for cfg in [best_lr, best_svc]:                                   # per-bird scores, which we haven't looked at yet
    print(cfg); print(classification_report(yf, classes[oofs[cfg].argmax(axis=1)], digits=3))

# %% [markdown]
# Catatan: kalibrasi probabilitas SVC (CalibratedClassifierCV, cv=3) tidak mengelompokkan source_group di lipatan dalamnya, sehingga probabilitasnya cenderung terlalu yakin untuk sumber baru. Skor CV luar tidak terpengaruh. Ambang Unknown berasal dari kuantil keyakinan out-of-fold, bukan nilai absolut, sehingga pengaruhnya diperkirakan kecil. SVC bukan model final.

# %%
# Cell 2: fake-Unknown test (hide one bird) for the control and the best SVC
def sim_unknown(kind, C, gamma=None):
    rows = []
    for held in classes:
        kn = yf != held                                           # the hidden bird plays Unknown
        Xk, yk, gk = Xf[kn], yf[kn], gf[kn]
        known = np.zeros(len(yk))
        for tr, va in StratifiedGroupKFold(4, shuffle=True, random_state=SEED).split(Xk, yk, groups=gk):
            known[va] = make_model(kind, C, gamma).fit(Xk[tr], yk[tr]).predict_proba(Xk[va]).max(axis=1)
        unk = make_model(kind, C, gamma).fit(Xk, yk).predict_proba(Xf[~kn]).max(axis=1)
        auc = roc_auc_score([1] * len(known) + [0] * len(unk), np.r_[known, unk])
        rows.append((held, round(auc, 3), round((unk < np.quantile(known, 0.10)).mean(), 2)))
    return pd.DataFrame(rows, columns=['hidden bird', 'AUROC', 'caught @ 10% known lost'])

for cfg in [best_lr, best_svc]:
    print(cfg); print(sim_unknown(*cfg))

# %% [markdown]
# ## 6. Validasi LOGO
# 
# LOGO = Leave One Group Out.
# 
# Setiap source_group pernah menjadi validation set, sementara model dilatih pada semua source_group lain.
# 
# Tujuan:
# - menguji apakah model hanya menghafal karakter recording/source,
# - melihat robustness terhadap distribusi audio yang berbeda,
# - memeriksa error per source_group.

# %%
# Existing notebook variables:
# Xf: one averaged embedding vector per recording
# yf: species label per recording
# gf: source_group per recording

SEED = 42
classes_file = np.unique(yf)

models = {
    "LR_C001_balanced": lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.01,
            class_weight="balanced",
            max_iter=2000,
            random_state=SEED,
        ),
    ),
    "LR_C01_unweighted": lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            class_weight=None,
            max_iter=2000,
            random_state=SEED,
        ),
    ),
    "SVC_RBF_C1_g001": lambda: make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1,
            gamma=0.001,
            class_weight="balanced",
        ),
    ),
}

# Each specification creates a fresh, deterministic splitter.
splitters = {
    "LOGO": lambda: LeaveOneGroupOut(),
    "SGKF_seed42": lambda: StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=42
    ),
    "SGKF_seed7": lambda: StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=7
    ),
    "SGKF_seed2026": lambda: StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=2026
    ),
}

results = []
logo_predictions = {}

for model_name, build in models.items():
    for split_name, make_splitter in splitters.items():
        splitter = make_splitter()
        oof_pred = np.empty(len(yf), dtype=object)

        for train_idx, val_idx in splitter.split(
            Xf, yf, groups=gf
        ):
            model = build()
            model.fit(Xf[train_idx], yf[train_idx])
            oof_pred[val_idx] = model.predict(Xf[val_idx])

        score = f1_score(
            yf,
            oof_pred,
            labels=classes_file,
            average="macro",
            zero_division=0,
        )

        results.append({
            "model": model_name,
            "validation": split_name,
            "macro_f1_3_known_classes": score,
        })

        if split_name == "LOGO":
            logo_predictions[model_name] = oof_pred.copy()

summary = pd.DataFrame(results)

display(
    summary.pivot(
        index="model",
        columns="validation",
        values="macro_f1_3_known_classes",
    ).round(4)
)

for model_name, predictions in logo_predictions.items():
    print(f"\n{'=' * 55}\n{model_name}\n")
    print(classification_report(
        yf,
        predictions,
        labels=classes_file,
        zero_division=0,
        digits=3,
    ))

# Diagnose source sensitivity using the LOGO predictions.
for model_name, predictions in logo_predictions.items():
    diagnostic = pd.DataFrame({
        "source_group": gf,
        "true_label": yf,
        "correct": predictions == yf,
    })

    print(f"\nPer-source accuracy: {model_name}")
    display(
        diagnostic.groupby(
            ["source_group", "true_label"]
        ).agg(
            recordings=("correct", "size"),
            accuracy=("correct", "mean"),
        ).round(3)
    )


# %% [markdown]
# ## 7. Threshold Unknown & Submission
# 
# Kelas Unknown tidak punya contoh latih, sehingga digunakan aturan berbasis familiarity / confidence.
# 
# Beberapa pendekatan:
# - quantile threshold dari known OOF confidence,
# - percentile threshold pada test confidence,
# - target share Unknown sekitar 38%.
# 
# Catatan penting:
# Angka 0.38 berasal dari skor public submission all_unknown.
# 
# Jika semua file diprediksi Unknown, macro-F1 public = 0.138.
# Maka:
# - F1_Unknown = 4 * 0.138 = 0.552
# - precision Unknown = p
# - F1 = 2p / (1+p)
# 
# Sehingga:
# - p ≈ 0.38
# 
# Jadi 0.38 adalah estimasi proporsi Unknown di public test, bukan angka acak.

# %% [markdown]
# ### 7.1 Known quantile threshold awal, window-level LR

# %%
# Cell A: the model, and the familiarity threshold from grouped CV (known birds only)
def fit_lr(Xtr, ytr):
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=0.01, class_weight='balanced', max_iter=2000, random_state=SEED).fit(sc.transform(Xtr), ytr)
    return lambda Z: lr.predict_proba(sc.transform(Z))     # window probabilities, columns in sorted class order

kc = np.zeros(len(y))
for tr, va in StratifiedGroupKFold(4, shuffle=True, random_state=SEED).split(X, y, groups=g):
    kc[va] = fit_lr(X[tr], y[tr])(X[va]).max(axis=1)        # top probability per window, from models that never saw these groups
K = to_file(kc, ids)                                        # a known file's familiarity = mean top probability over its windows

# %%
# Cell B: fit on all train, predict test, call low-familiarity files Unknown, write one CSV per Q
predict = fit_lr(X, y)
prob = predict(Xte)
P = pd.DataFrame(prob, columns=classes).assign(file_id=ite).groupby('file_id').mean()   # file-level class probabilities
fam = to_file(prob.max(axis=1), ite).loc[P.index]           # file-level familiarity, same recipe as K

for Q in [0.10, 0.20, 0.30]:                                # Q = share of known files we accept rejecting wrongly
    pred = P.idxmax(axis=1)
    pred.loc[(fam < np.quantile(K, Q)).to_numpy()] = 'Unknown'
    print('Q', Q, 'threshold', round(np.quantile(K, Q), 3))
    write_submission(pred, f'submission_q{int(Q * 100)}.csv')

# %% [markdown]
# ### 7.2 Proxy Unknown file-level untuk kalibrasi threshold

# %%
def evaluate_unknown_proxy(
    X_file,
    y_file,
    groups,
    seed=42,
    rejection_quantiles=(0.05, 0.10, 0.20),
):
    """
    Leave one known species out as a proxy for Unknown.

    For each fold, the SAME model scores:
      1. Known validation recordings from held-out source groups.
      2. Recordings from the withheld species.

    This is an open-set proxy experiment, not an estimate
    of the competition's exact four-class macro F1.
    """
    all_labels = np.unique(y_file)
    rows = []

    for held_species in all_labels:
        known_mask = y_file != held_species
        unknown_mask = ~known_mask

        X_known = X_file[known_mask]
        y_known = y_file[known_mask]
        g_known = groups[known_mask]
        X_unknown = X_file[unknown_mask]

        cv = StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=seed,
        )

        for fold, (tr, va) in enumerate(
            cv.split(X_known, y_known, groups=g_known),
            start=1,
        ):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.01,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                ),
            )
            model.fit(X_known[tr], y_known[tr])

            # Both sets are evaluated by this exact fitted model.
            known_prob = model.predict_proba(X_known[va])
            unknown_prob = model.predict_proba(X_unknown)

            known_conf = known_prob.max(axis=1)
            unknown_conf = unknown_prob.max(axis=1)

            known_pred = model.predict(X_known[va])
            unknown_pred = model.predict(X_unknown)

            proxy_labels = list(model.classes_) + ["Unknown"]
            true_labels = np.concatenate([
                y_known[va],
                np.full(len(X_unknown), "Unknown"),
            ])

            for q in rejection_quantiles:
                threshold = np.quantile(known_conf, q)

                pred_known = np.where(
                    known_conf < threshold,
                    "Unknown",
                    known_pred,
                )
                pred_unknown = np.where(
                    unknown_conf < threshold,
                    "Unknown",
                    unknown_pred,
                )

                predictions = np.concatenate([
                    pred_known,
                    pred_unknown,
                ])

                rows.append({
                    "withheld_species": held_species,
                    "fold": fold,
                    "quantile": q,
                    "threshold": threshold,
                    "known_rejection_rate": np.mean(
                        known_conf < threshold
                    ),
                    "unknown_catch_rate": np.mean(
                        unknown_conf < threshold
                    ),
                    "proxy_macro_f1": f1_score(
                        true_labels,
                        predictions,
                        labels=proxy_labels,
                        average="macro",
                        zero_division=0,
                    ),
                })

    return pd.DataFrame(rows)


unknown_results = evaluate_unknown_proxy(Xf, yf, gf)

display(
    unknown_results.groupby(
        ["withheld_species", "quantile"]
    ).agg(
        proxy_macro_f1=("proxy_macro_f1", "mean"),
        unknown_catch_rate=("unknown_catch_rate", "mean"),
        known_rejection_rate=("known_rejection_rate", "mean"),
    ).round(3)
)


# %% [markdown]
# ### 7.3 Submission SVC file-level dan all_unknown

# %%
# Cell 3: final SVC at file level -> two files for your 2 free submissions (about 1 min)
cfg = best_svc                                          # ('svc', 1, 0.001)
Xte, ite = load_cache('test')
Xt = pd.DataFrame(Xte).groupby(ite).mean()              # one row per test file = mean of its windows
fit = make_model(*cfg).fit(Xf, yf)                      # trained on all 1826 train files
Pt = pd.DataFrame(fit.predict_proba(Xt.to_numpy()), columns=classes, index=Xt.index)
K = oofs[cfg].max(axis=1)                               # known-file familiarity from the grouped CV
pred = Pt.idxmax(axis=1)
pred.loc[Pt.max(axis=1) < np.quantile(K, 0.20)] = 'Unknown'    # Q=20%, the better of your two earlier probes
write_submission(pred, 'submission_svc_q20.csv')
write_submission(pd.Series('Unknown', index=Xt.index), 'submission_all_unknown.csv')

# %% [markdown]
# ### 7.4 Refinement dengan fixed Unknown share 0.38

# %%
# Sel 1: dua fungsi. Satu menilai tiap file, satu menjalankan simulasi "spesies disembunyikan = Unknown"
lab = meta['label']                                          # label per file (meta dari sel kita sebelumnya)

def score_files(model, X, ids):
    """Per file: predicted class (from mean probabilities) and confidence (mean of per-window max probability)."""
    P = model.predict_proba(X)
    proba = pd.DataFrame(P, columns=model.classes_).groupby(ids).mean()
    conf = pd.Series(P.max(axis=1)).groupby(ids).mean().loc[proba.index]
    return proba.idxmax(axis=1), conf

def proxy_unknown(X, y, g, ids, build, share=0.30, qs=(0.10, 0.20, 0.25, 0.30, 0.40), seeds=(0, 1, 2)):
    rows = []
    for held in np.unique(y):                                # spesies yang disembunyikan berperan sebagai Unknown
        kn = y != held
        Xk, yk, gk, ik = X[kn], y[kn], g[kn], ids[kn]
        for seed in seeds:
            rng = np.random.default_rng(seed)
            for tr, va in StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(Xk, yk, groups=gk):
                m = build().fit(Xk[tr], yk[tr])
                pk, fk = score_files(m, Xk[va], ik[va])       # file dikenal dari kelompok sumber yang ditahan
                pu, fu = score_files(m, X[~kn], ids[~kn])     # file spesies yang disembunyikan
                thr = {q: np.quantile(fk, q) for q in qs}    # ambang: q% file dikenal yang paling ragu
                # samakan porsi Unknown = share (kurangi sisi yang lebih banyak)
                if len(fu) / (len(fu) + len(fk)) > share:
                    keep_k, keep_u = fk.index, rng.choice(fu.index, round(len(fk) * share / (1 - share)), replace=False)
                else:
                    keep_k, keep_u = rng.choice(fk.index, round(len(fu) * (1 - share) / share), replace=False), fu.index
                true = np.r_[lab.loc[keep_k].to_numpy(), ['Unknown'] * len(keep_u)]
                labs = list(m.classes_) + ['Unknown']
                for q in qs:
                    tk, tu = fk.loc[keep_k] < thr[q], fu.loc[keep_u] < thr[q]      # True = ditolak jadi Unknown
                    pred = np.r_[np.where(tk, 'Unknown', pk.loc[keep_k]), np.where(tu, 'Unknown', pu.loc[keep_u])]
                    rows.append(dict(hidden=held, q=q, seed=seed,
                                     share=len(keep_u) / (len(keep_u) + len(keep_k)),
                                     f1=f1_score(true, pred, labels=labs, average='macro', zero_division=0),
                                     f1_unknown=f1_score(true, pred, labels=['Unknown'], average='macro', zero_division=0),
                                     catch=tu.mean(), known_lost=tk.mean()))
    return pd.DataFrame(rows)

# %%
# Sel 2, hanya satu angka yang berubah: porsi Unknown 0.38 (tadinya 0.30)
res = proxy_unknown(X, y, g, ids, lambda: make_model('lr', 0.01), share=0.38)
print(res.groupby('hidden')['share'].mean().round(2).to_dict())          # harus sekitar 0.38
print(res.groupby(['hidden', 'q'])[['f1', 'f1_unknown', 'catch', 'known_lost']].mean().round(3).to_string())
print(res.groupby('q').agg(f1=('f1', 'mean'), sd=('f1', 'std'), f1_unknown=('f1_unknown', 'mean'),
                           catch=('catch', 'mean'), known_lost=('known_lost', 'mean')).round(3).to_string())

# %%
# File baru: 38% file tes paling tidak yakin jadi Unknown (ambang dari peringkat tes, bukan dari CV)
prob = fit_lr(X, y)(Xte)                                              # LogReg per jendela, C=0.01, balanced
P = pd.DataFrame(prob, columns=classes).assign(file_id=ite).groupby('file_id').mean()
fam = to_file(prob.max(axis=1), ite).loc[P.index]                     # keyakinan per file
pred = P.idxmax(axis=1)
pred.loc[fam < np.quantile(fam, 0.38)] = 'Unknown'
write_submission(pred, 'submission_lr_rank38.csv')

# %%
# Uji kepekaan (perkiraan 5-15 menit): kalau porsi Unknown sebenarnya 0.32 atau 0.44 (bukan 0.38),
# seberapa berubah skor kebijakan "tolak f% file yang paling tidak yakin"?
def proxy_rank(X, y, g, ids, build, share, fs=(0.30, 0.34, 0.38, 0.42, 0.46), seeds=(0, 1)):
    rows = []
    for held in np.unique(y):
        kn = y != held
        Xk, yk, gk, ik = X[kn], y[kn], g[kn], ids[kn]
        for seed in seeds:
            rng = np.random.default_rng(seed)
            for tr, va in StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(Xk, yk, groups=gk):
                m = build().fit(Xk[tr], yk[tr])
                pk, fk = score_files(m, Xk[va], ik[va]); pu, fu = score_files(m, X[~kn], ids[~kn])
                if len(fu) / (len(fu) + len(fk)) > share:
                    keep_k, keep_u = fk.index, rng.choice(fu.index, round(len(fk) * share / (1 - share)), replace=False)
                else:
                    keep_k, keep_u = rng.choice(fk.index, round(len(fu) * (1 - share) / share), replace=False), fu.index
                pooled = np.r_[fk.loc[keep_k], fu.loc[keep_u]]              # keyakinan semua file yang dinilai
                true = np.r_[lab.loc[keep_k].to_numpy(), ['Unknown'] * len(keep_u)]
                labs = list(m.classes_) + ['Unknown']
                for f in fs:
                    thr = np.quantile(pooled, f)                             # tolak f% file yang paling tidak yakin
                    tk, tu = fk.loc[keep_k] < thr, fu.loc[keep_u] < thr
                    pred = np.r_[np.where(tk, 'Unknown', pk.loc[keep_k]), np.where(tu, 'Unknown', pu.loc[keep_u])]
                    rows.append(dict(f=f, share=share, f1=f1_score(true, pred, labels=labs, average='macro', zero_division=0)))
    return pd.DataFrame(rows)

out = pd.concat([proxy_rank(X, y, g, ids, lambda: make_model('lr', 0.01), s) for s in (0.32, 0.38, 0.44)])
print(out.pivot_table(index='f', columns='share', values='f1', aggfunc='mean').round(3).to_string())

# %% [markdown]
# ## 8. Diagnosa Submission
# 
# Section ini membandingkan perilaku beberapa submission:
# 
# - LogReg q20,
# - SVC q20.
# 
# Analisis yang dilakukan:
# - distribusi label,
# - crosstab antar submission,
# - melihat berapa banyak file yang dipindahkan menjadi Unknown oleh satu model dibanding model lain.

# %%
# Cell A (1 min): what did the SVC file do differently from the logistic-regression file?
a = pd.read_csv('submission/submission_q20.csv').set_index('file_id')['label']        # LogReg, window-level, q20 (scored 0.555)
b = pd.read_csv('submission/submission_svc_q20.csv').set_index('file_id')['label']    # SVC, file-level, q20 (scored 0.507)
print(a.value_counts().to_dict()); print(b.value_counts().to_dict())
pd.crosstab(a, b.loc[a.index], rownames=['LogReg q20'], colnames=['SVC q20'])

# %% [markdown]
# Skor F1 public yang tercatat:
# - LogReg q10: 0.540
# - LogReg q20: 0.555
# - LogReg rank38: 0.531
# - SVC q20: 0.507
# - all Unknown: 0.138
# 
# Implikasi:
# - kelas Unknown signifikan,
# - threshold terbaik tidak stabil,
# - sinyal Unknown masih perlu diperbaiki.

# %% [markdown]
# ## 9.  Mulai gila dengan birdnet v3.0

# %%
# Pengganti seluruh sel ekstraksi V3.0 (V3-1 ... D2): satu fungsi, batch, dan pengisi sudah dibuang di dalamnya
def extract_batch(items, split, root, encoder, cfg, chunk=64):
    """Batch-encode files not yet cached; keep only ceil(duration / 3 s) windows per file (padding rows are junk)."""
    d = f'{root}/{split}'; os.makedirs(d, exist_ok=True)
    todo = [(f, p) for f, p in items if not os.path.exists(f'{d}/{f}.npy')]
    for i in tqdm(range(0, len(todo), chunk)):
        part = todo[i:i + chunk]
        res = encoder.encode([p for _, p in part], **cfg)
        pos = {path_key(a): j for j, a in enumerate(res.inputs)}              # hasil bisa datang dengan urutan lain: petakan lewat path
        for f, p in part:
            e = np.asarray(res.embeddings[pos[path_key(p)]], dtype=np.float32)
            e = e[:math.ceil(sf.info(p).duration / 3.0 - 1e-6)]
            assert np.isfinite(e).all(), f'NaN atau tak terhingga pada {f}'
            np.save(f'{d}/{f}.tmp.npy', e); os.replace(f'{d}/{f}.tmp.npy', f'{d}/{f}.npy')
    print(split, 'cached files:', len(os.listdir(d)), '| new:', len(todo))

birdnet_30 = birdnet.load("acoustic", "3.0", "onnx")
extract_batch(test_items, 'test', CACHE3, birdnet_30, CFG)
extract_batch(train_items, 'train', CACHE3, birdnet_30, CFG)

# %%
# Sel E1: loadv3 cache V3.0 (per jendela) dan samakan dengan label
def load_cache(split, root='features/cache'):
    """Window-level embeddings from the per-file cache: (X, file_ids)."""
    d = f'{root}/{split}'
    fids = sorted(f[:-4] for f in os.listdir(d) if f.endswith('.npy') and not f.endswith('.tmp.npy'))
    X, ids = [], []
    for f in fids: e = np.load(f'{d}/{f}.npy'); X.append(e); ids += [f] * len(e)
    return np.vstack(X), np.array(ids)

X3, ids3 = load_cache('train', CACHE3); Xte3, ite3 = load_cache('test', CACHE3)
y3 = meta.loc[ids3, 'label'].to_numpy(); g3 = meta.loc[ids3, 'source_group'].to_numpy()
print(X3.shape, Xte3.shape, pd.Series(y3).value_counts().to_dict())

# %%
# Sel E2 (sekitar 10 menit, perkiraan): CV 3 burung dan simulasi Unknown (porsi 0.38) untuk V3.0, model dan seed sama seperti 2.4
def cv_file_f1(X, y, g, ids, build, seed=42):
    """Grouped 4-fold CV, windows -> files, macro-F1 over the 3 known birds."""
    pred = []
    for tr, va in StratifiedGroupKFold(4, shuffle=True, random_state=seed).split(X, y, groups=g):
        p, _ = score_files(build().fit(X[tr], y[tr]), X[va], ids[va]); pred.append(p)
    pred = pd.concat(pred)
    return f1_score(lab.loc[pred.index], pred, average='macro'), pred

build = lambda: make_model('lr', 0.01)                      # LogReg per jendela, C=0.01, balanced
s3, _ = cv_file_f1(X3, y3, g3, ids3, build); print('V3.0 CV 3 burung:', round(s3, 3))
res3 = proxy_unknown(X3, y3, g3, ids3, build, share=0.38)
print(res3.groupby('q').agg(f1=('f1', 'mean'), sd=('f1', 'std'), f1_unknown=('f1_unknown', 'mean'),
                            catch=('catch', 'mean'), known_lost=('known_lost', 'mean')).round(3).to_string())
print(res3.groupby(['hidden', 'q'])[['f1', 'f1_unknown', 'catch']].mean().round(3).to_string())

# %%
# Pemeriksaan kesiapan (jalankan setelah Restart & Run All): hal-hal yang diandalkan seluruh pipeline
X3, ids3 = load_cache('train', CACHE3); Xte3, ite3 = load_cache('test', CACHE3)
sub_path = glob.glob(f'{path}/**/sample_submission.csv', recursive=True)[0]
assert set(np.unique(ite3)) == set(pd.read_csv(sub_path)['file_id']), 'file tes harus sama dengan sample_submission'
assert set(np.unique(ids3)) == set(meta.index), 'tiap file train harus punya jendela dan label'
assert X3.shape[1] == 1280 and Xte3.shape[1] == 1280 and np.isfinite(X3).all() and np.isfinite(Xte3).all()
print('OK:', len(np.unique(ids3)), 'file train,', len(np.unique(ite3)), 'file tes,', X3.shape[0], 'jendela train')

# %%



