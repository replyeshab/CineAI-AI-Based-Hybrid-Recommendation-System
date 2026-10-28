"""CineAI v2: retrieval -> LightGBM ranking -> exact TreeSHAP explanations.
Colab-free-tier friendly: float32 everywhere, user states are folded in on the fly
(so cold-start users use exactly the same code path as known users), no giant
user x item dense matrices, no shap dependency at serving time."""
import re, warnings
import numpy as np, pandas as pd, scipy.sparse as sp, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize

CFG = dict(
    als_factors=64, als_reg=0.05, als_iters=15,
    content_dim=64, min_ratings_item=5,
    n_als=100, n_content=30, n_pop=20,                 # candidate pool per user ~150
    k_choices=[1, 2, 3, 5, 8, 12, 20, 40, 0],          # simulated history sizes (0 = full)
    rounds=500, lr=0.05, leaves=63,
)
FEATS = ["u_cnt", "u_mean", "u_std", "u_pos", "u_year",
         "i_cnt", "i_mean", "i_pos", "i_year", "year_gap",
         "als", "als_z", "cont", "cont_z", "gaff", "gshare", "base_pred", "pop_w"]
GROUPS = {
    "Your genre taste": ["gaff", "gshare"],
    "Similar to what you liked": ["cont", "cont_z"],
    "People with similar taste": ["als", "als_z"],
    "Community rating": ["i_mean", "i_pos", "i_cnt", "base_pred", "pop_w"],
    "Era & profile fit": ["year_gap", "i_year", "u_year", "u_cnt", "u_mean", "u_std", "u_pos"],
}

# ----------------------------------------------------------------- data helpers
def load_ratings(path):
    return pd.read_csv(path, usecols=["userId", "movieId", "rating", "timestamp"],
                       dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int32"})

def prepare_movies(df):
    m = df.copy()
    if "clean_title" not in m:
        m["year"] = pd.to_numeric(m["title"].str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce")
        m["clean_title"] = m["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True).str.strip()
    if "year" not in m:
        m["year"] = np.nan
    m["year"] = pd.to_numeric(m["year"], errors="coerce")
    m["genres"] = m["genres"].fillna("")
    return m[["movieId", "clean_title", "genres", "year"]].drop_duplicates("movieId").reset_index(drop=True)

def temporal_split(r, val_frac=0.10, test_frac=0.15):
    """Per-user chronological split: oldest -> train, middle -> val, newest -> test."""
    r = r.sort_values(["userId", "timestamp"], kind="stable").reset_index(drop=True)
    n = r.groupby("userId")["userId"].transform("size").to_numpy()
    f = (r.groupby("userId").cumcount().to_numpy() + 1) / n
    te = f > 1 - test_frac
    va = (f > 1 - test_frac - val_frac) & ~te
    return r[~(te | va)], r[va], r[te]

def to_csr(df, uidx, midx):
    u = uidx.get_indexer(df["userId"].to_numpy()); i = midx.get_indexer(df["movieId"].to_numpy())
    ok = (u >= 0) & (i >= 0)
    return sp.csr_matrix((df["rating"].to_numpy(np.float32)[ok], (u[ok], i[ok])), shape=(len(uidx), len(midx)))

def row_counts(csr): return np.diff(csr.indptr)

# ----------------------------------------------------------------- ALS
def _als_numpy(C, d, reg, iters, seed=0):
    """Exact implicit ALS in numpy. Only for tiny/test data; install `implicit` for real runs."""
    rng = np.random.default_rng(seed)
    V = rng.normal(0, .1, (C.shape[1], d)).astype(np.float32)
    def step(M, Y):
        YtY = Y.T @ Y + reg * np.eye(d, dtype=np.float32); X = np.zeros((M.shape[0], d), np.float32)
        for u in range(M.shape[0]):
            s, e = M.indptr[u], M.indptr[u + 1]
            if s == e: continue
            Ys, c = Y[M.indices[s:e]], M.data[s:e]
            X[u] = np.linalg.solve(YtY + (Ys.T * (c - 1)) @ Ys, (Ys.T * c).sum(1))
        return X
    Ct = C.T.tocsr()
    for _ in range(iters):
        U = step(C, V); V = step(Ct, U)
    return V

def fit_als(conf, d, reg, iters):
    try:
        from implicit.als import AlternatingLeastSquares
    except ImportError:
        warnings.warn("`implicit` missing -> slow numpy ALS (small data only)")
        return _als_numpy(conf, d, reg, iters)
    m = AlternatingLeastSquares(factors=d, regularization=reg, iterations=iters, random_state=42)
    m.fit(conf, show_progress=True)                       # implicit>=0.5: fit(user_items)
    V = m.item_factors
    return np.asarray(V.to_numpy() if hasattr(V, "to_numpy") else V, dtype=np.float32)

# ----------------------------------------------------------------- the model
class Recsys:
    def __init__(self, movies, cfg=None):
        self.cfg = {**CFG, **(cfg or {})}
        self.movies = prepare_movies(movies)
        self.midx = pd.Index(self.movies["movieId"].to_numpy())
        self.n_items = len(self.movies)
        self.year = self.movies["year"].to_numpy(np.float32)
        gl = sorted({g for s in self.movies["genres"] for g in s.split("|") if g and g != "(no genres listed)"})
        self.genre_names = gl
        gi = {g: j for j, g in enumerate(gl)}
        self.G = np.zeros((self.n_items, len(gl)), np.float32)
        for r, s in enumerate(self.movies["genres"]):
            for g in s.split("|"):
                if g in gi: self.G[r, gi[g]] = 1
        self.ranker = self.liker = None
        self.demo = {}

    # ---- stage 0: everything that is learned from TRAIN ratings only
    def fit_base(self, train, item_text=None):
        c = self.cfg
        n_i = train.getnnz(axis=0); self.i_cnt = n_i.astype(np.float32)
        s = np.asarray(train.sum(0)).ravel(); ss = np.asarray(train.multiply(train).sum(0)).ravel()
        B = train.copy(); B.data = (B.data >= 4).astype(np.float32); p = np.asarray(B.sum(0)).ravel()
        self.gmean = float(train.data.mean()); gpos = float((train.data >= 4).mean())
        k = 10.0
        self.i_raw_mean = np.where(n_i > 0, s / np.maximum(n_i, 1), self.gmean).astype(np.float32)
        self.i_mean = ((s + k * self.gmean) / (n_i + k)).astype(np.float32)
        self.i_pos = ((p + k * gpos) / (n_i + k)).astype(np.float32)
        var = np.where(n_i > 1, ss / np.maximum(n_i, 1) - (s / np.maximum(n_i, 1)) ** 2, 0)
        self.i_std = np.sqrt(np.clip(var, 0, None)).astype(np.float32)
        m = np.quantile(n_i[n_i > 0], 0.9)                  # IMDB weighted rating
        self.pop_w = (n_i / (n_i + m) * self.i_raw_mean + m / (n_i + m) * self.gmean).astype(np.float32)
        self.min_votes = float(np.quantile(n_i[n_i > 0], 0.8))
        ok = np.flatnonzero(n_i >= self.min_votes)
        self.pop_order = ok[np.argsort(-self.pop_w[ok])][:500]
        # content embedding (genres + title [+ tags]) -> TF-IDF -> SVD
        gtok = self.movies["genres"].map(lambda s: " ".join("g_" + re.sub(r"[-\s]", "", g).lower() for g in s.split("|") if g) )
        text = (gtok + " " + gtok + " " + self.movies["clean_title"].fillna("")).tolist()
        if item_text is not None:
            extra = self.movies["movieId"].map(item_text).fillna("")
            text = [a + " " + b for a, b in zip(text, extra)]
        self.tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2, stop_words="english",
                                     sublinear_tf=True, dtype=np.float32)
        X = self.tfidf.fit_transform(text)
        self.svd = TruncatedSVD(min(c["content_dim"], X.shape[1] - 1), random_state=42)
        self.E = normalize(self.svd.fit_transform(X)).astype(np.float32)
        # collaborative item factors (users are folded in later, nothing per-user is stored)
        conf = train.copy(); conf.data = (1 + np.clip(conf.data - 2, 0, None)).astype(np.float32)
        V = fit_als(conf.tocsr(), c["als_factors"], c["als_reg"], c["als_iters"])
        # NEW-ITEM cold start: rarely-rated items borrow factors predicted from content
        rare = n_i < c["min_ratings_item"]
        if rare.any() and (~rare).sum() > 50:
            pred = Ridge(alpha=1.0).fit(self.E[~rare], V[~rare]).predict(self.E[rare])
            w = (n_i[rare] / (n_i[rare] + c["min_ratings_item"]))[:, None]
            V[rare] = w * V[rare] + (1 - w) * pred
        self.VF = V.astype(np.float32)
        self.YtY = (self.VF.T @ self.VF + c["als_reg"] * np.eye(self.VF.shape[1])).astype(np.float32)
        self.En = self.E; self.Vn = normalize(self.VF)
        return self

    # ---- per-user state, built from any list of (item, rating): known OR brand-new users
    def state(self, it, rt, prior=None):
        it = np.asarray(it); rt = np.asarray(rt, np.float32); n = len(it)
        mean, std, pos = rt.mean(), rt.std(), (rt >= 4).mean()
        y = self.year[it]; ym = np.nanmean(y) if np.isfinite(y).any() else np.nan
        g = self.G[it]; cnt = g.sum(0)
        gp = ((rt - self.gmean)[:, None] * g).sum(0) / (cnt + 3.0)       # shrunk genre affinity
        if prior is not None: gp = gp + prior
        w = np.clip(rt - 3, 0, None)
        if w.sum() == 0: w = np.ones_like(rt)
        cp = (self.E[it] * w[:, None]).sum(0); cp = cp / (np.linalg.norm(cp) + 1e-9)
        Vs = self.VF[it]; cf = 1 + np.clip(rt - 2, 0, None)              # ALS fold-in (one exact ALS step)
        uf = np.linalg.solve(self.YtY + (Vs.T * (cf - 1)) @ Vs, (Vs.T * cf).sum(1))
        return dict(uf=uf.astype(np.float32), cp=cp.astype(np.float32), gp=gp.astype(np.float32),
                    gs=(cnt / n).astype(np.float32), sc=np.array([n, mean, std, pos, ym], np.float32))

    # ---- candidate generation + feature building (shared by training, eval and serving)
    def build_pairs(self, users, hist, holdout=None, mode="retrieved", k_fn=None, rng=None,
                    batch=250, extra=None, genre_prior=None):
        """mode: 'train' = retrieved + holdout items | 'retrieved' = retrieved only | 'holdout' = holdout items only.
        hist/holdout: users x items CSR. k_fn(n)->k truncates the history the model SEES (cold-start simulation);
        already-seen filtering always uses the full history."""
        c = self.cfg; out = []
        for b0 in range(0, len(users), batch):
            ub = np.asarray(users[b0:b0 + batch]); B = len(ub)
            seen, sts = [], []
            for u in ub:
                s, e = hist.indptr[u], hist.indptr[u + 1]
                it, rt = hist.indices[s:e], hist.data[s:e]; seen.append(it)
                k = k_fn(len(it)) if k_fn else None
                if k and k < len(it):
                    sel = rng.choice(len(it), k, replace=False); it, rt = it[sel], rt[sel]
                sts.append(self.state(it, rt, genre_prior))
            UF = np.stack([s["uf"] for s in sts]); CP = np.stack([s["cp"] for s in sts])
            GP = np.stack([s["gp"] for s in sts]); GS = np.stack([s["gs"] for s in sts])
            sc = np.stack([s["sc"] for s in sts])
            S = UF @ self.VF.T; Sc = CP @ self.E.T
            mu_s, sd_s = S.mean(1), S.std(1) + 1e-6; mu_c, sd_c = Sc.mean(1), Sc.std(1) + 1e-6
            H = holdout[ub].toarray() if holdout is not None else None
            if mode != "holdout":
                Sm, Scm = S.copy(), Sc.copy()
                for r in range(B): Sm[r, seen[r]] = -np.inf; Scm[r, seen[r]] = -np.inf
                ta = np.argpartition(-Sm, c["n_als"], axis=1)[:, :c["n_als"]]
                tc = np.argpartition(-Scm, c["n_content"], axis=1)[:, :c["n_content"]]
                del Sm, Scm
            cands = []
            for r in range(B):
                parts = []
                if mode != "holdout":
                    ss = set(seen[r].tolist())
                    parts += [ta[r], tc[r], np.array([i for i in self.pop_order if i not in ss][:c["n_pop"]], dtype=np.int64)]
                if extra is not None: parts.append(np.asarray(extra))
                if mode in ("train", "holdout"): parts.append(np.flatnonzero(H[r]))
                cd = np.unique(np.concatenate(parts)); cands.append(cd[~np.isin(cd, seen[r])])
            rr = np.repeat(np.arange(B), [len(x) for x in cands]); ii = np.concatenate(cands)
            gi = self.G[ii]; gsz = np.maximum(gi.sum(1), 1)
            a, cs = S[rr, ii], Sc[rr, ii]
            f = {"u_cnt": np.log1p(sc[rr, 0]), "u_mean": sc[rr, 1], "u_std": sc[rr, 2], "u_pos": sc[rr, 3],
                 "u_year": sc[rr, 4], "i_cnt": np.log1p(self.i_cnt[ii]), "i_mean": self.i_mean[ii],
                 "i_pos": self.i_pos[ii], "i_year": self.year[ii], "year_gap": self.year[ii] - sc[rr, 4],
                 "als": a, "als_z": (a - mu_s[rr]) / sd_s[rr], "cont": cs, "cont_z": (cs - mu_c[rr]) / sd_c[rr],
                 "gaff": (GP[rr] * gi).sum(1) / gsz, "gshare": (GS[rr] * gi).sum(1) / gsz,
                 "base_pred": sc[rr, 1] + self.i_mean[ii] - self.gmean, "pop_w": self.pop_w[ii]}
            df = pd.DataFrame(f).astype(np.float32)
            df.insert(0, "i", ii); df.insert(0, "u", ub[rr])
            df["rating"] = H[rr, ii] if H is not None else 0.0
            out.append(df)
        return pd.concat(out, ignore_index=True)

    # ---- training of the two models
    def fit_rankers(self, tr, es):
        import lightgbm as lgb
        c = self.cfg
        def gain(d): r = d["rating"].to_numpy(); return np.select([r >= 4.5, r >= 4, r >= 3.5], [3, 2, 1], 0).astype(int)
        def grp(d): return d.groupby("u", sort=False).size().to_numpy()
        cb = [lgb.early_stopping(40, verbose=False), lgb.log_evaluation(50)]
        base = dict(learning_rate=c["lr"], num_leaves=c["leaves"], min_data_in_leaf=50, feature_fraction=0.8,
                    bagging_fraction=0.8, bagging_freq=1, verbose=-1)
        dt = lgb.Dataset(tr[FEATS], gain(tr), group=grp(tr)); dv = lgb.Dataset(es[FEATS], gain(es), group=grp(es), reference=dt)
        self.ranker = lgb.train({**base, "objective": "lambdarank", "metric": "ndcg", "eval_at": [10]},
                                dt, c["rounds"], valid_sets=[dv], callbacks=cb)
        tr_r, es_r = tr[tr["rating"] > 0], es[es["rating"] > 0]            # taste model: rated items only
        dt = lgb.Dataset(tr_r[FEATS], (tr_r["rating"] >= 4).astype(int)); dv = lgb.Dataset(es_r[FEATS], (es_r["rating"] >= 4).astype(int), reference=dt)
        self.liker = lgb.train({**base, "objective": "binary", "metric": "auc"}, dt, c["rounds"], valid_sets=[dv], callbacks=cb)
        p = self.liker.predict(es_r[FEATS]); self._calib = (p, (es_r["rating"] >= 4).to_numpy())
        return self

    def threshold_for_precision(self, target=0.80):
        """Smallest 'like' probability whose precision (on validation users) reaches `target`."""
        p, y = self._calib; o = np.argsort(-p); cum = np.cumsum(y[o]) / np.arange(1, len(o) + 1)
        ok = np.flatnonzero(cum[100:] >= target)
        return float(p[o][ok[-1] + 100]) if len(ok) else None

    # ---- explanation helpers
    def contributions(self, X):
        """Exact TreeSHAP from LightGBM itself: (n, n_features) log-odds-like contributions + bias."""
        c = self.ranker.predict(X[FEATS], pred_contrib=True)
        return c[:, :-1], c[:, -1]

    def grouped(self, contrib):
        return {g: contrib[:, [FEATS.index(f) for f in fs]].sum(1) for g, fs in GROUPS.items()}

    def _because(self, it, rt, item):
        liked = it[rt >= 4] if (rt >= 4).any() else it[np.argsort(-rt)[:3]]
        sim = 0.5 * (self.En[liked] @ self.En[item]) + 0.5 * (self.Vn[liked] @ self.Vn[item])
        j = int(np.argmax(sim))
        return {"movieId": int(self.movies.movieId[liked[j]]), "title": self.movies.clean_title[liked[j]]} if sim[j] > 0.15 else None

    def _info(self, i):
        m = self.movies.iloc[i]
        return {"movieId": int(m.movieId), "title": m.clean_title, "year": None if np.isnan(m.year) else int(m.year),
                "genres": [g for g in m.genres.split("|") if g and g != "(no genres listed)"]}

    def _reason(self, grp, i, st, row):
        gi = np.flatnonzero(self.G[i])
        if grp == "Your genre taste":
            top = [self.genre_names[j] for j in gi[np.argsort(-st["gp"][gi])] if st["gp"][j] > 0][:2]
            return "You tend to rate %s films highly" % " & ".join(top) if top else "Matches genres you watch often"
        if grp == "Similar to what you liked": return "Close in style to movies you rated highly"
        if grp == "People with similar taste": return "Viewers whose ratings resemble yours loved it"
        if grp == "Community rating": return "Rated %.1f/5 by %s viewers" % (self.i_raw_mean[i], f"{int(self.i_cnt[i]):,}")
        y = self.year[i]; return "Fits your taste for %ds films" % (int(y) // 10 * 10) if np.isfinite(y) else "Fits your profile"

    def _mmr(self, z, items, n, lam=0.6):
        pool = list(np.argsort(-z)[:max(n * 4, 60)]); sel = []
        while pool and len(sel) < n:
            pen = np.array([max((self.En[items[p]] @ self.En[items[s]] for s in sel), default=0.0) for p in pool])
            j = int(np.argmax(z[pool] - lam * pen)); sel.append(pool.pop(j))
        return np.array(sel)

    def text_scores(self, query):
        q = normalize(self.svd.transform(self.tfidf.transform([query]))).astype(np.float32)[0]
        return self.E @ q

    # ---- public API --------------------------------------------------------
    def recommend(self, history=None, genres=None, query=None, top_n=20, diversify=True, qs=None, qw=1.0, qonly=0):
        """history: [(movieId, rating), ...]. Works for 0 ratings (cold start) up to thousands.
        qs: optional per-item text/semantic score vector (len n_items) that replaces TF-IDF `query`; qw: its weight in the blend;
        qonly: if >0, only items inside the top-qonly of qs may be returned (description search that is still personalised)."""
        hh = {}
        for m, r in (history or []):
            if m in self.midx: hh[int(self.midx.get_loc(m))] = float(np.clip(r, 0.5, 5))
        if not hh: return self.cold_start(genres, top_n)
        it = np.array(list(hh)); rt = np.array(list(hh.values()), np.float32)
        prior = np.zeros(len(self.genre_names), np.float32)
        for g in genres or []:
            if g in self.genre_names: prior[self.genre_names.index(g)] = 1.0
        hist = sp.csr_matrix((rt, (np.zeros(len(it), int), it)), shape=(1, self.n_items))
        qs = qs if qs is not None else (self.text_scores(query) if query else None)
        extra = np.argsort(-qs)[:100] if qs is not None else None
        df = self.build_pairs(np.array([0]), hist, mode="retrieved", extra=extra, genre_prior=prior)
        if qs is not None and qonly:
            sub = df[df["i"].isin(np.argsort(-qs)[:qonly])]
            if len(sub) >= min(top_n, 8): df = sub.reset_index(drop=True)
        items = df["i"].to_numpy(); s = self.ranker.predict(df[FEATS]); p = self.liker.predict(df[FEATS])
        z = (s - s.mean()) / (s.std() + 1e-9)
        if qs is not None: q = qs[items]; z = z + qw * (q - q.mean()) / (q.std() + 1e-9)
        order = self._mmr(z, items, top_n) if diversify else np.argsort(-z)[:top_n]
        contrib, _ = self.contributions(df.iloc[order]); grp = self.grouped(contrib)
        st = self.state(it, rt, prior); res = []
        for n, o in enumerate(order):
            i = int(items[o]); pos = sorted(((g, float(v[n])) for g, v in grp.items() if v[n] > 0), key=lambda x: -x[1])[:3]
            d = self._info(i)
            d.update(score=float(s[o]), match=int(round(100 * float(p[o]))),
                     reasons=[{"group": g, "text": self._reason(g, i, st, None), "impact": round(v, 3)} for g, v in pos],
                     because=self._because(it, rt, i),
                     contributions=[{"group": g, "value": round(float(v[n]), 3)} for g, v in grp.items()],
                     mode="personalized", n_ratings=len(it))
            res.append(d)
        return res

    def _diverse(self, base, pool, n, pen=0.4):
        """Greedy pick: high base score, but penalise genres already chosen."""
        z = (base[pool] - base[pool].mean()) / (base[pool].std() + 1e-9)
        taken = np.zeros(len(pool), bool); used = np.zeros(self.G.shape[1]); sel = []
        gp, gsz = self.G[pool], np.maximum(self.G[pool].sum(1), 1)
        for _ in range(min(n, len(pool))):
            sc = z - pen * (gp @ used) / gsz; sc[taken] = -np.inf
            j = int(np.argmax(sc)); taken[j] = True; sel.append(pool[j]); used += gp[j]
        return np.array(sel)

    def cold_start(self, genres=None, top_n=20, explore=0.2, seed=None):
        """No ratings at all: genre-aware, genre-diverse popularity with a little exploration."""
        gi = [self.genre_names.index(g) for g in genres or [] if g in self.genre_names]
        ok = self.i_cnt >= self.min_votes
        if gi: ok &= self.G[:, gi].sum(1) > 0
        pool = np.flatnonzero(ok); pool = pool[np.argsort(-self.pop_w[pool])][:max(top_n * 8, 100)]
        n_exp = int(top_n * explore); rng = np.random.default_rng(seed)
        main = self._diverse(self.pop_w, pool, top_n - n_exp)
        rest = np.setdiff1d(pool, main); exp = rng.choice(rest, min(n_exp, len(rest)), replace=False) if len(rest) else []
        res = []
        for i in list(main) + list(exp):
            d = self._info(int(i))
            txt = "Rated %.1f/5 by %s viewers" % (self.i_raw_mean[i], f"{int(self.i_cnt[i]):,}")
            if gi and self.G[i, gi].any(): txt = "Top-rated in " + self.genre_names[gi[int(np.argmax(self.G[i, gi]))]] + ". " + txt
            d.update(score=float(self.pop_w[i]), match=None, mode="cold-start", n_ratings=0,
                     reasons=[{"group": "Community rating", "text": txt, "impact": 1.0}], because=None,
                     contributions=[{"group": "Community rating", "value": 1.0}])
            res.append(d)
        return res

    def onboarding_items(self, n=12):
        """Items worth asking a new user about: well known, polarising, genre-diverse."""
        pool = np.flatnonzero(self.i_cnt >= np.quantile(self.i_cnt[self.i_cnt > 0], 0.97))
        return [self._info(int(i)) for i in self._diverse(self.i_std, pool, n, pen=0.6)]

    def search(self, query, top_n=10):
        q = self.text_scores(query); top = np.argsort(-q)[:top_n]
        return [{**self._info(int(i)), "score": float(q[i])} for i in top]

    def save(self, path): joblib.dump(self, path, compress=3)
    @staticmethod
    def load(path): return joblib.load(path)

# ----------------------------------------------------------------- training + evaluation
def pick_users(csr, min_items, n, rng, exclude=()):
    rows = np.setdiff1d(np.flatnonzero(row_counts(csr) >= min_items), exclude)
    return rng.choice(rows, min(n, len(rows)), replace=False)

def make_k_fn(rng, choices):
    return lambda n: int(rng.choice(choices))

def build_training_sets(rs, train, val, n_rank=15000, n_es=3000, seed=42):
    """Ranker learns from VAL-period labels (unseen by ALS) with histories randomly truncated -> robust to cold start."""
    rng = np.random.default_rng(seed)
    tr_u = pick_users(val, 3, n_rank, rng); es_u = pick_users(val, 3, n_es, rng, exclude=tr_u)
    tr = rs.build_pairs(tr_u, train, val, "train", make_k_fn(rng, rs.cfg["k_choices"]), rng)
    es = rs.build_pairs(es_u, train, val, "train", None, rng)
    return tr, es, np.concatenate([tr_u, es_u])

def _rel_total(csr, thr=4): d = csr.copy(); d.data = (d.data >= thr).astype(np.float32); return np.asarray(d.sum(1)).ravel()

def topk_metrics(df, score, k, n_rel):
    d = df.assign(rel=(df["rating"] >= 4).astype(np.float32)).sort_values(["u", score], ascending=[True, False], kind="stable")
    d["r"] = d.groupby("u").cumcount(); t = d[d["r"] < k]
    hits = t.groupby("u")["rel"].sum(); disc = (t["rel"] / np.log2(t["r"] + 2)).groupby(t["u"]).sum()
    nr = pd.Series(n_rel[hits.index.to_numpy()], index=hits.index)
    idcg = nr.map(lambda x: (1 / np.log2(np.arange(2, min(int(x), k) + 2))).sum() if x > 0 else np.nan)
    ok = nr > 0
    return {"P@%d" % k: float((hits / k)[ok].mean()), "Recall@%d" % k: float((hits / nr)[ok].mean()),
            "NDCG@%d" % k: float((disc / idcg)[ok].mean()), "HitRate@%d" % k: float((hits[ok] > 0).mean()), "users": int(ok.sum())}

def _diversity(G):
    m = G @ G.T; d = np.diag(m); jac = m / (d[:, None] + d[None, :] - m + 1e-9)
    return float(1 - jac[np.triu_indices(len(G), 1)].mean())

def evaluate(rs, train, val, test, n_users=2000, k=10, seed=7, exclude=()):
    rng = np.random.default_rng(seed); hist = (train + val).tocsr(); nrel = _rel_total(test)
    users = pick_users(test, k, n_users, rng, exclude=exclude); users = users[nrel[users] > 0]
    rep = {}
    # Protocol A: "taste" — rank the movies the user really rated later (relevant = rating >= 4)
    A = rs.build_pairs(users, hist, test, "holdout"); A["p_like"] = rs.liker.predict(A[FEATS])
    A["rel"] = (A["rating"] >= 4)
    rows = {n: topk_metrics(A, c, k, nrel) for n, c in [("LightGBM taste model", "p_like"), ("ALS only", "als"),
            ("user mean + item mean", "base_pred"), ("popularity", "pop_w")]}
    rep["A"] = pd.DataFrame(rows).T; rep["A_base_rate"] = float(A.groupby("u")["rel"].mean().mean())
    # Protocol B: "discovery" — rank the WHOLE catalogue (retrieved candidates only, no peeking)
    Bf = rs.build_pairs(users, hist, test, "retrieved"); Bf["s"] = rs.ranker.predict(Bf[FEATS])
    rows = {n: topk_metrics(Bf, c, k, nrel) for n, c in [("LightGBM ranker", "s"), ("ALS only", "als"), ("popularity", "pop_w")]}
    rep["B"] = pd.DataFrame(rows).T
    top = Bf.sort_values(["u", "s"], ascending=[True, False]).groupby("u").head(k)
    lists = {u: g["i"].to_numpy() for u, g in top.groupby("u")}
    p_item = np.clip(rs.i_cnt / rs.i_cnt.sum(), 1e-12, None)
    div = [_diversity(rs.G[v]) for v in lists.values()]
    rep["B_extra"] = {"coverage": len({i for v in lists.values() for i in v}) / rs.n_items,
                      "novelty_bits": float(np.mean([-np.log2(p_item[v]).mean() for v in lists.values()])),
                      "genre_diversity": float(np.mean(div))}
    # Precision vs. confidence (threshold chosen on validation users, applied to test)
    t = rs.threshold_for_precision(0.80)
    if t is not None:
        sel = A[A["p_like"] >= t]; rep["conf80"] = {"threshold": t, "test_precision": float(sel["rel"].mean()),
                                                      "share_of_items_kept": float(len(sel) / len(A))}
    curve = []
    for th in np.arange(0.5, 0.96, 0.05):
        s = A[A["p_like"] >= th]; curve.append((round(th, 2), float(s["rel"].mean()) if len(s) else np.nan, len(s) / len(A)))
    rep["curve"] = pd.DataFrame(curve, columns=["p_like>=", "precision", "share_kept"])
    return rep

def evaluate_cold_start(rs, train, val, test, ks=(0, 1, 3, 5, 10, 20), n_users=1000, k=10, seed=11, exclude=()):
    """Simulate users who have rated only `k0` movies (ML-32M has no real cold users, so we truncate)."""
    rng = np.random.default_rng(seed); hist = (train + val).tocsr(); nrel = _rel_total(test)
    users = pick_users(test, k, n_users, rng, exclude=exclude); users = users[nrel[users] > 0]
    rows = {}
    for k0 in ks:
        if k0 == 0:                                    # popularity-only path
            hits = []
            for u in users:
                s = set(hist.indices[hist.indptr[u]:hist.indptr[u + 1]].tolist())
                rec = [i for i in rs.pop_order if i not in s][:k]
                t = test[u]; good = set(t.indices[t.data >= 4].tolist()); hits.append(len(good & set(rec)) / k)
            rows[0] = {"B: P@%d" % k: float(np.mean(hits)), "A: P@%d" % k: np.nan}; continue
        kf = (lambda n, k0=k0: k0)
        Bf = rs.build_pairs(users, hist, test, "retrieved", kf, rng); Bf["s"] = rs.ranker.predict(Bf[FEATS])
        A = rs.build_pairs(users, hist, test, "holdout", kf, rng); A["p"] = rs.liker.predict(A[FEATS])
        rows[k0] = {"B: P@%d" % k: topk_metrics(Bf, "s", k, nrel)["P@%d" % k], "A: P@%d" % k: topk_metrics(A, "p", k, nrel)["P@%d" % k]}
    return pd.DataFrame(rows).T.rename_axis("ratings given")
