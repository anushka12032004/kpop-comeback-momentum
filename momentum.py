"""Core logic: data validation, chart re-entry detection, momentum metrics, fandom proxy."""
import numpy as np
import pandas as pd

BASE = 51  # rank used as the "just off the chart" baseline


def pct(s):
    return s.rank(pct=True) * 100


# ------------------------------------------------------------------ 1. validation
def load_clean(path):
    d = pd.read_csv(path)
    rep = {"rows_raw": int(len(d))}
    d["date"] = pd.to_datetime(d["date"], dayfirst=True, errors="coerce")
    rep["bad_dates"] = int(d["date"].isna().sum())
    d = d.dropna(subset=["date"]).copy()
    d["song"] = d["song"].astype(str).str.strip()
    d["artist"] = d["artist"].astype(str).str.strip()
    d["key"] = d["song"] + " — " + d["artist"]
    d["duration_min"] = d["duration_ms"] / 60000
    cnt = d.groupby("date").size()
    rep["days_not_50_rows"] = {str(k.date()): int(v) for k, v in cnt[cnt != 50].items()}
    d["_o"] = np.arange(len(d))
    d = d.sort_values(["date", "_o"]).groupby("date").head(50)      # keep first 50 rows per day
    before = len(d)
    d = d.drop_duplicates(["date", "key"])
    rep["duplicate_song_rows_removed"] = int(before - len(d))
    alld = pd.date_range(d["date"].min(), d["date"].max())
    rep["missing_calendar_days"] = int(len(set(alld) - set(d["date"].unique())))
    rep["days_observed"] = int(d["date"].nunique())
    rep["date_min"], rep["date_max"] = str(d["date"].min().date()), str(d["date"].max().date())
    rep["popularity_zero_as_missing"] = int((d["popularity"] == 0).sum())
    d.loc[d["popularity"] == 0, "popularity"] = np.nan
    d = d.sort_values(["key", "date"])
    d["popularity"] = d.groupby("key")["popularity"].transform(lambda s: s.ffill().bfill())
    d["popularity"] = d["popularity"].fillna(d["popularity"].median())
    rep["popularity_out_of_range"] = int(((d["popularity"] < 0) | (d["popularity"] > 100)).sum())
    mode = d.groupby("key")["album_type"].agg(lambda s: s.mode().iloc[0])
    rep["songs_with_conflicting_album_type"] = int((d.groupby("key")["album_type"].nunique() > 1).sum())
    d["album_type"] = d["key"].map(mode).replace({"compilation": "album"})
    dates = np.sort(d["date"].unique())
    d["di"] = d["date"].map({x: i for i, x in enumerate(dates)})
    rep["unique_songs"] = int(d["key"].nunique())
    return d.sort_values(["date", "position"]).reset_index(drop=True), rep


# ------------------------------------------------------------------ 2-4. episodes
def build_episodes(d, min_gap=3):
    """An episode = consecutive stay on the chart. A new episode starts after the song was
    absent for >= min_gap observed days. Episode 0 is the first entry; later ones are re-entries."""
    last_di = d["di"].max()
    rows = []
    for key, g in d.sort_values("di").groupby("key", sort=False):
        di, pos, pop = g["di"].values, g["position"].values, g["popularity"].values
        dt = pd.DatetimeIndex(g["date"].values)
        cuts = np.where(np.diff(di) - 1 >= min_gap)[0] + 1
        starts, ends = np.r_[0, cuts], np.r_[cuts, len(di)]
        prev_pop, prev_exit = None, None
        for r, (s, e) in enumerate(zip(starts, ends)):
            p, q, t = pos[s:e], pop[s:e], dt[s:e]
            dd = np.asarray((t - t[0]).days)
            k = int(np.argmin(p))
            kp = int(np.argmax(q))
            ref = prev_pop if r > 0 else q[0]
            dtp = max(int(dd[kp]), 1)
            days_to_peak = max(int(dd[k]), 1)
            post = int((t[-1] - t[k]).days)
            rows.append(dict(
                key=key, song=g["song"].iloc[0], artist=g["artist"].iloc[0], episode_no=r,
                entry_type=("Initial (pre-window)" if (r == 0 and di[s] == 0)
                            else "First entry" if r == 0 else "Re-entry"),
                true_entry=not (r == 0 and di[s] == 0),
                entry_date=t[0], exit_date=t[-1], ongoing=bool(di[e - 1] == last_di),
                gap_days_before=(int((t[0] - prev_exit).days - 1) if r > 0 else np.nan),
                entry_rank=int(p[0]), peak_rank=int(p.min()), rank_jump=BASE - int(p[0]),
                peak_strength=BASE - int(p.min()), climb=int(p[0] - p.min()),
                days_to_peak=days_to_peak, rank_recovery_speed=float((p[0] - p.min()) / days_to_peak),
                entry_pop=float(q[0]), peak_pop=float(q.max()), pop_gain=float(max(q.max() - ref, 0)),
                pop_rate=float(max(q.max() - ref, 0) / dtp),
                retention_days=int((t[-1] - t[0]).days + 1), days_present=int(len(p)),
                rank_decay_speed=float((p[-1] - p.min()) / post) if post > 0 else 0.0,
                volatility=float(np.abs(np.diff(p)).mean()) if len(p) > 1 else 0.0))
            prev_pop, prev_exit = float(q[-1]), t[-1]
    ep = pd.DataFrame(rows)
    m = ep["true_entry"]
    for c, n in [("rank_jump", "_a"), ("peak_strength", "_b"), ("pop_rate", "_c")]:
        ep[n] = np.nan
        ep.loc[m, n] = pct(ep.loc[m, c])
    ep["momentum_score"] = ep[["_a", "_b", "_c"]].mean(axis=1)
    return ep.drop(columns=["_a", "_b", "_c"])


# ------------------------------------------------------------------ 6. songs + fandom proxy
def build_songs(d, ep):
    a = d.groupby("key").agg(song=("song", "first"), artist=("artist", "first"),
                             album_type=("album_type", "first"),
                             total_tracks=("total_tracks", "median"),
                             is_explicit=("is_explicit", "max"),
                             duration_min=("duration_min", "median"),
                             days_on_chart=("date", "count"), best_rank=("position", "min"),
                             first_seen=("date", "min"), last_seen=("date", "max"))
    re = ep[ep.entry_type == "Re-entry"].groupby("key").agg(
        reentries=("episode_no", "size"), mean_gap_days=("gap_days_before", "mean"))
    t = ep[ep.true_entry].groupby("key").agg(
        momentum=("momentum_score", "mean"), retention=("retention_days", "mean"),
        spike_sharpness=("pop_rate", "max"), recovery=("rank_recovery_speed", "mean"))
    s = a.join(re).join(t)
    s["reentries"] = s["reentries"].fillna(0).astype(int)
    for c in ["momentum", "retention", "spike_sharpness", "recovery"]:
        s[c] = s[c].fillna(0)
    s["fandom_score"] = (0.4 * pct(s["reentries"]) + 0.3 * pct(s["spike_sharpness"])
                         + 0.3 * pct(s["recovery"])).round(1)
    return s.reset_index()


# ------------------------------------------------------------------ KPIs
def kpis(ep, songs):
    re = ep[ep.entry_type == "Re-entry"]
    a = re[re.key.isin(songs[songs.album_type == "album"].key)]["momentum_score"].mean()
    s = re[re.key.isin(songs[songs.album_type == "single"].key)]["momentum_score"].mean()
    return {
        "Re-Entry Frequency": float(songs["reentries"].mean()) if len(songs) else np.nan,
        "Momentum Spike Score": float(ep["momentum_score"].mean()),
        "Post-Comeback Retention Days": float(re["retention_days"].median()),
        "Rank Recovery Speed": float(re["rank_recovery_speed"].mean()),
        "Album Comeback Advantage Index": float(a / s) if s and s == s else np.nan,
        "Fandom Intensity Proxy Score": float(songs["fandom_score"].mean()) if len(songs) else np.nan}
