"""Run: python analyze.py [csv_path]  -> artifacts/ (tables) and figures/ (paper charts)."""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr
from momentum import load_clean, build_episodes, build_songs, kpis

warnings.filterwarnings("ignore")
CSV = sys.argv[1] if len(sys.argv) > 1 else "data/Atlantic_South_Korea.csv"
MIN_GAP = 3
ART, FIG = Path("artifacts"), Path("figures")
ART.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

d, rep = load_clean(CSV)
ep = build_episodes(d, MIN_GAP)
songs = build_songs(d, ep)
ep.to_csv(ART / "episodes.csv", index=False); songs.to_csv(ART / "songs.csv", index=False)
K = kpis(ep, songs)
out = {"validation": rep, "kpis": K}
print("VALIDATION", json.dumps(rep, indent=1))

te = ep[ep.true_entry]; re = ep[ep.entry_type == "Re-entry"]; fe = ep[ep.entry_type == "First entry"]
out["episodes"] = {"total": len(ep), "true_entries": len(te), "first_entries": len(fe), "reentries": len(re),
                   "songs_with_reentry": int((songs.reentries > 0).sum()), "max_reentries": int(songs.reentries.max()),
                   "median_gap_days": float(re.gap_days_before.median())}
cmp = te.groupby("entry_type")[["momentum_score", "retention_days", "peak_rank", "rank_recovery_speed",
                                 "pop_rate", "volatility", "rank_decay_speed"]].median().round(2)
print(out["episodes"]); print(cmp)
out["first_vs_reentry"] = cmp.to_dict()

def mw(a, b):
    a, b = a.dropna(), b.dropna()
    return float(mannwhitneyu(a, b).pvalue) if len(a) > 2 and len(b) > 2 else np.nan

meta = songs.set_index("key")
te = te.join(meta[["album_type", "total_tracks", "is_explicit", "duration_min"]], on="key")
rte = te[te.entry_type == "Re-entry"]
tests = {
    "first_vs_reentry_score_p": mw(fe.momentum_score, re.momentum_score),
    "first_vs_reentry_retention_p": mw(fe.retention_days, re.retention_days),
    "single_vs_album_score_p": mw(te[te.album_type == "single"].momentum_score, te[te.album_type == "album"].momentum_score),
    "explicit_vs_clean_score_p": mw(te[te.is_explicit].momentum_score, te[~te.is_explicit].momentum_score),
    "duration_vs_score_rho": float(spearmanr(te.duration_min, te.momentum_score)[0]),
    "tracks_vs_score_rho": float(spearmanr(te.total_tracks, te.momentum_score)[0]),
    "reentry_gap_vs_score_rho": float(spearmanr(rte.gap_days_before, rte.momentum_score, nan_policy="omit")[0])}
out["tests"] = tests; print(json.dumps(tests, indent=1))
attr = {}
for name, col in [("album_type", "album_type"), ("explicit", "is_explicit")]:
    attr[name] = te.groupby(col)[["momentum_score", "retention_days", "rank_recovery_speed", "peak_rank"]].agg(["median", "count"]).round(2)
    print(attr[name])
te["track_bin"] = pd.cut(te.total_tracks, [0, 1, 5, 12, 1000], labels=["1", "2-5", "6-12", "13+"])
te["dur_bin"] = pd.qcut(te.duration_min, 4, labels=["Shortest", "Short", "Long", "Longest"])
print(te.groupby("track_bin")["momentum_score"].agg(["median", "count"]).round(1))
print(te.groupby("dur_bin")["momentum_score"].agg(["median", "count"]).round(1))
out["attr_album"] = te.groupby("album_type")["momentum_score"].agg(["mean", "median", "count"]).round(2).to_dict()
out["attr_explicit"] = te.groupby("is_explicit")["momentum_score"].agg(["mean", "median", "count"]).round(2).to_dict()
out["attr_reentry_album"] = rte.groupby("album_type")["momentum_score"].agg(["mean", "median", "count"]).round(2).to_dict()
out["share_explicit_songs"] = float(songs.is_explicit.mean())
out["top_songs"] = songs.sort_values("fandom_score", ascending=False).head(10)[["song", "artist", "reentries", "fandom_score", "days_on_chart"]].to_dict("records")
art = songs.groupby("artist").agg(songs=("key", "count"), reentries=("reentries", "sum"),
                                  fandom=("fandom_score", "mean"), days=("days_on_chart", "sum"))
art["fandom_top"] = songs.groupby("artist")["fandom_score"].max()
out["top_artists_reentries"] = art.sort_values("reentries", ascending=False).head(8).round(1).reset_index().to_dict("records")
print(json.dumps(out["top_songs"], indent=1)); print(art.sort_values("reentries", ascending=False).head(8).round(1))
ret = re.retention_days; out["reentry_retention"] = {"median": float(ret.median()), "mean": float(ret.mean()),
    "share_over_7d": float((ret > 7).mean()), "first_entry_median": float(fe.retention_days.median())}
out["reentry_count_dist"] = songs.reentries.value_counts().sort_index().head(12).to_dict()
json.dump(out, open(ART / "summary.json", "w"), indent=1, default=str)

# ---- charts
songs.reentries.value_counts().sort_index().plot.bar(figsize=(7, 4), title="Songs by number of re-entries"); plt.xlabel("Re-entries"); plt.tight_layout(); plt.savefig(FIG / "01_reentry_distribution.png", dpi=150); plt.close()
te.boxplot(column="momentum_score", by="entry_type", figsize=(6, 4.5)); plt.suptitle(""); plt.title("Momentum spike score: first entry vs re-entry"); plt.tight_layout(); plt.savefig(FIG / "02_first_vs_reentry.png", dpi=150); plt.close()
te.groupby("album_type")["momentum_score"].median().plot.bar(figsize=(5, 4), title="Median spike score: single vs album"); plt.tight_layout(); plt.savefig(FIG / "03_single_vs_album.png", dpi=150); plt.close()
te.groupby("is_explicit")["momentum_score"].median().rename({False: "Clean", True: "Explicit"}).plot.bar(figsize=(5, 4), title="Median spike score: explicit vs clean"); plt.tight_layout(); plt.savefig(FIG / "04_explicit_vs_clean.png", dpi=150); plt.close()
plt.figure(figsize=(6, 4.5)); plt.scatter(te.duration_min, te.momentum_score, alpha=.4); plt.xlabel("Duration (min)"); plt.ylabel("Spike score"); plt.title("Duration vs spike"); plt.tight_layout(); plt.savefig(FIG / "05_duration_vs_spike.png", dpi=150); plt.close()
songs.sort_values("fandom_score", ascending=False).head(15)[::-1].plot.barh(x="song", y="fandom_score", legend=False, figsize=(8, 6), title="Top 15 songs by fandom intensity proxy"); plt.tight_layout(); plt.savefig(FIG / "06_fandom_leaderboard.png", dpi=150); plt.close()
ep[ep.true_entry].set_index("entry_date").resample("MS").size().plot(figsize=(8, 4), title="New entries and re-entries per month", marker="o"); plt.tight_layout(); plt.savefig(FIG / "07_monthly_entries.png", dpi=150); plt.close()
print("Saved. KPIs:", json.dumps(K, indent=1))
