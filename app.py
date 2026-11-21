from pathlib import Path
import numpy as np, pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from momentum import load_clean, build_episodes, build_songs, kpis

st.set_page_config(page_title="K-Pop Comeback Momentum", page_icon="🎤", layout="wide")
CSV = next((p for p in ["Atlantic_South_Korea.csv", "data/Atlantic_South_Korea.csv"] if Path(p).exists()), None)
if CSV is None:
    st.error("Atlantic_South_Korea.csv not found next to app.py"); st.stop()


@st.cache_data
def get_data():
    return load_clean(CSV)


@st.cache_data
def get_episodes(min_gap):
    d, _ = get_data()
    ep = build_episodes(d, min_gap)
    return ep, build_songs(d, ep)


d, rep = get_data()

# ------------------------------------------------ sidebar filters
st.sidebar.header("Filters")
dmin, dmax = d["date"].min().date(), d["date"].max().date()
rng = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if len(rng) != 2:
    st.stop()
gap = st.sidebar.slider("Absence (days) that counts as an exit", 1, 14, 3,
                        help="A song must be off the Top 50 for at least this many days before coming back counts as a re-entry.")
ep, songs = get_episodes(gap)
types = st.sidebar.multiselect("Album type", ["single", "album"], default=["single", "album"])
artists = st.sidebar.multiselect("Artist", sorted(songs.artist.unique()))
pool = songs[songs.artist.isin(artists)] if artists else songs
picked = st.sidebar.multiselect("Song", sorted(pool.song.unique()))
min_re = st.sidebar.slider("Minimum re-entries per song", 0, int(songs.reentries.max()), 0)

sf = songs[songs.album_type.isin(types) & (songs.reentries >= min_re)]
if artists: sf = sf[sf.artist.isin(artists)]
if picked: sf = sf[sf.song.isin(picked)]
t0, t1 = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
epf = ep[ep.key.isin(sf.key) & (ep.entry_date >= t0) & (ep.entry_date <= t1)]
df = d[d.key.isin(sf.key) & (d.date >= t0) & (d.date <= t1)]
te = epf[epf.true_entry].merge(songs[["key", "album_type", "total_tracks", "is_explicit", "duration_min"]], on="key")
if sf.empty or te.empty:
    st.warning("No songs match these filters."); st.stop()

st.title("🎤 Comeback Momentum, Chart Re-Entry & Fandom Intensity")
st.caption(f"South Korea Top 50 playlist, {rep['date_min']} to {rep['date_max']} | "
           f"{rep['unique_songs']} songs | {rep['days_observed']} daily snapshots")

K = kpis(epf, sf)
c = st.columns(6)
c[0].metric("Re-Entry Frequency", f"{K['Re-Entry Frequency']:.2f}", help="Average re-entries per song")
c[1].metric("Momentum Spike Score", f"{K['Momentum Spike Score']:.1f}", help="0-100 composite of rank jump, peak rank and popularity spike")
c[2].metric("Post-Comeback Retention", f"{K['Post-Comeback Retention Days']:.1f} d", help="Median days a re-entry stays on the chart")
c[3].metric("Rank Recovery Speed", f"{K['Rank Recovery Speed']:.2f}", help="Rank places gained per day until peak")
aci = K["Album Comeback Advantage Index"]
c[4].metric("Album Comeback Advantage", "n/a" if aci != aci else f"{aci:.2f}", help="Album re-entry score / single re-entry score")
c[5].metric("Fandom Intensity Proxy", f"{K['Fandom Intensity Proxy Score']:.1f}", help="0-100 blend of re-entry frequency, spike sharpness, recovery speed")

t1_, t2_, t3_, t4_, t5_ = st.tabs(["Re-entry timeline", "Momentum spikes", "Comeback vs first entry",
                                    "Content attributes", "Fandom leaderboard"])

# ------------------------------------------------ 1 timeline
with t1_:
    cand = sf.sort_values(["reentries", "days_on_chart"], ascending=False)
    label = cand.song + " - " + cand.artist
    choice = st.selectbox("Song", label.tolist())
    key = cand.loc[label == choice, "key"].iloc[0]
    g = df[df.key == key]
    fig = go.Figure(go.Scatter(x=g.date, y=g.position, mode="lines+markers", name="Rank"))
    for _, r in epf[(epf.key == key) & (epf.entry_type == "Re-entry")].iterrows():
        fig.add_vline(x=r.entry_date, line_dash="dot", line_color="red")
    fig.update_yaxes(autorange="reversed", title="Chart position (1 = top)")
    fig.update_layout(title="Daily rank (red lines = re-entries)")
    st.plotly_chart(fig, width="stretch")
    top = cand.head(15).key
    gt = epf[epf.key.isin(top)].copy()
    gt["end"] = gt.exit_date + pd.Timedelta(days=1)
    gt["label"] = gt.song + " - " + gt.artist
    st.plotly_chart(px.timeline(gt, x_start="entry_date", x_end="end", y="label", color="entry_type",
                                title="Chart stays of the 15 songs with the most re-entries").update_yaxes(autorange="reversed"),
                    width="stretch")
    m = epf[epf.entry_type == "Re-entry"].set_index("entry_date").resample("MS").size().rename("Re-entries").reset_index()
    st.plotly_chart(px.bar(m, x="entry_date", y="Re-entries", title="Re-entries per month"), width="stretch")

# ------------------------------------------------ 2 spikes
with t2_:
    st.plotly_chart(px.scatter(te, x="entry_date", y="momentum_score", color="entry_type", size="rank_jump",
                               hover_data=["song", "artist", "peak_rank", "retention_days"],
                               title="Momentum spike score of every entry / re-entry (bubble size = rank jump)"),
                    width="stretch")
    st.plotly_chart(px.histogram(te, x="momentum_score", color="entry_type", nbins=25, barmode="overlay",
                                 title="Distribution of spike scores"), width="stretch")
    st.subheader("Top 15 spikes")
    st.dataframe(te.sort_values("momentum_score", ascending=False).head(15)[
        ["song", "artist", "entry_type", "entry_date", "entry_rank", "peak_rank", "pop_gain", "retention_days", "momentum_score"]
    ].round({"pop_gain": 1, "momentum_score": 1}), width="stretch", hide_index=True)

# ------------------------------------------------ 3 comeback vs first
with t3_:
    metric = st.selectbox("Metric", ["momentum_score", "retention_days", "peak_rank", "rank_recovery_speed",
                                     "pop_rate", "volatility", "rank_decay_speed"])
    cmp = te[te.entry_type != "Initial (pre-window)"]
    st.plotly_chart(px.box(cmp, x="entry_type", y=metric, color="entry_type", points="outliers",
                           title=f"{metric}: first entry vs re-entry"), width="stretch")
    st.dataframe(cmp.groupby("entry_type")[["momentum_score", "retention_days", "peak_rank", "rank_recovery_speed",
                                            "volatility", "rank_decay_speed"]].median().round(2).T, width="stretch")
    st.caption("Medians. Songs already on the chart on day 1 are excluded because their true entry date is unknown.")

# ------------------------------------------------ 4 content
with t4_:
    attr = st.radio("Compare by", ["Single vs album", "Album size", "Song duration", "Explicit vs clean"], horizontal=True)
    x = te.copy()
    if attr == "Single vs album": x["group"] = x.album_type
    elif attr == "Album size": x["group"] = pd.cut(x.total_tracks, [0, 1, 5, 12, 1000], labels=["1", "2-5", "6-12", "13+"]).astype(str)
    elif attr == "Song duration": x["group"] = pd.qcut(x.duration_min, 4, labels=["Shortest", "Short", "Long", "Longest"]).astype(str)
    else: x["group"] = x.is_explicit.map({True: "Explicit", False: "Clean"})
    s = x.groupby("group").agg(spike=("momentum_score", "median"), retention=("retention_days", "median"),
                               recovery=("rank_recovery_speed", "median"), episodes=("key", "size")).reset_index()
    a, b = st.columns(2)
    a.plotly_chart(px.bar(s, x="group", y="spike", text="episodes", title="Median spike score (label = episodes)"), width="stretch")
    b.plotly_chart(px.bar(s, x="group", y="retention", title="Median retention days"), width="stretch")
    st.dataframe(s.round(2), width="stretch", hide_index=True)
    if attr == "Song duration":
        st.plotly_chart(px.scatter(x, x="duration_min", y="momentum_score", opacity=.5, title="Duration (min) vs spike score"), width="stretch")

# ------------------------------------------------ 5 leaderboard
with t5_:
    mode = st.radio("Rank", ["Songs", "Artists"], horizontal=True)
    if mode == "Songs":
        lb = sf.sort_values("fandom_score", ascending=False).head(20)
        lb = lb.assign(name=lb.song + " - " + lb.artist)
        st.plotly_chart(px.bar(lb[::-1], x="fandom_score", y="name", orientation="h", hover_data=["reentries", "days_on_chart"],
                               title="Top 20 songs by Fandom Intensity Proxy"), width="stretch")
        st.dataframe(lb[["song", "artist", "album_type", "reentries", "days_on_chart", "fandom_score"]].round({"pop_gain": 1, "momentum_score": 1}), width="stretch", hide_index=True)
    else:
        a = sf.groupby("artist").agg(songs=("key", "count"), reentries=("reentries", "sum"),
                                     fandom_score=("fandom_score", "mean"), days_on_chart=("days_on_chart", "sum")).reset_index()
        a = a[a.songs >= 2].sort_values("reentries", ascending=False).head(20)
        st.plotly_chart(px.bar(a[::-1], x="reentries", y="artist", orientation="h", color="fandom_score",
                               title="Artists with the most re-entries (artists with 2+ songs)"), width="stretch")
        st.dataframe(a.round(1), width="stretch", hide_index=True)
    st.caption("Fandom Intensity Proxy = 40% re-entry frequency + 30% popularity spike sharpness + 30% rank recovery speed (percentile scaled 0-100).")
