# Comeback Momentum, Chart Re-Entry and Fandom Intensity - South Korea Top 50

Streamlit dashboard + analysis of daily Top 50 snapshots (18 May 2024 - 27 Nov 2025).

## Run
```
pip install -r requirements.txt
python analyze.py          # optional: tables + paper charts
python -m streamlit run app.py
```
Files: `momentum.py` (validation, re-entry detection, momentum metrics, fandom score), `analyze.py`, `app.py`, `Atlantic_South_Korea.csv`.

## Method
- Validation: 50 rows/day (one duplicated snapshot day trimmed), song-artist key, duration in minutes, popularity 0 treated as missing, album type made consistent per song.
- Episode = consecutive stay on chart. A song returning after >= 3 absent days (adjustable) is a re-entry.
- Momentum Spike Score (0-100) = mean percentile of rank jump, peak rank strength and popularity spike rate.
- Retention days, rank decay speed, volatility, rank recovery speed per episode.
- Fandom Intensity Proxy = 40% re-entry frequency + 30% spike sharpness + 30% rank recovery speed.
