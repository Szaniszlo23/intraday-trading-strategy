# Intraday Momentum Trading Strategy

An end-to-end intraday algorithmic trading system for SPY, built on a noise-band + VWAP momentum signal. Covers the full lifecycle: backtesting on historical data, live paper/live trading via Alpaca, containerisation with Docker, and cloud deployment on Google Cloud.

---

## Strategy Overview

### Baseline Strategy
The original signal is built around two ideas:
- **Noise band**: define an upper and lower boundary around the open price, scaled by a rolling measure of how much SPY typically moves in the first N minutes of a session (`sigma_open`). If the close breaks cleanly above the upper band *and* above VWAP, go long. Below the lower band *and* below VWAP, go short.
- **VWAP filter**: acts as a trend confirmation — we only trade in the direction the market is already leaning relative to the day's volume-weighted average price.

The baseline samples signals every 30 minutes and uses a 14-day rolling window for sigma.

### V6 Enhanced Strategy (final version)
Six incremental improvements over the baseline, each validated in isolation:

| Enhancement | What it does | Impact |
|---|---|---|
| **V1 — 90-day sigma** | Wider rolling window → more stable noise band | Fewer false signals |
| **V2 — 1-min monitoring** | Check every bar instead of every 30 min | Earlier entries/exits |
| **V3 — Dual-bar confirmation** | Require two consecutive bars beyond the band before entering | Reduces whipsaws |
| **V4 — VWAP trailing stop** | Exit when price crosses back through VWAP | Compresses loss tail |
| **V5 — Vol-regime sizing** | Scale position up in high-vol trending markets, down in choppy ones | Better risk-adjusted returns |
| **V6 — Flow composite** | Blend pre-market return + order imbalance into a sizing multiplier | Further sizing refinement |

**Full-period results (2016–2026, out-of-sample walk-forward confirmed):**
- Annualised Return: ~38%
- Sharpe Ratio: ~2.21
- Max Drawdown: ~12%
- Hit Ratio: ~54%
- Works both long and short — not just a bull-market strategy

---

## Project Structure

```
intraday-trading-strategy/
│
├── main.py                  # Live trader entry point (V6 + baseline)
├── backtest_run.py          # Backtesting entry point with CLI flags
├── test_live.py             # Replay a historical day through the live trader (dry run)
├── test.py                  # Quick Alpaca fetch sanity check
│
├── config/
│   ├── config.py            # AppConfig / StrategyConfig / AlpacaConfig dataclasses
│   └── config.yaml          # Strategy parameters (band_mult, vol_window, sizing, etc.)
│
├── data/
│   ├── fetch.py             # AlpacaFetcher — REST historical bars + WebSocket stream
│   ├── indicators.py        # All technical indicators (VWAP, move_open, vol_regime, etc.)
│   └── preprocess.py        # Feature engineering pipeline (transforms raw bars → model input)
│
├── trader/
│   ├── signals.py           # SignalGenerator (baseline) + EnhancedSignalGenerator (V6)
│   ├── sizing.py            # PositionSizer (baseline) + EnhancedPositionSizer (V6 Layer 2)
│   ├── backtest.py          # Backtester + EnhancedBacktester + BacktestResult
│   └── trading_model.py     # AlpacaTrader — order submission, position queries, account info
│
├── analysis/
│   ├── metrics.py           # PerformanceMetrics — Sharpe, drawdown, alpha/beta, etc.
│   └── plots/               # Saved charts from backtest runs
│
├── tests/
│   └── test_pipeline.py     # Unit tests (synthetic data, no API needed)
│
├── Dockerfile               # Container definition
├── requirements.txt         # Pinned Python dependencies
├── .env.example             # Credential template
└── .dockerignore            # Excludes .env, raw data, and test files from the image
```

---

## Setup

### Prerequisites
- Python 3.12+
- An [Alpaca](https://app.alpaca.markets) account (free tier works — paper trading)
- Docker (for containerised runs)

### Install dependencies
```bash
pip install -r requirements.txt
```

### Configure credentials
```bash
cp .env.example .env
```
Edit `.env` and fill in your Alpaca API key and secret:
```
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_PAPER=true
```
Your `.env` is gitignored and never committed.

---

## Running the Backtest

All backtest commands use `backtest_run.py`. Data can come from a local CSV or fetched live from Alpaca.

### Basic run
```bash
# From a saved CSV (fastest, no API needed)
python backtest_run.py --csv data/raw/spy_intraday.csv

# Fetch fresh data from Alpaca for a date range
python backtest_run.py --start 2016-01-01 --end 2026-01-01
```

### Available flags

| Flag | What it does |
|---|---|
| `--csv PATH` | Load data from a local CSV instead of Alpaca |
| `--start DATE` | Start date for Alpaca fetch (default: 2016-01-01) |
| `--end DATE` | End date for Alpaca fetch (default: 2026-01-01) |
| `--variants` | Run all 6 strategy variants side by side and plot comparisons |
| `--yearly` | Print year-by-year performance breakdown table |
| `--walkforward` | Walk-forward validation with rolling 2-year out-of-sample windows |
| `--longshort` | Split P&L into long vs short contributions with equity chart |
| `--optimize` | Grid-search over band_mult and vol_window parameters (baseline only) |
| `--save PATH` | Save fetched Alpaca data to a CSV for reuse |

### Examples
```bash
# Full analysis: all variants + yearly breakdown + walk-forward validation
python backtest_run.py --csv data/raw/spy_intraday.csv --variants --yearly --walkforward

# Check if the edge is long-only or symmetric
python backtest_run.py --csv data/raw/spy_intraday.csv --longshort

# Fetch 2024–2025 data and save it locally for future runs
python backtest_run.py --start 2024-01-01 --end 2025-12-31 --save data/raw/spy_2024_2025.csv
```

Plots are saved to `analysis/plots/`.

---

## Running the Live Trader

The live trader connects to Alpaca via WebSocket, bootstraps 120 days of historical data for sigma warmup, and trades the V6 strategy in real time.

### Start the live trader (V6, default)
```bash
python main.py
```

### Start with the original baseline strategy
```bash
python main.py --baseline
```

### What happens at startup
1. Fetches 120 days of 1-min historical bars from Alpaca
2. Preprocesses them to compute the per-minute `sigma_open` lookup table
3. Connects to the Alpaca WebSocket and subscribes to SPY bars
4. At 09:30 ET: computes vol-regime factor, pre-market return, order imbalance → fixes position size for the session
5. On every bar: runs the signal generator, rebalances if exposure changes
6. At 15:59 ET: force-closes all positions

### If you start after 09:30 ET
The trader automatically back-fills all missed bars since 09:30 via REST so that VWAP and `move_open` are computed from the correct open price — not the price at the time you started.

### Replay a historical day (dry run, no orders submitted)
```bash
# Replay the most recent day in the historical data
python test_live.py

# Replay a specific date
python test_live.py --date 2025-03-20
```

### Run unit tests (no API keys or data files needed)
```bash
pytest tests/test_pipeline.py -v
```

---

## Running with Docker

### Build the image
```bash
docker build -t trading-strategy .
```

### Run the container
Pass your Alpaca credentials as environment variables — never bake them into the image.
```bash
docker run \
  -e ALPACA_API_KEY=your_key_here \
  -e ALPACA_SECRET_KEY=your_secret_here \
  -e ALPACA_PAPER=true \
  trading-strategy
```

The container runs `python main.py` (V6 live trader) by default. Logs stream to stdout and are visible in the terminal.

---

## Cloud Deployment (Google Cloud)

The recommended setup is a small Compute Engine VM (`e2-micro`, ~$7/month) in `us-east1` (closest region to NYSE). It runs 24/7, restarts the container automatically on reboot, and sends all logs to Cloud Logging.

### Step 1 — Set up Google Cloud
1. Create a project at [console.cloud.google.com](https://console.cloud.google.com)
2. Enable billing
3. Enable the **Artifact Registry** and **Compute Engine** APIs

### Step 2 — Push the Docker image to Artifact Registry
```bash
# Authenticate Docker with Google Cloud
gcloud auth configure-docker us-east1-docker.pkg.dev

# Tag your local image
docker tag trading-strategy us-east1-docker.pkg.dev/YOUR_PROJECT_ID/trading/trading-strategy:latest

# Push
docker push us-east1-docker.pkg.dev/YOUR_PROJECT_ID/trading/trading-strategy:latest
```

### Step 3 — Store API keys in Secret Manager
Never pass secrets as plain text in the VM startup script.
```bash
echo -n "your_api_key" | gcloud secrets create ALPACA_API_KEY --data-file=-
echo -n "your_secret"  | gcloud secrets create ALPACA_SECRET_KEY --data-file=-
```

### Step 4 — Create the VM
```bash
gcloud compute instances create trading-vm \
  --zone=us-east1-b \
  --machine-type=e2-micro \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --scopes=cloud-platform
```

### Step 5 — SSH in and run the container
```bash
gcloud compute ssh trading-vm --zone=us-east1-b
```
Inside the VM:
```bash
# Install Docker
sudo apt-get update && sudo apt-get install -y docker.io

# Pull and run with auto-restart
sudo docker run -d --restart=always \
  -e ALPACA_API_KEY=$(gcloud secrets versions access latest --secret=ALPACA_API_KEY) \
  -e ALPACA_SECRET_KEY=$(gcloud secrets versions access latest --secret=ALPACA_SECRET_KEY) \
  -e ALPACA_PAPER=true \
  us-east1-docker.pkg.dev/YOUR_PROJECT_ID/trading/trading-strategy:latest
```

### View logs
```bash
# On the VM
sudo docker logs -f $(sudo docker ps -q)

# Or from anywhere via Cloud Logging
gcloud logging read "resource.type=gce_instance" --limit=50
```

---

## Key Design Decisions

- **No look-ahead bias**: `sigma_open` uses a `shift(1)` so today's band width is always derived from yesterday's data and earlier. The live trader replicates this exactly using a pre-computed lookup table built at bootstrap.
- **Timezone safety**: all time comparisons use `ZoneInfo("America/New_York")` explicitly, so the system works correctly regardless of where the server is hosted.
- **Leverage cap**: the sizer uses `equity / 2` as the sizing base so the dynamic 4× leverage range maps to 0–2× of total equity, staying within Alpaca's margin limits while preserving the full vol-regime scaling.
- **WebSocket + REST hybrid**: a heartbeat loop fires every 60 seconds. If no bar has arrived in 90+ seconds during market hours, it fetches the last 5 minutes via REST and replays any missed bars.
- **Separation of concerns**: signal logic, sizing logic, execution logic, and data fetching are in separate modules so each can be changed independently.

---

## Limitations

- **IEX feed only**: the free Alpaca tier only provides IEX data, which has lower volume coverage than SIP. This may affect volume-weighted indicators (VWAP, order imbalance) on low-liquidity names. SPY is liquid enough that this is not a material issue.
- **Single instrument**: the strategy is built and validated on SPY only. Extension to other instruments would require re-validating the sigma and band parameters.
- **No WebSocket auto-reconnect**: if the WebSocket stream drops, the heartbeat loop switches to REST polling but does not attempt to restart the stream. For production use, add reconnection logic.
- **Backtest assumes 2× max leverage**: the live sizer uses `equity / 2` but the backtest uses the full equity with a 4× cap. For a perfectly accurate live-vs-backtest comparison, the backtest would need the same scaling.
