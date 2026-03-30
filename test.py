from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent / ".env")

from data.fetch import AlpacaFetcher

fetcher = AlpacaFetcher()
df = fetcher.get_historical_bars(
    symbol="SPY",
    start="2020-01-14 09:30",
    end="2020-01-14 16:00",
)
print(df.head())
print(f"Shape: {df.shape}")
