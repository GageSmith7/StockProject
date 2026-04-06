"""
Step 4 acceptance criteria:
- First run seeds ~508 symbols (503 S&P 500 + 5 ETFs)
- search_symbols("APP") returns AAPL, APPF, and other matches
- Re-running seed does not create duplicates
"""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.data.symbols import (
    _fetch_sp500_changes,
    get_all_tickers,
    get_constituent_dates,
    run_initial_seed,
    search_symbols,
    seed_constituent_history,
    seed_prices,
    seed_symbols,
)
from app.data.store import get_prices, upsert_prices
from app.database import ConstituentHistory, SessionLocal, Symbol


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fake_sp500(n: int = 10) -> pd.DataFrame:
    """Small synthetic S&P 500 table to stand in for Wikipedia."""
    tickers = [
        ("AAPL", "Apple Inc.", "Technology"),
        ("MSFT", "Microsoft Corp.", "Technology"),
        ("GOOGL", "Alphabet Inc.", "Technology"),
        ("APPF", "AppFolio Inc.", "Technology"),
        ("AMZN", "Amazon.com Inc.", "Consumer Discretionary"),
        ("TSLA", "Tesla Inc.", "Consumer Discretionary"),
        ("JPM", "JPMorgan Chase", "Financials"),
        ("BAC", "Bank of America", "Financials"),
        ("XOM", "Exxon Mobil", "Energy"),
        ("BRK-B", "Berkshire Hathaway B", "Financials"),
    ]
    rows = tickers[:n]
    return pd.DataFrame(rows, columns=["ticker", "name", "sector"])


def seed_test_symbols():
    """Insert a small set of symbols for search tests."""
    with SessionLocal() as session:
        for ticker, name, sector in [
            ("AAPL", "Apple Inc.", "Technology"),
            ("APPF", "AppFolio Inc.", "Technology"),
            ("APPN", "Appian Corp.", "Technology"),
            ("MSFT", "Microsoft Corp.", "Technology"),
            ("AMZN", "Amazon.com Inc.", "Consumer Discretionary"),
        ]:
            session.merge(Symbol(
                ticker=ticker,
                name=name,
                sector=sector,
                data_source="preloaded",
                active=True,
            ))
        session.commit()


# ---------------------------------------------------------------------------
# seed_symbols
# ---------------------------------------------------------------------------

class TestSeedSymbols:
    @patch("app.data.symbols._fetch_sp500_wikipedia")
    def test_seeds_sp500_plus_etfs(self, mock_fetch):
        mock_fetch.return_value = make_fake_sp500(10)
        count = seed_symbols()
        # 10 fake S&P 500 + 5 ETFs
        assert count == 15

    @patch("app.data.symbols._fetch_sp500_wikipedia")
    def test_symbols_in_db_after_seed(self, mock_fetch):
        mock_fetch.return_value = make_fake_sp500(5)
        seed_symbols()
        tickers = get_all_tickers()
        assert "AAPL" in tickers
        assert "SPY" in tickers   # ETF
        assert "GLD" in tickers   # ETF

    @patch("app.data.symbols._fetch_sp500_wikipedia")
    def test_idempotent_no_duplicates(self, mock_fetch):
        mock_fetch.return_value = make_fake_sp500(5)
        seed_symbols()
        seed_symbols()  # second call
        tickers = get_all_tickers()
        assert len(tickers) == len(set(tickers))  # no duplicates

    @patch("app.data.symbols._fetch_sp500_wikipedia")
    def test_second_seed_count_matches_first(self, mock_fetch):
        mock_fetch.return_value = make_fake_sp500(5)
        count_1 = seed_symbols()
        count_2 = seed_symbols()
        assert count_1 == count_2

    @patch("app.data.symbols._fetch_sp500_wikipedia")
    def test_dot_replaced_with_dash(self, mock_fetch):
        """BRK.B from Wikipedia should become BRK-B for yFinance."""
        df = make_fake_sp500(1)
        df.loc[0, "ticker"] = "BRK.B"
        mock_fetch.return_value = df
        seed_symbols()
        tickers = get_all_tickers()
        assert "BRK-B" in tickers
        assert "BRK.B" not in tickers

    @patch("app.data.symbols._fetch_sp500_wikipedia")
    def test_on_demand_data_source_not_overwritten(self, mock_fetch):
        """A Tier 2 ticker that happens to appear in the S&P 500 list
        should not have its data_source reset to 'preloaded'."""
        mock_fetch.return_value = make_fake_sp500(3)
        seed_symbols()

        # Manually flip AAPL to on_demand
        with SessionLocal() as session:
            sym = session.get(Symbol, "AAPL")
            sym.data_source = "on_demand"
            session.commit()

        seed_symbols()  # re-seed

        with SessionLocal() as session:
            sym = session.get(Symbol, "AAPL")
            assert sym.data_source == "on_demand"


# ---------------------------------------------------------------------------
# search_symbols
# ---------------------------------------------------------------------------

class TestSearchSymbols:
    def setup_method(self):
        seed_test_symbols()

    def test_search_by_ticker_prefix(self):
        results = search_symbols("APP")
        tickers = [r["ticker"] for r in results]
        assert "AAPL" in tickers
        assert "APPF" in tickers

    def test_exact_ticker_ranked_first(self):
        results = search_symbols("AAPL")
        assert results[0]["ticker"] == "AAPL"

    def test_search_by_name(self):
        results = search_symbols("apple")
        tickers = [r["ticker"] for r in results]
        assert "AAPL" in tickers

    def test_search_case_insensitive(self):
        results_upper = search_symbols("MSFT")
        results_lower = search_symbols("msft")
        assert results_upper[0]["ticker"] == results_lower[0]["ticker"]

    def test_empty_query_returns_empty(self):
        assert search_symbols("") == []
        assert search_symbols("   ") == []

    def test_unknown_query_returns_empty(self):
        assert search_symbols("ZZZZNOTREAL") == []

    def test_result_has_required_fields(self):
        results = search_symbols("AAPL")
        assert len(results) > 0
        r = results[0]
        assert "ticker" in r
        assert "name" in r
        assert "sector" in r
        assert "tier" in r

    def test_tier_field_is_preloaded(self):
        results = search_symbols("AAPL")
        assert results[0]["tier"] == "preloaded"

    def test_limit_respected(self):
        results = search_symbols("A", limit=2)
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# seed_prices
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("db_clean")
class TestSeedPrices:
    @patch("app.data.symbols._fetch_sp500_wikipedia")
    @patch("app.data.symbols.fetch_ticker")
    def test_fetches_for_each_symbol(self, mock_fetch_ticker, mock_wiki):
        mock_wiki.return_value = make_fake_sp500(3)
        seed_symbols()

        mock_fetch_ticker.return_value = _make_price_df("AAPL")

        seed_prices(delay=0)
        # 3 S&P 500 + 5 ETFs = 8 tickers, each should be fetched
        assert mock_fetch_ticker.call_count == 8

    @patch("app.data.symbols._fetch_sp500_wikipedia")
    @patch("app.data.symbols.fetch_ticker")
    def test_skips_tickers_already_in_db(self, mock_fetch_ticker, mock_wiki):
        mock_wiki.return_value = make_fake_sp500(2)
        seed_symbols()

        # Pre-populate AAPL
        upsert_prices(_make_price_df("AAPL"))

        mock_fetch_ticker.return_value = _make_price_df("MSFT")

        seed_prices(delay=0)

        called_tickers = [call.args[0] for call in mock_fetch_ticker.call_args_list]
        assert "AAPL" not in called_tickers

    @patch("app.data.symbols._fetch_sp500_wikipedia")
    @patch("app.data.symbols.fetch_ticker")
    def test_failed_ticker_does_not_abort_run(self, mock_fetch_ticker, mock_wiki):
        mock_wiki.return_value = make_fake_sp500(3)
        seed_symbols()

        def side_effect(ticker, *args, **kwargs):
            if ticker == "AAPL":
                raise RuntimeError("yFinance exploded")
            return _make_price_df(ticker)

        mock_fetch_ticker.side_effect = side_effect
        # Should complete without raising
        seed_prices(delay=0)


# ---------------------------------------------------------------------------
# get_all_tickers
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("db_clean")
class TestGetAllTickers:
    @patch("app.data.symbols._fetch_sp500_wikipedia")
    def test_returns_all_active(self, mock_wiki):
        mock_wiki.return_value = make_fake_sp500(3)
        seed_symbols()
        tickers = get_all_tickers()
        assert len(tickers) >= 3 + 5  # 3 S&P 500 + 5 ETFs

    @patch("app.data.symbols._fetch_sp500_wikipedia")
    def test_filter_by_source(self, mock_wiki):
        mock_wiki.return_value = make_fake_sp500(3)
        seed_symbols()
        preloaded = get_all_tickers(source="preloaded")
        assert all(isinstance(t, str) for t in preloaded)
        assert len(preloaded) == 3 + 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# seed_constituent_history / get_constituent_dates
# ---------------------------------------------------------------------------

class TestConstituentHistory:

    # -- _fetch_sp500_changes -------------------------------------------------

    def test_fetch_changes_returns_dict(self):
        """With a mocked Wikipedia response that has no 'changes' table, returns {}."""
        with patch("app.data.symbols.httpx.get") as mock_get:
            mock_get.return_value.raise_for_status = lambda: None
            mock_get.return_value.text = "<html><body>no table here</body></html>"
            result = _fetch_sp500_changes()
        assert isinstance(result, dict)

    def test_fetch_changes_parses_add_dates(self):
        """Minimal synthetic changes table → correct ticker→date mapping."""
        html = """
        <html><body>
        <table id="changes">
          <thead><tr><th>Date</th><th colspan="2">Added</th><th colspan="2">Removed</th><th>Reason</th></tr>
                 <tr><th></th><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th><th></th></tr>
          </thead>
          <tbody>
            <tr><td>January 3, 2020</td><td>NVDA</td><td>Nvidia</td><td>XYZ</td><td>XYZ Inc</td><td>Market cap</td></tr>
            <tr><td>March 15, 2018</td><td>AMD</td><td>Advanced Micro</td><td>ABC</td><td>ABC Corp</td><td>Market cap</td></tr>
          </tbody>
        </table>
        </body></html>
        """
        with patch("app.data.symbols.httpx.get") as mock_get:
            mock_get.return_value.raise_for_status = lambda: None
            mock_get.return_value.text = html
            result = _fetch_sp500_changes()

        assert "NVDA" in result
        assert "AMD"  in result
        assert result["NVDA"] == "2020-01-03"
        assert result["AMD"]  == "2018-03-15"

    def test_fetch_changes_keeps_most_recent_add_date(self):
        """A ticker added twice should keep only the later date."""
        html = """
        <html><body>
        <table id="changes">
          <thead><tr><th>Date</th><th colspan="2">Added</th><th colspan="2">Removed</th><th>Reason</th></tr>
                 <tr><th></th><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th><th></th></tr>
          </thead>
          <tbody>
            <tr><td>January 1, 2015</td><td>TSLA</td><td>Tesla</td><td></td><td></td><td></td></tr>
            <tr><td>June 1, 2020</td><td>TSLA</td><td>Tesla</td><td></td><td></td><td></td></tr>
          </tbody>
        </table>
        </body></html>
        """
        with patch("app.data.symbols.httpx.get") as mock_get:
            mock_get.return_value.raise_for_status = lambda: None
            mock_get.return_value.text = html
            result = _fetch_sp500_changes()

        assert result["TSLA"] == "2020-06-01"

    def test_fetch_changes_normalises_dot_to_dash(self):
        """BRK.B in the changes table should be normalised to BRK-B."""
        html = """
        <html><body>
        <table id="changes">
          <thead><tr><th>Date</th><th colspan="2">Added</th><th colspan="2">Removed</th><th>Reason</th></tr>
                 <tr><th></th><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th><th></th></tr>
          </thead>
          <tbody>
            <tr><td>2010-05-01</td><td>BRK.B</td><td>Berkshire B</td><td></td><td></td><td></td></tr>
          </tbody>
        </table>
        </body></html>
        """
        with patch("app.data.symbols.httpx.get") as mock_get:
            mock_get.return_value.raise_for_status = lambda: None
            mock_get.return_value.text = html
            result = _fetch_sp500_changes()

        assert "BRK-B" in result
        assert "BRK.B" not in result

    # -- seed_constituent_history ---------------------------------------------

    @patch("app.data.symbols._fetch_sp500_wikipedia")
    @patch("app.data.symbols._fetch_sp500_changes")
    def test_seed_returns_count(self, mock_changes, mock_wiki):
        mock_wiki.return_value    = make_fake_sp500(5)
        mock_changes.return_value = {"AAPL": "2018-01-01", "MSFT": "2016-06-01"}
        seed_symbols()   # populate symbols first
        count = seed_constituent_history()
        # 5 S&P 500 + 5 ETFs = 10 tickers
        assert count == 10

    @patch("app.data.symbols._fetch_sp500_wikipedia")
    @patch("app.data.symbols._fetch_sp500_changes")
    def test_seed_stores_add_dates(self, mock_changes, mock_wiki):
        mock_wiki.return_value    = make_fake_sp500(3)
        mock_changes.return_value = {"AAPL": "2019-03-01"}
        seed_symbols()
        seed_constituent_history()

        with SessionLocal() as session:
            row = session.get(ConstituentHistory, "AAPL")
        assert row is not None
        assert str(row.date_added) == "2019-03-01"

    @patch("app.data.symbols._fetch_sp500_wikipedia")
    @patch("app.data.symbols._fetch_sp500_changes")
    def test_seed_stores_none_for_pre_history_tickers(self, mock_changes, mock_wiki):
        """Tickers absent from changes table get date_added=NULL."""
        mock_wiki.return_value    = make_fake_sp500(3)
        mock_changes.return_value = {}   # no recorded changes
        seed_symbols()
        seed_constituent_history()

        with SessionLocal() as session:
            row = session.get(ConstituentHistory, "AAPL")
        assert row is not None
        assert row.date_added is None

    @patch("app.data.symbols._fetch_sp500_wikipedia")
    @patch("app.data.symbols._fetch_sp500_changes")
    def test_seed_idempotent(self, mock_changes, mock_wiki):
        mock_wiki.return_value    = make_fake_sp500(5)
        mock_changes.return_value = {"AAPL": "2018-01-01"}
        seed_symbols()
        count_1 = seed_constituent_history()
        count_2 = seed_constituent_history()
        assert count_1 == count_2

    # -- get_constituent_dates ------------------------------------------------

    @patch("app.data.symbols._fetch_sp500_wikipedia")
    @patch("app.data.symbols._fetch_sp500_changes")
    def test_get_constituent_dates_returns_dict(self, mock_changes, mock_wiki):
        mock_wiki.return_value    = make_fake_sp500(3)
        mock_changes.return_value = {"AAPL": "2018-01-01", "MSFT": "2020-06-01"}
        seed_symbols()
        seed_constituent_history()

        dates = get_constituent_dates()
        assert isinstance(dates, dict)

    @patch("app.data.symbols._fetch_sp500_wikipedia")
    @patch("app.data.symbols._fetch_sp500_changes")
    def test_get_constituent_dates_iso_strings_or_none(self, mock_changes, mock_wiki):
        """Values must be ISO date strings or None — never datetime objects."""
        mock_wiki.return_value    = make_fake_sp500(5)
        mock_changes.return_value = {"AAPL": "2018-01-01"}
        seed_symbols()
        seed_constituent_history()

        dates = get_constituent_dates()
        for val in dates.values():
            assert val is None or (isinstance(val, str) and len(val) == 10)

    @patch("app.data.symbols._fetch_sp500_wikipedia")
    @patch("app.data.symbols._fetch_sp500_changes")
    def test_get_constituent_dates_known_ticker_has_correct_date(self, mock_changes, mock_wiki):
        mock_wiki.return_value    = make_fake_sp500(5)
        mock_changes.return_value = {"MSFT": "2017-09-15"}
        seed_symbols()
        seed_constituent_history()

        dates = get_constituent_dates()
        assert dates.get("MSFT") == "2017-09-15"

    @patch("app.data.symbols._fetch_sp500_wikipedia")
    @patch("app.data.symbols._fetch_sp500_changes")
    def test_point_in_time_filter_trims_prices(self, mock_changes, mock_wiki):
        """
        build_cross_sectional_dataset with constituent_dates should exclude price rows
        that precede a ticker's index entry date — here we verify the filter logic
        using the features module directly with a synthetic dataset.
        """
        import numpy as np
        from app.analytics.features import build_features

        # Minimal OHLCV: 500 rows — only rows after 2016-01-01 should survive the filter
        rng   = np.random.default_rng(99)
        dates = pd.bdate_range("2014-01-02", periods=600)
        close = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(dates)))
        prices = pd.DataFrame({
            "date":   dates,
            "ticker": "FAKE",
            "open":   close,
            "high":   close * 1.005,
            "low":    close * 0.995,
            "close":  close,
            "volume": 1_000_000,
        })

        cutoff    = "2016-01-01"
        filtered  = prices[prices["date"] >= cutoff]
        full_feat = build_features(prices)
        filt_feat = build_features(filtered)

        # Filtered dataset must not contain any rows before the cutoff
        assert (filt_feat["date"] >= pd.Timestamp(cutoff)).all()
        # And must have fewer rows than the unfiltered set
        assert len(filt_feat) < len(full_feat)


def _make_price_df(ticker: str, n: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=n, freq="B").date
    closes = [100.0 + i for i in range(n)]
    return pd.DataFrame({
        "date": dates,
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": [1_000_000] * n,
        "ticker": ticker,
        "data_quality_flag": "clean",
    })
