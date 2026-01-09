from .alpha_vantage_common import _make_api_request, format_datetime_for_api

def get_news(ticker, start_date, end_date) -> dict[str, str] | str:
    """Returns live and historical market news & sentiment data from premier news outlets worldwide.

    Covers stocks, cryptocurrencies, forex, and topics like fiscal policy, mergers & acquisitions, IPOs.

    Args:
        ticker: Stock symbol for news articles.
        start_date: Start date for news search.
        end_date: End date for news search.

    Returns:
        Dictionary containing news sentiment data or JSON string.
    """
    print(f"[ALPHA_VANTAGE_NEWS] ====== get_news() 함수 시작 (Alpha Vantage) ======")
    print(f"[ALPHA_VANTAGE_NEWS] 입력: ticker={ticker}, start_date={start_date}, end_date={end_date}")

    params = {
        "tickers": ticker,
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api(end_date),
        "sort": "LATEST",
        "limit": "50",
    }
    print(f"[ALPHA_VANTAGE_NEWS] API 파라미터: {params}")
    print(f"[ALPHA_VANTAGE_NEWS] _make_api_request('NEWS_SENTIMENT', ...) 호출 예정...")
    
    result = _make_api_request("NEWS_SENTIMENT", params)
    print(f"[ALPHA_VANTAGE_NEWS] API 요청 완료, 결과 타입: {type(result)}")
    print(f"[ALPHA_VANTAGE_NEWS] ====== get_news() 함수 종료 (Alpha Vantage) ======")
    return result

def get_insider_transactions(symbol: str) -> dict[str, str] | str:
    """Returns latest and historical insider transactions by key stakeholders.

    Covers transactions by founders, executives, board members, etc.

    Args:
        symbol: Ticker symbol. Example: "IBM".

    Returns:
        Dictionary containing insider transaction data or JSON string.
    """

    params = {
        "symbol": symbol,
    }

    return _make_api_request("INSIDER_TRANSACTIONS", params)