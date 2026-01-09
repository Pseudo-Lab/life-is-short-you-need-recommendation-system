from typing import Annotated
from datetime import datetime
from dateutil.relativedelta import relativedelta
from .googlenews_utils import getNewsData


def get_google_news(
    query: Annotated[str, "Query to search with"],
    curr_date: Annotated[str, "Curr date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    print(f"[GOOGLE_NEWS] ====== get_google_news() 함수 시작 ======")
    print(f"[GOOGLE_NEWS] 입력: query={query}, curr_date={curr_date}, look_back_days={look_back_days}")
    query = query.replace(" ", "+")
    print(f"[GOOGLE_NEWS] 쿼리 변환: {query}")

    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")
    print(f"[GOOGLE_NEWS] 날짜 범위: {before} ~ {curr_date}")

    print(f"[GOOGLE_NEWS] getNewsData() 호출 예정...")
    news_results = getNewsData(query, before, curr_date)
    print(f"[GOOGLE_NEWS] getNewsData() 완료, 결과 개수: {len(news_results)}")

    news_str = ""

    for news in news_results:
        news_str += (
            f"### {news['title']} (source: {news['source']}) \n\n{news['snippet']}\n\n"
        )

    if len(news_results) == 0:
        print(f"[GOOGLE_NEWS] ====== get_google_news() 함수 종료 (결과 없음) ======")
        return ""

    result = f"## {query} Google News, from {before} to {curr_date}:\n\n{news_str}"
    print(f"[GOOGLE_NEWS] 결과 문자열 생성 완료, 길이: {len(result)}")
    print(f"[GOOGLE_NEWS] ====== get_google_news() 함수 종료 ======")
    return result