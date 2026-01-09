from openai import OpenAI
from .config import get_config


def get_stock_news_openai(query, start_date, end_date):
    print(f"[OPENAI_NEWS] ====== get_stock_news_openai() 함수 시작 ======")
    print(f"[OPENAI_NEWS] 입력: query={query}, start_date={start_date}, end_date={end_date}")
    config = get_config()
    print(f"[OPENAI_NEWS] OpenAI 클라이언트 생성 중...")
    client = OpenAI(base_url=config["backend_url"])
    print(f"[OPENAI_NEWS] OpenAI API 호출 예정 (responses.create)...")

    response = client.responses.create(
        model=config["quick_think_llm"],
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Can you search Social Media for {query} from {start_date} to {end_date}? Make sure you only get the data posted during that period.",
                    }
                ],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[
            {
                "type": "web_search_preview",
                "user_location": {"type": "approximate"},
                "search_context_size": "low",
            }
        ],
        temperature=1,
        max_output_tokens=4096,
        top_p=1,
        store=True,
    )
    print(f"[OPENAI_NEWS] OpenAI API 응답 수신 완료")
    result = response.output[1].content[0].text
    print(f"[OPENAI_NEWS] 결과 추출 완료, 길이: {len(result) if result else 0}")
    print(f"[OPENAI_NEWS] ====== get_stock_news_openai() 함수 종료 ======")
    return result


def get_global_news_openai(curr_date, look_back_days=7, limit=5):
    config = get_config()
    client = OpenAI(base_url=config["backend_url"])

    response = client.responses.create(
        model=config["quick_think_llm"],
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Can you search global or macroeconomics news from {look_back_days} days before {curr_date} to {curr_date} that would be informative for trading purposes? Make sure you only get the data posted during that period. Limit the results to {limit} articles.",
                    }
                ],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[
            {
                "type": "web_search_preview",
                "user_location": {"type": "approximate"},
                "search_context_size": "low",
            }
        ],
        temperature=1,
        max_output_tokens=4096,
        top_p=1,
        store=True,
    )

    return response.output[1].content[0].text


def get_fundamentals_openai(ticker, curr_date):
    config = get_config()
    client = OpenAI(base_url=config["backend_url"])

    response = client.responses.create(
        model=config["quick_think_llm"],
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Can you search Fundamental for discussions on {ticker} during of the month before {curr_date} to the month of {curr_date}. Make sure you only get the data posted during that period. List as a table, with PE/PS/Cash flow/ etc",
                    }
                ],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[
            {
                "type": "web_search_preview",
                "user_location": {"type": "approximate"},
                "search_context_size": "low",
            }
        ],
        temperature=1,
        max_output_tokens=4096,
        top_p=1,
        store=True,
    )

    return response.output[1].content[0].text