from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_news, get_global_news
from tradingagents.dataflows.config import get_config
from tradingagents.agents.utils.macro_tools import get_net_liquidity_tool, get_macro_indicators_tool



def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        domain = state.get("domain_category", "Unknown")
        schema_id = state.get("analysis_schema_id", "Unknown")
        analysis_schema = state.get("analysis_schema") or {}
        analysis_questions = analysis_schema.get("analysis_questions", [])
        news_bundle = state.get("news_bundle") or {}
        company_cards = news_bundle.get("company_news_cards", [])
        macro_cards = news_bundle.get("macro_news_cards", [])

        def _fmt_cards(cards, label):
            lines = []
            for c in cards:
                lines.append(
                    f"- [{label}] {c.get('title','')[:120]} | src={c.get('source','')} | score={c.get('relevance_score','')}, event={c.get('event_label','')}"
                )
            return "\n".join(lines) if lines else "- (none)"

        context_block = (
            f"[Category] asset_type={state.get('asset_type','')} domain={domain} schema={schema_id}\n"
            f"[Schema Questions] " + ("; ".join(analysis_questions) if analysis_questions else "(none)") + "\n"
            f"[Top Company News]\n{_fmt_cards(company_cards, 'company')}\n"
            f"[Top Macro News]\n{_fmt_cards(macro_cards, 'macro')}"
        )

        tools = [
            get_news,
            get_global_news,
            get_net_liquidity_tool,  # 추가  
            get_macro_indicators_tool,  # 추가  
        ]

        system_message = (
            "You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Use the available tools: get_news(query, start_date, end_date) for company-specific or targeted news searches, and get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news. Do not simply state the trends are mixed, provide detailed and finegrained analysis and insights that may help traders make decisions."
            + """ Also, consider following causal relationships analysis framework"""
            + """ analyzing causal relationships: \n\n Macro Indicators (Cause) → Fed Response (Mediator) → Asset Price Changes (Effect). \n\n 
            **Analysis Framework:**\n
            ① Net Liquidity (순유동성): Use get_net_liquidity_tool() to track Fed Balance Sheet - (TGA + RRP).
            If INCREASING → recommend higher equity allocation. If DECREASING → recommend cash/bonds.\n
            ② Inflation & Employment: Use get_macro_indicators_tool() for CPI, PCE, Unemployment Rate.   
            High inflation → Fed tightening risk → defensive positioning. Rising unemployment → recession signal → bonds.\n 
            ③ Market Sentiment: Analyze 10Y-2Y yield curve inversion and HYG (high-yield bond ETF) for risk appetite.\n
            \n  
            Use get_news() and get_global_news() for recent macroeconomic news context. 
            Provide a comprehensive causal analysis linking these indicators to trading implications."""
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
        )
        # system_message = (  
        #     "You are a macroeconomic analyst tasked with analyzing causal relationships: "  
        #     "Macro Indicators (Cause) → Fed Response (Mediator) → Asset Price Changes (Effect). "  
        #     "\n\n"  
        #     "**Analysis Framework:**\n"  
        #     "① Net Liquidity (순유동성): Use get_net_liquidity_tool() to track Fed Balance Sheet - (TGA + RRP). "  
        #     "If INCREASING → recommend higher equity allocation. If DECREASING → recommend cash/bonds.\n"  
        #     "② Inflation & Employment: Use get_macro_indicators_tool() for CPI, PCE, Unemployment Rate. "  
        #     "High inflation → Fed tightening risk → defensive positioning. Rising unemployment → recession signal → bonds.\n"  
        #     "③ Market Sentiment: Analyze 10Y-2Y yield curve inversion and HYG (high-yield bond ETF) for risk appetite.\n"  
        #     "\n"  
        #     "Use get_news() and get_global_news() for recent macroeconomic news context. "  
        #     "Provide a comprehensive causal analysis linking these indicators to trading implications."  
        #     + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""  
        # )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. We are looking at the company {ticker}."
                    "\n\n[Context Preloaded]\n{context_block}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)
        prompt = prompt.partial(context_block=context_block)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
