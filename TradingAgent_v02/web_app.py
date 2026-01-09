from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import os
import json
import asyncio
import yfinance as yf
from fastapi import FastAPI, Request, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta

# Import TradingAgents components
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

class AnalysisRequest(BaseModel):
    ticker: str
    date: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_stock(request: AnalysisRequest):
    # Backward compatibility or fallback
    ticker = request.ticker.upper()
    date_str = request.date
    try:
        ta = TradingAgentsGraph(debug=False, config=DEFAULT_CONFIG.copy())
        final_state, decision = ta.propagate(ticker, date_str)
        accuracy_info = calculate_accuracy(ticker, date_str, decision)
        return JSONResponse(content={
            "status": "success",
            "ticker": ticker,
            "date": date_str,
            "decision": decision,
            "report": final_state.get("trader_investment_plan", "No report available."),
            "accuracy": accuracy_info,
            "full_state": {
                "sentiment": final_state.get("sentiment_report", ""),
                "fundamentals": final_state.get("fundamentals_report", ""),
                "technical": final_state.get("market_report", ""),
                "risk": final_state.get("risk_debate_state", {}).get("judge_decision", "")
            }
        })
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

# --- Profile API ---
from tradingagents.agents.utils.user_profile import profile_manager

class ProfileUpdate(BaseModel):
    text: str

@app.get("/api/profile")
async def get_profile():
    """Get current user investment profile"""
    profile = profile_manager.load_profile()
    return profile.model_dump()

@app.post("/api/profile")
async def update_profile(update: ProfileUpdate):
    """Update profile based on natural language input"""
    updated_profile = profile_manager.update_profile_from_text(update.text)
    return updated_profile.model_dump()

# --- Trade Execution API ---
from tradingagents.agents.utils.portfolio_manager import portfolio_manager

class TradeRequest(BaseModel):
    ticker: str
    price: float
    confidence: str
    reason: str
    action: str = "BUY"
    exchange: str = "NASD"

@app.post("/api/execute_trade")
async def execute_trade(req: TradeRequest):
    print(f"DEBUG: Trade Request Received: {req}")
    result = portfolio_manager.judge_and_execute(req.ticker, req.price, req.confidence, req.reason, action=req.action, exchange_cd=req.exchange)
    return result


@app.websocket("/ws/analyze")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_text()
        request_data = json.loads(data)
        ticker = request_data.get("ticker", "").upper()
        date_str = request_data.get("date", "")
        # Default to 'deep' if not provided
        analysis_mode = request_data.get("mode", "deep").lower() 
        
        if not ticker or not date_str:
            await websocket.send_json({"type": "error", "message": "Missing ticker or date (티커 또는 날짜 누락)"})
            return

        mode_msg = "⚡ Quick Mode (Speed/Cost)" if analysis_mode == "quick" else "🧠 Deep Mode (Precision)"
        await websocket.send_json({"type": "log", "message": f"Initializing analysis for {ticker} on {date_str} [{mode_msg}]..."})

        # Pre-phase logs (highlight in UI: level=error to render red if supported)
        await websocket.send_json({
            "type": "log",
            "level": "error",
            "message": "[PHASE] Category Routing & Schema Injection (분류/스키마 주입)"
        })
        await websocket.send_json({
            "type": "log",
            "level": "error",
            "message": "[PHASE] News Fetch + Preprocess (뉴스 수집/전처리)"
        })
        await websocket.send_json({
            "type": "log",
            "level": "error",
            "message": "[PHASE] Analysts consume routed schema & weighted news (분석 가중 반영)"
        })

        # --- Callback Definition ---
        from langchain_core.callbacks import BaseCallbackHandler
        from langchain_core.outputs import LLMResult
        
        class WebSocketCallback(BaseCallbackHandler):
            def __init__(self, ws):
                self.ws = ws
                self.cost = 0.0
                self.current_step = ""
                self.loop = asyncio.get_running_loop()
            
            def _send_json(self, data):
                asyncio.run_coroutine_threadsafe(self.ws.send_json(data), self.loop)

            def on_chain_start(self, serialized, inputs, **kwargs):
                # LangGraph usually passes node name in metadata
                node_name = kwargs.get("metadata", {}).get("langgraph_node", "")
                if not node_name:
                    # Fallback
                    node_name = serialized.get("name", "") if serialized else ""
                
                print(f"DEBUG: on_chain_start node_name='{node_name}'")

                # Filter out system chains and common internal nodes
                if node_name and node_name not in ["LangGraph", "__start__", "branches", "RunnableSequence"]:
                     step_map = {
                        "Market Analyst": "Market Research (시장 조사)",
                        "News Analyst": "News Analysis (뉴스 분석)", 
                        "Social Analyst": "Social Analysis (소셜 분석)", 
                        "Fundamentals Analyst": "Fundamentals (기본적 분석)", 
                        "Bull Researcher": "Debate (토론 - Bull)",
                        "Bear Researcher": "Debate (토론 - Bear)",
                        "Research Manager": "Debate (토론 - Manager)",
                        "Risky Analyst": "Risk Assessment (리스크 평가 - Risky)",
                        "Safe Analyst": "Risk Assessment (리스크 평가 - Safe)",
                        "Neutral Analyst": "Risk Assessment (리스크 평가 - Neutral)",
                        "Risk Judge": "Risk Assessment (리스크 평가 - Judge)",
                        "Trader": "Final Decision (최종 결정)"
                    }
                     if node_name in step_map:
                        new_step = step_map[node_name]
                        if new_step != self.current_step:
                            self.current_step = new_step
                            self._send_json({"type": "progress", "step": new_step})
                            self._send_json({"type": "log", "message": f"\n=== {new_step} ==="})

            def on_llm_new_token(self, token: str, **kwargs):
                # Basic Cost Estimation (Model Dependent)
                # Quick Mode (gpt-4o-mini) is ~30x cheaper than Deep Mode (gpt-4o) used as baseline here
                cost_multiplier = 0.00003
                if analysis_mode == "quick":
                    cost_multiplier = 0.000001
                    
                self.cost += (len(token)/4.0) * cost_multiplier
                self._send_json({
                    "type": "stream_chunk",
                    "cost": self.cost
                })

            def on_llm_end(self, response: LLMResult, **kwargs):
                 pass

            def on_tool_start(self, serialized, input_str, **kwargs):
                 tool_name = serialized.get("name", "Unknown Tool")
                 self._send_json({"type": "log", "message": f"[Tool] Executing {tool_name}..."})
                 
            def on_tool_end(self, output, **kwargs):
                 self._send_json({"type": "log", "message": f"[Tool] Finished."})

        # --- Refactored Analysis Loop ---
        from tradingagents.agents.utils.recommender import recommender
        
        # Helper for running one analysis flow
        async def run_analysis(target_ticker: str, target_date: str, is_primary: bool = False):
            try:
                # Callback (Fresh instance per run)
                callback = WebSocketCallback(websocket)
                
                # Dynamic Config based on Mode
                current_config = DEFAULT_CONFIG.copy()
                if analysis_mode == "quick":
                    current_config["deep_think_llm"] = "gpt-4o-mini"
                    print(f"DEBUG: Using Quick Mode (gpt-4o-mini)")
                else:
                    print(f"DEBUG: Using Deep Mode (gpt-4o)")
                
                # Graph Setup
                target_ta = TradingAgentsGraph(debug=False, config=current_config)
                target_ta.ticker = target_ticker
                
                run_config = target_ta.propagator.get_graph_args()
                run_config["callbacks"] = [callback]
                run_config["recursion_limit"] = 150

                # Load Profile
                current_profile = profile_manager.load_profile()
                init_state = target_ta.propagator.create_initial_state(target_ticker, target_date, current_profile.summary)

                # Pre-graph focus log (category/schema/news bundle) - use both log+progress for UI highlighting
                def _fmt_topk(cands):
                    return ", ".join(
                        [f"{c.get('domain_category','')}:{c.get('score','')}" for c in (cands or [])][:3]
                    )

                topk_str = _fmt_topk(init_state.get("classification_candidates", []))
                c_cnt = len((init_state.get("news_bundle") or {}).get("company_news", []) or [])
                m_cnt = len((init_state.get("news_bundle") or {}).get("macro_news", []) or [])
                evidence_list = init_state.get("classification_evidence", []) or []
                evidence_list = evidence_list if isinstance(evidence_list, list) else []
                ev_pretty = []
                for ev in evidence_list[:8]:
                    src = ev.get("source", "")
                    rule = ev.get("rule", "")
                    matched = ev.get("matched_text", "")
                    ev_pretty.append(f"{src}:{rule} => {matched}")
                ev_msg = "; ".join(ev_pretty) if ev_pretty else "no evidence (fallback to Others)"
                focus_msg = (
                    f"🔥 [CATEGORY FOCUS] asset={init_state.get('asset_type','')} "
                    f"| domain={init_state.get('domain_category','')} "
                    f"| schema={init_state.get('analysis_schema_id','')} "
                    f"| topk=({topk_str}) | news company/macro={c_cnt}/{m_cnt}"
                )
                await websocket.send_json({"type": "log", "level": "error", "message": focus_msg})
                await websocket.send_json({"type": "progress", "step": focus_msg})
                await websocket.send_json({
                    "type": "log",
                    "level": "error",
                    "message": f"[CATEGORY EVIDENCE] {ev_msg}"
                })
                
                # Run
                await websocket.send_json({"type": "log", "message": f"\n[System] Starting analysis for {target_ticker}..."})
                print(f"DEBUG: Invoking graph for {target_ticker}")
                final_state = await target_ta.graph.ainvoke(init_state, run_config)

                # Highlight classification focus (red/error level for UI emphasis)
                focus_asset = final_state.get("asset_type", "")
                focus_domain = final_state.get("domain_category", "")
                focus_schema = final_state.get("analysis_schema_id", "")
                focus_conf = final_state.get("classification_candidates", [])
                topk = final_state.get("classification_candidates", []) or []
                topk_str = ", ".join([f"{c.get('domain_category','')}:{c.get('score','')}" for c in topk[:3]])
                bundle = final_state.get("news_bundle", {}) or {}
                c_cnt = len(bundle.get("company_news", []) or [])
                m_cnt = len(bundle.get("macro_news", []) or [])
                await websocket.send_json({
                    "type": "log",
                    "level": "error",
                    "message": f"🔥 [CATEGORY FOCUS] asset={focus_asset} | domain={focus_domain} | schema={focus_schema} | topk=({topk_str}) | news company/macro={c_cnt}/{m_cnt}"
                })
                await websocket.send_json({"type": "progress", "step": f"[CATEGORY FOCUS] {focus_domain} ({focus_schema})"})
                
                # Process Result
                raw_decision = final_state.get("final_trade_decision", "HOLD")
                import re
                verdict = "HOLD"
                confidence = "Medium"
                if "buy" in raw_decision.lower(): verdict = "BUY"
                elif "sell" in raw_decision.lower(): verdict = "SELL"
                
                conf_match = re.search(r"Confidence:\s*(\w+)", raw_decision, re.IGNORECASE)
                if conf_match: confidence = conf_match.group(1).upper()
                
                accuracy_info = calculate_accuracy(target_ticker, target_date, verdict)
                
                result_payload = {
                    "type": "result",
                    "ticker": target_ticker,        # Added Ticker field
                    "is_primary": is_primary,       # Added Flag
                    "decision": verdict,
                    "confidence": confidence,
                    "reasoning": raw_decision,
                    "report": final_state.get("trader_investment_plan", ""),
                    "accuracy": accuracy_info,
                    "full_state": {
                        "sentiment": final_state.get("sentiment_report", ""),
                        "fundamentals": final_state.get("fundamentals_report", ""),
                        "technical": final_state.get("market_report", ""),
                        "risk": final_state.get("risk_debate_state", {}).get("judge_decision", "")
                    }
                }
                
                await websocket.send_json(result_payload)
                print(f"DEBUG: Sent result for {target_ticker}")

            except Exception as e:
                print(f"Error in run_analysis({target_ticker}): {e}")
                await websocket.send_json({"type": "error", "message": f"Error organizing {target_ticker}: {str(e)}"})


        # 1. Primary Analysis
        await run_analysis(ticker, date_str, is_primary=True)
        
        # 2. Recommendation Phase
        await websocket.send_json({"type": "log", "message": "\n[Discovery] Identifying related opportunities based on your profile..."})
        
        current_profile = profile_manager.load_profile()
        recs = recommender.get_recommendations(ticker, current_profile.summary)
        
        if recs.tickers:
            await websocket.send_json({
                "type": "recommendations",
                "tickers": recs.tickers,
                "reasoning": recs.reasoning
            })
            
            # 3. Loop Analysis
            for rec_ticker in recs.tickers:
                await asyncio.sleep(1) # Breath
                await run_analysis(rec_ticker, date_str, is_primary=False)
        else:
            await websocket.send_json({"type": "log", "message": "[Discovery] No recommendations found."})

        # End of Session
        await websocket.send_json({"type": "done"})

    except Exception as e:
         print(f"Error during analysis session: {e}")
         await websocket.send_json({"type": "error", "message": f"Session Error: {str(e)}"})
    finally:
        await websocket.close()

def calculate_accuracy(ticker: str, date_str: str, decision: str) -> dict:
    """
    Calculate the 5-day return following the analysis date.
    """
    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        
        # If target date is today or future, cannot calculate accuracy
        if target_date.date() >= datetime.now().date():
            return {"calculable": False, "message": "Future date/Today (미래/오늘 날짜)"}

        # Fetch data: target_date to target_date + 7 days (to cover weekends)
        end_date = target_date + timedelta(days=10)
        data = yf.download(ticker, start=target_date, end=end_date, progress=False)
        
        if data.empty:
             return {"calculable": False, "message": "No Data (데이터 없음)"}

        # Get close price on the analysis date (or next trading day)
        # yfinance includes start date.
        if len(data) < 2:
             return {"calculable": False, "message": "Insufficient data (데이터 부족)"}
        
        start_price = data['Close'].iloc[0] # This can be a Series if MultiIndex, ensure scalar
        if hasattr(start_price, 'item'): start_price = start_price.item()

        # Look for price 5 trading days later, or last available
        days_later_idx = 5 if len(data) > 5 else len(data) - 1
        end_price = data['Close'].iloc[days_later_idx]
        if hasattr(end_price, 'item'): end_price = end_price.item()
        
        period_return = ((end_price - start_price) / start_price) * 100
        
        # Simple accuracy check: 
        # Buy -> Return > 0 = Good
        # Sell -> Return < 0 = Good (assuming short or avoiding loss)
        # Hold -> Return near 0? (Tricky, keeping it simple)
        
        is_accurate = False
        decision_lower = decision.lower()
        if "buy" in decision_lower and period_return > 0:
            is_accurate = True
        elif "sell" in decision_lower and period_return < 0:
            is_accurate = True
        elif "hold" in decision_lower and abs(period_return) < 2.0: # Arbitrary threshold
             is_accurate = True
             
        return {
            "calculable": True,
            "start_price": f"{start_price:.2f}",
            "end_price": f"{end_price:.2f}",
            "return_pct": f"{period_return:.2f}%",
            "is_accurate": is_accurate,
            "explanation": f"Price moved {start_price:.2f} -> {end_price:.2f} over {days_later_idx} days (주가 변동)"
        }

    except Exception as e:
        return {"calculable": False, "message": f"Error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=True)
