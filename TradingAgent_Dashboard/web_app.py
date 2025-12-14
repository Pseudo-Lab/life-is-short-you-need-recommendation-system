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

@app.websocket("/ws/analyze")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_text()
        request_data = json.loads(data)
        ticker = request_data.get("ticker", "").upper()
        date_str = request_data.get("date", "")
        
        if not ticker or not date_str:
            await websocket.send_json({"type": "error", "message": "Missing ticker or date (티커 또는 날짜 누락)"})
            return

        await websocket.send_json({"type": "log", "message": f"Initializing analysis for {ticker} on {date_str}..."})

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
                self.cost += (len(token)/4.0) * 0.00003
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

        # --- Graph Execution ---
        ta = TradingAgentsGraph(debug=False, config=DEFAULT_CONFIG.copy())
        ta.ticker = ticker
        
        callback = WebSocketCallback(websocket)
        run_config = ta.propagator.get_graph_args()
        run_config["callbacks"] = [callback]
        run_config["recursion_limit"] = 150  # Increase limit for complex debates

        init_agent_state = ta.propagator.create_initial_state(ticker, date_str)
        
        # Single Execution
        print("DEBUG: Starting graph.ainvoke...")
        try:
            final_state = await ta.graph.ainvoke(init_agent_state, run_config)
            print("DEBUG: graph.ainvoke completed.")
        except Exception as e:
            print(f"ERROR: graph.ainvoke failed: {e}")
            raise e
        
        await websocket.send_json({"type": "log", "message": "\n[System] Analysis complete. Generating final report..."})
        
        # Process and Send Result
        print("DEBUG: Processing final state...")
        print(f"DEBUG: final_state keys: {final_state.keys()}")
        
        raw_decision = final_state.get("final_trade_decision", "HOLD")
        print(f"DEBUG: raw_decision length: {len(str(raw_decision))}")
        
        # Parse Decision: "BUY" / "SELL" / "HOLD" and Confidence
        import re
        verdict = "HOLD"
        confidence = "Medium"
        reasoning = raw_decision
        
        if "buy" in raw_decision.lower(): verdict = "BUY"
        elif "sell" in raw_decision.lower(): verdict = "SELL"
        
        # Try to extract confidence if explicitly stated
        conf_match = re.search(r"Confidence:\s*(\w+)", raw_decision, re.IGNORECASE)
        if conf_match:
            confidence = conf_match.group(1).upper()
        
        print("DEBUG: Calculating accuracy...")
        try:
            accuracy_info = calculate_accuracy(ticker, date_str, verdict)
            print(f"DEBUG: Accuracy info: {accuracy_info}")
        except Exception as e:
            print(f"ERROR: calculate_accuracy failed: {e}")
            accuracy_info = {"calculable": False, "message": "Error calculating accuracy"}

        print("DEBUG: Constructing result JSON...")
        result_payload = {
            "type": "result",
            "decision": verdict,             # Short verdict for Big Title
            "confidence": confidence,        # Parsed confidence
            "reasoning": reasoning,          # Full text for sub-text
            "report": final_state.get("trader_investment_plan", "No report available."),
            "accuracy": accuracy_info,
            "full_state": {
                "sentiment": final_state.get("sentiment_report", "No data"),
                "fundamentals": final_state.get("fundamentals_report", "No data"),
                "technical": final_state.get("market_report", "No data"),
                "risk": final_state.get("risk_debate_state", {}).get("judge_decision", "No data")
            }
        }
        
        print("DEBUG: Sending result JSON...")
        await websocket.send_json(result_payload)
        print("DEBUG: Result JSON sent.")
        
    except Exception as e:
         print(f"Error during analysis: {e}") # Print to server terminal
         await websocket.send_json({"type": "error", "message": f"Server Error: {str(e)}"})
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
