"""
Aether Bayesian Kernel - Bloomberg Dashboard Server
Provides a lightweight HTTP server to serve the dashboard UI, stream live execution
logs, parse decision audit logs, and trigger backtests.
"""

import http.server
import socketserver
import json
import os
import subprocess
import threading
import sys
import joblib
import time
import yfinance as yf
import re

PORT = 8080
RUNNING_PROCESS = None
LIVE_PROCESS = None
PROCESS_LOCK = threading.Lock()
LIVE_LOCK = threading.Lock()

# Ticker Alias Map & Cache buffers
VALIDATION_CACHE = {}
HISTORY_CACHE = {}

# Security Input Validation Whitelists
TICKER_REGEX = re.compile(r'^[A-Za-z0-9\.\-\^=\:]{1,50}$')
VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]

def resolve_ticker_alias(ticker):
    ticker_upper = ticker.upper().strip()
    if ticker_upper in ["ZOMATO.NS", "ZOMATO"]:
        return "ETERNAL.NS"
    elif ticker_upper == "ZOMATO.BO":
        return "ETERNAL.BO"
    return ticker_upper

class DashboardHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence default server logging in console to keep terminal clean
        pass

    def do_GET(self):
        global RUNNING_PROCESS, LIVE_PROCESS
        
        # Handle CORS
        self.send_response(200)
        
        if self.path == "/" or self.path == "/index.html":
            self.send_header("Content-type", "text/html")
            self.end_headers()
            
            # Read templates/dashboard.html
            html_path = os.path.join(os.path.dirname(__file__), "templates", "dashboard.html")
            if os.path.exists(html_path):
                with open(html_path, "r", encoding="utf-8") as f:
                    self.wfile.write(f.read().encode("utf-8"))
            else:
                self.wfile.write(b"<h1>Error: templates/dashboard.html not found.</h1>")
                
        elif self.path == "/api/log":
            self.send_header("Content-type", "application/json")
            self.end_headers()
            
            audit_file = os.path.join(os.path.dirname(__file__), "logs", "audit", "decision_audit.jsonl")
            logs = []
            if os.path.exists(audit_file):
                try:
                    with open(audit_file, "r", encoding="utf-8") as f:
                        for line in f:
                            cleaned = line.strip()
                            if cleaned:
                                logs.append(json.loads(cleaned))
                except Exception as e:
                    logs = [{"error": str(e)}]
            self.wfile.write(json.dumps(logs).encode("utf-8"))

        elif self.path == "/api/live_log":
            self.send_header("Content-type", "application/json")
            self.end_headers()
            
            live_file = os.path.join(os.path.dirname(__file__), "logs", "audit", "live_audit.jsonl")
            logs = []
            if os.path.exists(live_file):
                try:
                    with open(live_file, "r", encoding="utf-8") as f:
                        for line in f:
                            cleaned = line.strip()
                            if cleaned:
                                logs.append(json.loads(cleaned))
                except Exception as e:
                    logs = [{"error": str(e)}]
            self.wfile.write(json.dumps(logs).encode("utf-8"))
            
        elif self.path == "/api/output":
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            
            log_path = os.path.join(os.path.dirname(__file__), "logs", "dashboard_run.log")
            content = ""
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    content = f"Error reading log file: {str(e)}"
            self.wfile.write(content.encode("utf-8"))

        elif self.path == "/api/live_output":
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            
            log_path = os.path.join(os.path.dirname(__file__), "logs", "live_monitor_run.log")
            content = ""
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    content = f"Error reading log file: {str(e)}"
            self.wfile.write(content.encode("utf-8"))
            
        elif self.path == "/api/status":
            self.send_header("Content-type", "application/json")
            self.end_headers()
            
            with PROCESS_LOCK:
                is_running = RUNNING_PROCESS is not None and RUNNING_PROCESS.poll() is None
            
            self.wfile.write(json.dumps({"running": is_running}).encode("utf-8"))

        elif self.path == "/api/live_status":
            self.send_header("Content-type", "application/json")
            self.end_headers()
            
            with LIVE_LOCK:
                is_running = LIVE_PROCESS is not None and LIVE_PROCESS.poll() is None
            
            self.wfile.write(json.dumps({"running": is_running}).encode("utf-8"))

        elif self.path == "/api/model_info":
            self.send_header("Content-type", "application/json")
            self.end_headers()
            
            try:
                model_path = os.path.join(os.path.dirname(__file__), "models", "global_neural_network.pkl")
                model = joblib.load(model_path)
                
                # Extract shapes, weights, and biases to lists
                model_data = {
                    "weights": [coef.tolist() for coef in model.coefs_],
                    "biases": [bias.tolist() for bias in model.intercepts_],
                    "layer_sizes": [model.coefs_[0].shape[0]] + list(model.hidden_layer_sizes) + [model.coefs_[-1].shape[1]]
                }
            except Exception as e:
                model_data = {"error": f"Failed to load model weights: {str(e)}"}
                
            self.wfile.write(json.dumps(model_data).encode("utf-8"))

        elif self.path.startswith("/api/validate_ticker"):
            self.send_header("Content-type", "application/json")
            self.end_headers()
            
            ticker = "AAPL"
            if "ticker=" in self.path:
                try:
                    ticker = self.path.split("ticker=")[1].split("&")[0]
                except Exception:
                    pass
            
            if not isinstance(ticker, str) or not TICKER_REGEX.match(ticker):
                self.wfile.write(json.dumps({"valid": False, "ticker": ticker, "error": "Invalid ticker format"}).encode("utf-8"))
                return
            
            resolved_ticker = resolve_ticker_alias(ticker)
            
            if resolved_ticker in VALIDATION_CACHE:
                is_valid = VALIDATION_CACHE[resolved_ticker]
            else:
                is_valid = False
                try:
                    t = yf.Ticker(resolved_ticker)
                    hist = t.history(period="1d")
                    is_valid = not hist.empty
                except Exception:
                    pass
                VALIDATION_CACHE[resolved_ticker] = is_valid
                
            self.wfile.write(json.dumps({"valid": is_valid, "ticker": resolved_ticker}).encode("utf-8"))

        elif self.path.startswith("/api/history"):
            self.send_header("Content-type", "application/json")
            self.end_headers()
            
            ticker = "AAPL"
            period = "1mo"
            if "ticker=" in self.path:
                try:
                    ticker = self.path.split("ticker=")[1].split("&")[0]
                except Exception:
                    pass
            if "period=" in self.path:
                try:
                    period = self.path.split("period=")[1].split("&")[0]
                except Exception:
                    pass
            
            if not isinstance(ticker, str) or not TICKER_REGEX.match(ticker) or period not in VALID_PERIODS:
                self.wfile.write(json.dumps({"error": "Invalid ticker or period"}).encode("utf-8"))
                return
            
            ticker = resolve_ticker_alias(ticker)
            
            # Cache look up (expiry 60s)
            cache_key = (ticker, period)
            now = time.time()
            if cache_key in HISTORY_CACHE and (now - HISTORY_CACHE[cache_key]['time']) < 60:
                self.wfile.write(json.dumps(HISTORY_CACHE[cache_key]['data']).encode("utf-8"))
                return

            hist_data = []
            try:
                t = yf.Ticker(ticker)
                
                # Determine interval based on period to allow high granularity charts
                interval = "1d"
                if period == "1d":
                    interval = "5m"
                elif period == "5d":
                    interval = "15m"
                    
                df = t.history(period=period, interval=interval)
                ccy = "INR" if (ticker.endswith(".NS") or ticker.endswith(".BO")) else "USD"
                
                for date, row in df.iterrows():
                    hist_data.append({
                        "timestamp": str(date),
                        "price": float(row['Close']),
                        "features": {"v": 0.0, "r": 0.0, "c": 0.0, "a": 0.0, "d": 0.0, "l": 0.0, "b": 0.0},
                        "dominant_regime": 0,
                        "regime_confidence": 0.0,
                        "entropy": 0.0,
                        "mlp_conviction": 0.5,
                        "signal": "HOLD",
                        "holdings": 0.0,
                        "equity": 100000.0,
                        "cash": 100000.0,
                        "market_status": "HISTORICAL PRE-LOAD",
                        "currency": ccy
                    })
            except Exception as e:
                print(f"[API ERROR] History fetch failed for {ticker}: {e}")
                pass
                
            HISTORY_CACHE[cache_key] = {'time': now, 'data': hist_data}
            self.wfile.write(json.dumps(hist_data).encode("utf-8"))
            
        else:
            self.send_response(404)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"404 Not Found")

    def do_POST(self):
        global RUNNING_PROCESS, LIVE_PROCESS
        
        if self.path == "/api/run":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length).decode("utf-8")
            
            try:
                params = json.loads(post_data) if post_data else {}
            except Exception:
                params = {}
                
            run_type = params.get("type", "master")
            if run_type not in ["master", "sandbox"]:
                run_type = "master"
                
            raw_ticker = params.get("ticker", "AAPL")
            if not isinstance(raw_ticker, str) or not TICKER_REGEX.match(raw_ticker):
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid ticker format"}).encode("utf-8"))
                return
            ticker = resolve_ticker_alias(raw_ticker)
            
            # Form command
            python_exec = sys.executable
            if run_type == "master":
                cmd = [python_exec, "main.py"]
            else:
                cmd = [python_exec, "src/research/sandbox.py", "--ticker", ticker]
                
            # Direct output logs
            os.makedirs("logs", exist_ok=True)
            log_path = os.path.join(os.path.dirname(__file__), "logs", "dashboard_run.log")
            
            # Prevent concurrent runs
            with PROCESS_LOCK:
                if RUNNING_PROCESS is not None and RUNNING_PROCESS.poll() is None:
                    try:
                        RUNNING_PROCESS.terminate()
                        RUNNING_PROCESS.wait(timeout=2)
                    except Exception:
                        pass
                
                # Clear previous run logs
                try:
                    with open(log_path, "w", encoding="utf-8") as f:
                        f.write(f"--- Launching Backtest: {' '.join(cmd)} ---\n")
                except Exception:
                    pass
                
                # Launch new process
                log_file = open(log_path, "a", encoding="utf-8")
                RUNNING_PROCESS = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=log_file,
                    env=os.environ.copy(),
                    bufsize=1,
                    universal_newlines=True
                )
            
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "launched", "type": run_type}).encode("utf-8"))
            
        elif self.path == "/api/kill":
            with PROCESS_LOCK:
                if RUNNING_PROCESS is not None and RUNNING_PROCESS.poll() is None:
                    try:
                        RUNNING_PROCESS.terminate()
                        RUNNING_PROCESS.wait(timeout=2)
                        status = "terminated"
                    except Exception as e:
                        status = f"error: {str(e)}"
                else:
                    status = "not_running"
            
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": status}).encode("utf-8"))

        elif self.path == "/api/run_live":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length).decode("utf-8")
            
            try:
                params = json.loads(post_data) if post_data else {}
            except Exception:
                params = {}
                
            raw_ticker = params.get("ticker", "AAPL")
            if not isinstance(raw_ticker, str) or not TICKER_REGEX.match(raw_ticker):
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid ticker format"}).encode("utf-8"))
                return
            ticker = resolve_ticker_alias(raw_ticker)
            
            try:
                interval_int = int(params.get("interval", 10))
                if not (1 <= interval_int <= 3600):
                    raise ValueError()
                interval = str(interval_int)
            except (ValueError, TypeError):
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid interval value"}).encode("utf-8"))
                return

            try:
                allocation_float = float(params.get("allocation", 100000.0))
                if not (0.0 < allocation_float <= 1e12):
                    raise ValueError()
                allocation = f"{allocation_float:.2f}"
            except (ValueError, TypeError):
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid allocation value"}).encode("utf-8"))
                return
            
            # Form command
            python_exec = sys.executable
            cmd = [python_exec, "scripts/live_monitor.py", "--ticker", ticker, "--interval", interval, "--allocation", allocation]
                
            # Direct output logs
            os.makedirs("logs", exist_ok=True)
            log_path = os.path.join(os.path.dirname(__file__), "logs", "live_monitor_run.log")
            
            # Prevent concurrent runs
            with LIVE_LOCK:
                if LIVE_PROCESS is not None and LIVE_PROCESS.poll() is None:
                    try:
                        LIVE_PROCESS.terminate()
                        LIVE_PROCESS.wait(timeout=2)
                    except Exception:
                        pass
                
                # Clear previous run logs
                try:
                    with open(log_path, "w", encoding="utf-8") as f:
                        f.write(f"--- Launching Live Observation Feed: {' '.join(cmd)} ---\n")
                except Exception:
                    pass
                
                # Launch new process
                log_file = open(log_path, "a", encoding="utf-8")
                LIVE_PROCESS = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=log_file,
                    env=os.environ.copy(),
                    bufsize=1,
                    universal_newlines=True
                )
            
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "launched", "ticker": ticker}).encode("utf-8"))
            
        elif self.path == "/api/kill_live":
            with LIVE_LOCK:
                if LIVE_PROCESS is not None and LIVE_PROCESS.poll() is None:
                    try:
                        LIVE_PROCESS.terminate()
                        LIVE_PROCESS.wait(timeout=2)
                        status = "terminated"
                    except Exception as e:
                        status = f"error: {str(e)}"
                else:
                    status = "not_running"
            
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": status}).encode("utf-8"))
            
        else:
            self.send_response(404)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"404 Not Found")

class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

def run_server():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), DashboardHandler)
    print(f"\n[DASHBOARD SERVER] Running Bloomberg-Terminal Dashboard on: http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[DASHBOARD SERVER] Shutting down...")
        server.shutdown()

if __name__ == "__main__":
    run_server()
