"""
Aether Bayesian Kernel - Dashboard Server Validation Test Suite
Validates that GET and POST request parameters are properly checked,
preventing command injection or argument injection vulnerabilities.
"""

import sys
import os
import json
import io
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dashboard_server import DashboardHandler

class MockWfile(io.BytesIO):
    def write(self, b):
        super().write(b)

class MockRequest:
    def makefile(self, *args, **kwargs):
        return io.BytesIO()

class MockServer:
    def __init__(self):
        self.server_address = ('127.0.0.1', 8080)

class TestHandler(DashboardHandler):
    def __init__(self, request_body=b"", path="/", method="GET"):
        self.rfile = io.BytesIO(request_body)
        self.wfile = MockWfile()
        self.request = MockRequest()
        self.server = MockServer()
        self.headers = {}
        self.command = method
        self.path = path
        self.response_code = None
        self.response_headers = {}

    def send_response(self, code, message=None):
        self.response_code = code

    def send_header(self, keyword, value):
        self.response_headers[keyword] = value

    def end_headers(self):
        pass


def test_validate_ticker_valid():
    handler = TestHandler(path="/api/validate_ticker?ticker=AAPL")
    from dashboard_server import VALIDATION_CACHE
    VALIDATION_CACHE["AAPL"] = True
    
    handler.do_GET()
    response = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert response["valid"] is True
    assert response["ticker"] == "AAPL"


def test_validate_ticker_invalid():
    handler = TestHandler(path="/api/validate_ticker?ticker=AAPL;rm -rf /")
    handler.do_GET()
    response = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert response["valid"] is False
    assert "error" in response


def test_history_invalid():
    handler = TestHandler(path="/api/history?ticker=AAPL&period=invalid_period")
    handler.do_GET()
    response = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert "error" in response


def test_run_invalid_ticker():
    body = json.dumps({"ticker": "AAPL;rm -rf /", "type": "sandbox"}).encode("utf-8")
    handler = TestHandler(request_body=body, path="/api/run", method="POST")
    handler.headers["Content-Length"] = str(len(body))
    handler.do_POST()
    assert handler.response_code == 400
    response = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert "error" in response


def test_run_live_invalid_ticker():
    body = json.dumps({"ticker": "AAPL;rm -rf /", "interval": 10, "allocation": 100000.0}).encode("utf-8")
    handler = TestHandler(request_body=body, path="/api/run_live", method="POST")
    handler.headers["Content-Length"] = str(len(body))
    handler.do_POST()
    assert handler.response_code == 400
    response = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert "error" in response


def test_run_live_invalid_interval():
    body = json.dumps({"ticker": "AAPL", "interval": 9999, "allocation": 100000.0}).encode("utf-8")
    handler = TestHandler(request_body=body, path="/api/run_live", method="POST")
    handler.headers["Content-Length"] = str(len(body))
    handler.do_POST()
    assert handler.response_code == 400
    response = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert "error" in response


def test_run_live_invalid_allocation():
    body = json.dumps({"ticker": "AAPL", "interval": 10, "allocation": -100.0}).encode("utf-8")
    handler = TestHandler(request_body=body, path="/api/run_live", method="POST")
    handler.headers["Content-Length"] = str(len(body))
    handler.do_POST()
    assert handler.response_code == 400
    response = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert "error" in response
