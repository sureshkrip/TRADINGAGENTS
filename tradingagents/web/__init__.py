"""HTTP service layer around :class:`TradingAgentsGraph`.

Exposes the trading pipeline as a small FastAPI app so the framework can be
deployed as a normal web service (with a URL + health check) on platforms like
Coolify, instead of only as an interactive CLI. See ``server.py``.
"""
