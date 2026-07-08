"""Flask Frontend Server — serves frontend/ + proxies /api/* to FastAPI."""
from flask import Flask, request, Response, send_from_directory
import requests, os
from pathlib import Path

BACKEND_URL = os.environ.get("GEOSCAN_BACKEND_URL", "http://127.0.0.1:8766")
_env = os.environ.get("FRONTEND_DIR")
FRONTEND_DIR = Path(_env) if _env else (Path(__file__).resolve().parent.parent / "frontend")
PORT = int(os.environ.get("FLASK_PORT", "8765"))

app = Flask(__name__, static_folder=None)

@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(str(FRONTEND_DIR), filename)

@app.route("/api/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
def api_proxy(path):
    target_url = f"{BACKEND_URL}/api/{path}"
    if request.query_string:
        target_url += "?" + request.query_string.decode("utf-8")
    body = request.get_data()
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length", "connection")}
    try:
        resp = requests.request(method=request.method, url=target_url,
            headers=headers, data=body, timeout=120, stream=True)
        excluded = {"content-encoding", "content-length", "transfer-encoding", "connection"}
        rh = [(k, v) for k, v in resp.headers.items() if k.lower() not in excluded]
        return Response(resp.content, status=resp.status_code, headers=rh)
    except requests.exceptions.ConnectionError:
        return Response('{"detail":"Backend unreachable"}', status=502, content_type="application/json")
    except requests.exceptions.Timeout:
        return Response('{"detail":"Backend timeout"}', status=504, content_type="application/json")

if __name__ == "__main__":
    print(f"Flask on :{PORT} -> {BACKEND_URL}")
    app.run(host="0.0.0.0", port=PORT, debug=True)
