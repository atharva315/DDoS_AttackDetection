import requests
import json

try:
    r = requests.post('http://127.0.0.1:5000/rescan_models', timeout=5)
    try:
        print(r.json())
    except Exception:
        print('Non-JSON response:', r.text)
except Exception as e:
    print('Request failed:', e)
