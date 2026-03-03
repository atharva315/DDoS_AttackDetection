import urllib.request

u = 'http://127.0.0.1:5000/rescan_models'
try:
    with urllib.request.urlopen(u, timeout=5) as r:
        print(r.read().decode())
except Exception as e:
    print('ERROR', e)
