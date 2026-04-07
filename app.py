# Core imports
import glob
import json
import pandas as pd
import numpy as np
import time
import socket
import struct
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
import joblib
import threading
import sys
from scapy.all import sniff


# Discover available model files (*.pkl) in the project root
MODEL_FILES = { }
for p in glob.glob("*.pkl"):
    name = p.rsplit('.', 1)[0]
    MODEL_FILES[name] = p

# Load a default model if available, else None
current_model_name = None
model = None
model_expected_features = None
model_lock = threading.Lock()
if 'logistic_DDoS' in MODEL_FILES:
    current_model_name = 'logistic_DDoS'
    loaded = joblib.load(MODEL_FILES[current_model_name])
    # support payloads saved as {'pipeline': pipeline, 'feature_names': [...]}
    if isinstance(loaded, dict) and 'pipeline' in loaded:
        model = loaded['pipeline']
        model_expected_features = loaded.get('feature_names')
        # try to obtain feature names from pipeline final estimator if available
        try:
            if model_expected_features is None and hasattr(model, 'named_steps') and 'clf' in model.named_steps and hasattr(model.named_steps['clf'], 'feature_names_in_'):
                model_expected_features = list(model.named_steps['clf'].feature_names_in_)
        except Exception:
            pass
    else:
        model = loaded
        # try to get expected features from a pipeline-like object
        try:
            if hasattr(model, 'named_steps') and 'clf' in model.named_steps and hasattr(model.named_steps['clf'], 'feature_names_in_'):
                model_expected_features = list(model.named_steps['clf'].feature_names_in_)
        except Exception:
            pass
elif MODEL_FILES:
    # pick the first available model
    current_model_name = list(MODEL_FILES.keys())[0]
    loaded = joblib.load(MODEL_FILES[current_model_name])
    if isinstance(loaded, dict) and 'pipeline' in loaded:
        model = loaded['pipeline']
        model_expected_features = loaded.get('feature_names')
        try:
            if model_expected_features is None and hasattr(model, 'named_steps') and 'clf' in model.named_steps and hasattr(model.named_steps['clf'], 'feature_names_in_'):
                model_expected_features = list(model.named_steps['clf'].feature_names_in_)
        except Exception:
            pass
    else:
        model = loaded
        try:
            if hasattr(model, 'named_steps') and 'clf' in model.named_steps and hasattr(model.named_steps['clf'], 'feature_names_in_'):
                model_expected_features = list(model.named_steps['clf'].feature_names_in_)
        except Exception:
            pass
else:
    print("Warning: no .pkl model files found. Prediction will be disabled until a model is added.")



# Define constants and global variables
TIME_WINDOW = 10  # Time window in seconds
PACKET_THRESHOLD = 200  # Threshold for packets per second
packet_count = 0
byte_count = 0
pkt_size_list = []
timestamps = []
attack_result = "No Attack Detected"  # Default state
logs = []  # Store network logs
stop_sniffer = False  # Flag to stop the packet sniffer thread
metrics = []  # time-series metrics: {'t': timestamp, 'pkt_rate': x, 'byte_rate': y}
METRICS_MAX_LEN = 360

# Feature names used by training and evaluation
MODEL_FEATURE_NAMES = [
    'PKT_SIZE', 'NUMBER_OF_PKT', 'NUMBER_OF_BYTE', 'PKT_DELAY_NODE',
    'PKT_RATE', 'BYTE_RATE', 'PKT_AVG_SIZE', 'UTILIZATION', 'PKT_TYPE_ENCODED'
]


def generate_synthetic_eval(n=300, seed=123):
    """Generate a small synthetic dataset for quick model evaluation in the UI.
    This mirrors the lightweight generator used in the training script but keeps
    values constrained so evaluation is fast and deterministic-ish.
    Returns: pandas.DataFrame with columns MODEL_FEATURE_NAMES and a label column.
    """
    rng = np.random.RandomState(seed)
    PKT_SIZE = rng.normal(600, 300, n).clip(40, 1500)
    NUMBER_OF_PKT = rng.poisson(20, n).clip(1, None)
    NUMBER_OF_BYTE = (PKT_SIZE * NUMBER_OF_PKT).astype(float)
    PKT_DELAY_NODE = rng.exponential(0.05, n)
    PKT_RATE = rng.normal(5, 10, n).clip(0, None)
    BYTE_RATE = PKT_RATE * PKT_SIZE
    PKT_AVG_SIZE = PKT_SIZE
    UTILIZATION = (NUMBER_OF_PKT / 100.0)
    PKT_TYPE_ENCODED = rng.choice([0,1,2,3], size=n, p=[0.2,0.2,0.1,0.5])

    # Label heuristic similar to training generator for evaluation purposes
    label = ((PKT_RATE > 100) | (rng.rand(n) < 0.06)).astype(int)

    df = pd.DataFrame({
        'PKT_SIZE': PKT_SIZE,
        'NUMBER_OF_PKT': NUMBER_OF_PKT,
        'NUMBER_OF_BYTE': NUMBER_OF_BYTE,
        'PKT_DELAY_NODE': PKT_DELAY_NODE,
        'PKT_RATE': PKT_RATE,
        'BYTE_RATE': BYTE_RATE,
        'PKT_AVG_SIZE': PKT_AVG_SIZE,
        'UTILIZATION': UTILIZATION,
        'PKT_TYPE_ENCODED': PKT_TYPE_ENCODED,
        'label': label
    })
    return df


@app.route('/model_metrics', methods=['GET'])
def model_metrics():
    """Return simple per-model precision/recall/f1 computed on a small synthetic set.
    This is intended for UI display only and is not an authoritative benchmark.
    """
    results = {}
    # Generate a small eval set once
    eval_df = generate_synthetic_eval(n=300)
    X_eval = eval_df[MODEL_FEATURE_NAMES]
    y_true = eval_df['label'].values

    for name, path in MODEL_FILES.items():
        try:
            loaded = joblib.load(path)
            mdl = loaded['pipeline'] if isinstance(loaded, dict) and 'pipeline' in loaded else loaded
            # Align features if model expects a specific order
            try:
                expected = None
                if isinstance(loaded, dict):
                    expected = loaded.get('feature_names')
                if expected is None and hasattr(mdl, 'named_steps') and 'clf' in mdl.named_steps and hasattr(mdl.named_steps['clf'], 'feature_names_in_'):
                    expected = list(mdl.named_steps['clf'].feature_names_in_)
                if expected is None and hasattr(mdl, 'feature_names_in_'):
                    expected = list(mdl.feature_names_in_)
                if expected:
                    X_in = X_eval.reindex(columns=expected, fill_value=0)
                else:
                    X_in = X_eval
            except Exception:
                X_in = X_eval

            # Predict
            try:
                y_pred = mdl.predict(X_in)
            except Exception:
                y_pred = mdl.predict(X_in.values)

            p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
            results[name] = {'precision': float(p), 'recall': float(r), 'f1': float(f)}
        except Exception as e:
            results[name] = {'error': str(e)}

    return jsonify(results)

# Helper function to convert IP to integer
def ip_to_int(ip):
    return struct.unpack("!I", socket.inet_aton(ip))[0]

# Function to encode packet type manually
def encode_pkt_type(pkt_type):
    if pkt_type == 6:  # TCP protocol
        return 3
    elif pkt_type == 0:  # ACK
        return 0
    elif pkt_type == 1:  # CBR
        return 1
    elif pkt_type == 2:  # Ping
        return 2
    else:
        return -1  # Unknown type

# Function to print logs every 10 seconds
def print_logs():
    global logs
    while True:
        time.sleep(TIME_WINDOW)
        if logs:
            print("\n=== Network Logs ===")
            for log in logs:
                print(log)
            logs = []  # Clear logs after printing
        else:
            print("\nNo packets captured in the last 10 seconds.")

# Packet processing function
def packet_sniffer():
    global packet_count, byte_count, pkt_size_list, timestamps, attack_result, logs, stop_sniffer, model

    def process_packet(packet):
        global packet_count, byte_count, pkt_size_list, timestamps, attack_result, logs, stop_sniffer, model

        try:
            # Extract features from the packet
            src_ip = ip_to_int(packet[1].src)
            dst_ip = ip_to_int(packet[1].dst)
            pkt_size = len(packet)
            pkt_type = packet[1].proto  # Protocol type (e.g., TCP=6, UDP=17)

            # Encode PKT_TYPE manually
            pkt_type_encoded = encode_pkt_type(pkt_type)

            # Update packet stats
            packet_count += 1
            byte_count += pkt_size

            # Track packet sizes and timestamps
            pkt_size_list.append(pkt_size)
            timestamps.append(time.time())

            # Remove old packets outside the time window
            while timestamps and timestamps[0] < time.time() - TIME_WINDOW:
                timestamps.pop(0)
                pkt_size_list.pop(0)

            # Check for DDoS condition: packets exceed PACKET_THRESHOLD
            if len(timestamps) > PACKET_THRESHOLD:
                attack_result = "DDoS Attack Detected: THRESHOLD EXCEEDED !"
                print(attack_result)
                # Stop sniffer but don't exit the whole process
                stop_sniffer = True
                return

            # Calculate derived features
            pkt_rate = len(timestamps) / TIME_WINDOW
            byte_rate = sum(pkt_size_list) / TIME_WINDOW
            pkt_avg_size = byte_count / packet_count if packet_count > 0 else 0
            utilization = packet_count / 100  # Example calculation

            # Record metrics for UI charting (keep bounded length)
            try:
                utilization = utilization if 'utilization' in locals() else (packet_count / 100)
                metrics.append({'t': time.time(), 'pkt_rate': pkt_rate, 'byte_rate': byte_rate, 'utilization': utilization})
                if len(metrics) > METRICS_MAX_LEN:
                    metrics.pop(0)
            except Exception:
                pass

            # Feature vector for prediction
            features = pd.DataFrame([{
                'SRC_ADD': src_ip,
                'DES_ADD': dst_ip,
                'PKT_SIZE': pkt_size,
                'NUMBER_OF_PKT': packet_count,
                'NUMBER_OF_BYTE': byte_count,
                'PKT_DELAY_NODE': 0,  # Placeholder
                'PKT_RATE': pkt_rate,
                'BYTE_RATE': byte_rate,
                'PKT_AVG_SIZE': pkt_avg_size,
                'UTILIZATION': utilization,
                'PKT_TYPE_ENCODED': pkt_type_encoded
            }])

            # Prepare features to match the model's expected feature names
            # Many saved sklearn models expose 'feature_names_in_' after fit. Use it if available.
            try:
                if model_expected_features:
                    expected_features = list(model_expected_features)
                elif model is not None and hasattr(model, 'feature_names_in_'):
                    expected_features = list(model.feature_names_in_)
                else:
                    # Fallback to the default expected features used by the training script
                    expected_features = [
                        'PKT_SIZE', 'NUMBER_OF_PKT', 'NUMBER_OF_BYTE', 'PKT_DELAY_NODE',
                        'PKT_RATE', 'BYTE_RATE', 'PKT_AVG_SIZE', 'UTILIZATION', 'PKT_TYPE_ENCODED'
                    ]

                # Ensure all expected features are present in the features DataFrame
                missing = [f for f in expected_features if f not in features.columns]
                if missing:
                    # Can't predict if required features are missing; log and skip prediction
                    raise ValueError(f"Missing features for model prediction: {missing}")

                # Select and reorder columns to match expected feature order
                model_input = features[expected_features].copy()

                # Predict using the currently loaded model (if any). If the model is a pipeline which
                # includes preprocessing (StandardScaler), it will handle scaling internally.
                if model is None:
                    attack_result = "No model loaded - cannot predict"
                else:
                    with model_lock:
                        try:
                            # Some sklearn estimators accept DataFrame directly; otherwise convert to numpy
                            try:
                                prediction = model.predict(model_input)
                            except Exception:
                                prediction = model.predict(model_input.values)
                            if prediction[0] == 1:
                                attack_result = "DDoS Attack Detected"
                            else:
                                attack_result = "No Attack Detected"
                        except Exception as e:
                            attack_result = f"Model prediction error: {e}"
            except Exception as e:
                # Any issue preparing features should not crash the sniffer — record the error as part of the result
                attack_result = f"Model prediction error: {e}"

            # Add log entry
            logs.append({
                "SRC_ADD": packet[1].src,
                "DES_ADD": packet[1].dst,
                "PKT_SIZE": pkt_size,
                "PKT_TYPE": pkt_type_encoded,  # Log encoded value
                "RESULT": attack_result
            })

        except Exception as e:
            print(f"Error processing packet: {e}")

    # Sniff packets until stopped
    try:
        sniff(filter="ip", prn=process_packet, store=False)
    except Exception as e:
        print(f"Sniffer error: {e}")
    if stop_sniffer:
        print("Packet sniffer stopped.")
    return


@app.route('/metrics', methods=['GET'])
def get_metrics():
    # Return the last METRICS_MAX_LEN metrics for charting
    return jsonify({'metrics': metrics[-METRICS_MAX_LEN:]})


@app.route('/rescan_models', methods=['POST', 'GET'])
def rescan_models():
    """Rescan .pkl model files in the project root and update MODEL_FILES mapping.
    Returns the updated list of model names and the currently selected model.
    """
    global MODEL_FILES, current_model_name, model
    MODEL_FILES = {}
    for p in glob.glob("*.pkl"):
        name = p.rsplit('.', 1)[0]
        MODEL_FILES[name] = p
    # If current model not present, switch to first available
    if current_model_name not in MODEL_FILES:
        if MODEL_FILES:
            current_model_name = list(MODEL_FILES.keys())[0]
            try:
                with model_lock:
                    loaded = joblib.load(MODEL_FILES[current_model_name])
                    if isinstance(loaded, dict) and 'pipeline' in loaded:
                        model = loaded['pipeline']
                        model_expected_features = loaded.get('feature_names')
                        try:
                            if model_expected_features is None and hasattr(model, 'named_steps') and 'clf' in model.named_steps and hasattr(model.named_steps['clf'], 'feature_names_in_'):
                                model_expected_features = list(model.named_steps['clf'].feature_names_in_)
                        except Exception:
                            pass
                    else:
                        model = loaded
                        try:
                            if hasattr(model, 'named_steps') and 'clf' in model.named_steps and hasattr(model.named_steps['clf'], 'feature_names_in_'):
                                model_expected_features = list(model.named_steps['clf'].feature_names_in_)
                        except Exception:
                            pass
            except Exception as e:
                print(f"Could not load model {current_model_name}: {e}")
        else:
            current_model_name = None
            model = None
    return jsonify({'available_models': list(MODEL_FILES.keys()), 'current_model': current_model_name})


@app.route('/config', methods=['GET'])
def get_config():
    # Expose simple config values for frontend (read-only)
    try:
        pps_threshold = PACKET_THRESHOLD / TIME_WINDOW if TIME_WINDOW else PACKET_THRESHOLD
    except Exception:
        pps_threshold = None
    return jsonify({
        'TIME_WINDOW': TIME_WINDOW,
        'PACKET_THRESHOLD': PACKET_THRESHOLD,
        'PACKETS_PER_SECOND_THRESHOLD': pps_threshold
    })

# Thread handles (created on demand)
sniffer_thread = None

# Run the log printer in a background thread (always running)
log_printer_thread = threading.Thread(target=print_logs, daemon=True)
log_printer_thread.start()

def start_sniffer():
    global sniffer_thread, stop_sniffer
    if sniffer_thread is None or not sniffer_thread.is_alive():
        stop_sniffer = False
        sniffer_thread = threading.Thread(target=packet_sniffer, daemon=True)
        sniffer_thread.start()
        print("Packet sniffer started.")
        return True
    return False

def stop_sniffer_thread():
    global stop_sniffer
    stop_sniffer = True
    return True

# Flask API: list available models
@app.route('/models', methods=['GET'])
def list_models():
    return jsonify({
        'available_models': list(MODEL_FILES.keys()),
        'current_model': current_model_name
    })

# Change currently used model
@app.route('/select_model', methods=['POST'])
def select_model():
    global model, current_model_name
    data = request.get_json() or {}
    name = data.get('model_name')
    if not name or name not in MODEL_FILES:
        return jsonify({'error': 'model not found', 'available': list(MODEL_FILES.keys())}), 400
    path = MODEL_FILES[name]
    try:
        with model_lock:
            loaded = joblib.load(path)
            if isinstance(loaded, dict) and 'pipeline' in loaded:
                model = loaded['pipeline']
                model_expected_features = loaded.get('feature_names')
            else:
                model = loaded
            current_model_name = name
        return jsonify({'status': 'ok', 'current_model': current_model_name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def predict_features(features_df):
    """Helper: given a pandas DataFrame with feature columns, align to model and predict.
    Returns: (pred_label:int, message:str)
    Also appends to logs and metrics.
    """
    global attack_result, logs, metrics
    if model is None:
        return None, 'No model loaded'

    # Determine expected features
    expected = None
    try:
        # If model_expected_features was set when loading, use it
        if model_expected_features:
            expected = list(model_expected_features)
    except Exception:
        expected = None
    try:
        if expected is None and hasattr(model, 'named_steps') and 'clf' in model.named_steps and hasattr(model.named_steps['clf'], 'feature_names_in_'):
            expected = list(model.named_steps['clf'].feature_names_in_)
    except Exception:
        pass
    try:
        if expected is None and hasattr(model, 'feature_names_in_'):
            expected = list(model.feature_names_in_)
    except Exception:
        pass

    if expected is None:
        expected = [
            'PKT_SIZE', 'NUMBER_OF_PKT', 'NUMBER_OF_BYTE', 'PKT_DELAY_NODE',
            'PKT_RATE', 'BYTE_RATE', 'PKT_AVG_SIZE', 'UTILIZATION', 'PKT_TYPE_ENCODED'
        ]

    missing = [f for f in expected if f not in features_df.columns]
    if missing:
        return None, f"Missing features: {missing}"

    model_input = features_df[expected].copy()

    with model_lock:
        try:
            try:
                pred = model.predict(model_input)
            except Exception:
                pred = model.predict(model_input.values)
        except Exception as e:
            return None, f'Prediction error: {e}'

    label = int(pred[0])

    # Append a lightweight metric point and log
    try:
        pkt_rate = float(model_input.get('PKT_RATE', pd.Series([0])).iloc[0])
    except Exception:
        pkt_rate = 0.0
    try:
        byte_rate = float(model_input.get('BYTE_RATE', pd.Series([0])).iloc[0])
    except Exception:
        byte_rate = 0.0
    metrics.append({'t': time.time(), 'pkt_rate': pkt_rate, 'byte_rate': byte_rate, 'utilization': float(model_input.get('UTILIZATION', pd.Series([0])).iloc[0])})
    if len(metrics) > METRICS_MAX_LEN:
        metrics.pop(0)

    try:
        logs.append({
            'SRC_ADD': str(features_df.get('SRC_ADD', pd.Series(['-'])).iloc[0]),
            'DES_ADD': str(features_df.get('DES_ADD', pd.Series(['-'])).iloc[0]),
            'PKT_SIZE': int(features_df.get('PKT_SIZE', pd.Series([0])).iloc[0]),
            'PKT_TYPE': int(features_df.get('PKT_TYPE_ENCODED', pd.Series([0])).iloc[0]),
            'RESULT': 'DDoS Attack Detected' if label == 1 else 'No Attack Detected'
        })
        # Keep logs bounded
        if len(logs) > 2000:
            logs.pop(0)
    except Exception:
        pass

    return label, ('DDoS Attack Detected' if label == 1 else 'No Attack Detected')


@app.route('/simulate_sample', methods=['POST'])
def simulate_sample():
    """Accept a JSON body with feature values matching model features and run a prediction.
    Example JSON: {"PKT_SIZE":400, "NUMBER_OF_PKT":5, ...}
    Returns prediction result and appended log entry.
    """
    data = request.get_json() or {}
    if not data:
        return jsonify({'error': 'no data provided'}), 400
    # Build DataFrame with at least the MODEL_FEATURE_NAMES
    try:
        df = pd.DataFrame([data])
    except Exception as e:
        return jsonify({'error': f'bad data: {e}'}), 400

    label, msg = predict_features(df)
    if label is None:
        return jsonify({'error': msg}), 400
    return jsonify({'prediction': int(label), 'message': msg})


@app.route('/simulate_attack', methods=['POST', 'GET'])
def simulate_attack():
    """Simulate a batch of synthetic samples at different intensities.
    Query params or JSON body: intensity: low|medium|high, count: int
    Returns a summary of detections injected into logs/metrics.
    """
    # read params
    data = request.get_json() or {}
    intensity = (data.get('intensity') or request.args.get('intensity') or 'low').lower()
    count = int(data.get('count') or request.args.get('count') or 10)
    # map intensity to count multiplier or attack-like feature bias
    if intensity == 'low':
        attack_prob = 0.1
    elif intensity == 'medium':
        attack_prob = 0.4
    elif intensity == 'high':
        attack_prob = 0.95
    else:
        attack_prob = 0.1

    injected = 0
    detected = 0
    details = []

    rng = np.random.RandomState(int(time.time()) % 2**32)
    for i in range(count):
        # make a sample biased by intensity
        pkt_size = int(np.clip(rng.normal(600 if rng.rand() > attack_prob else 1000, 200), 40, 1500))
        pkt_count = int(np.clip(rng.poisson(10 if rng.rand() > attack_prob else 200), 1, None))
        pkt_rate = float(np.clip(rng.normal(5 if rng.rand() > attack_prob else 800, 50), 0, None))
        byte_rate = pkt_rate * pkt_size
        util = float(np.clip(pkt_count / 100.0, 0, 1))
        sample = {
            'PKT_SIZE': pkt_size,
            'NUMBER_OF_PKT': pkt_count,
            'NUMBER_OF_BYTE': pkt_size * pkt_count,
            'PKT_DELAY_NODE': float(abs(rng.normal(0.02, 0.01))),
            'PKT_RATE': pkt_rate,
            'BYTE_RATE': byte_rate,
            'PKT_AVG_SIZE': pkt_size,
            'UTILIZATION': util,
            'PKT_TYPE_ENCODED': int(rng.choice([0,1,2,3])),
            'SRC_ADD': f'10.0.{rng.randint(1,254)}.{rng.randint(1,254)}',
            'DES_ADD': f'10.0.{rng.randint(1,254)}.{rng.randint(1,254)}'
        }
        df = pd.DataFrame([sample])
        label, msg = predict_features(df)
        injected += 1
        if label == 1:
            detected += 1
        details.append({'sample': sample, 'prediction': int(label) if label is not None else None, 'message': msg})

    return jsonify({'injected': injected, 'detected': detected, 'details': details[:10]})

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'attack_result': attack_result,
        'current_model': current_model_name,
        'sniffer_running': sniffer_thread.is_alive() if sniffer_thread else False
    })

@app.route('/logs', methods=['GET'])
def get_logs():
    # return last 50 logs
    return jsonify({'logs': logs[-50:]})

@app.route('/toggle_sniffer', methods=['POST'])
def toggle_sniffer():
    data = request.get_json() or {}
    action = data.get('action')
    if action == 'start':
        started = start_sniffer()
        return jsonify({'started': started})
    elif action == 'stop':
        stopped = stop_sniffer_thread()
        return jsonify({'stopped': stopped})
    else:
        return jsonify({'error': 'invalid action, use start or stop'}), 400

# Flask route for the web interface
@app.route('/')
def home():
    global attack_result
    return render_template('index.html', attack_result=attack_result,
                           available_models=list(MODEL_FILES.keys()),
                           current_model=current_model_name)


@app.route('/models_page')
def models_page():
    return render_template('models.html', available_models=list(MODEL_FILES.keys()), current_model=current_model_name)


@app.route('/analytics')
def analytics_page():
    return render_template('analytics.html')


@app.route('/export')
def export_page():
    return render_template('export.html')


@app.route('/help')
def help_page():
    return render_template('help.html')


@app.route('/logs_page')
def logs_page():
    # logs will be fetched by client-side AJAX
    return render_template('logs.html')


@app.route('/settings')
def settings_page():
    return render_template('settings.html', TIME_WINDOW=TIME_WINDOW, PACKET_THRESHOLD=PACKET_THRESHOLD)


@app.route('/about')
def about_page():
    return render_template('about.html')

if __name__ == "__main__":
    # Start sniffer automatically (same behavior as before)
    try:
        start_sniffer()
    except Exception:
        print("Could not start sniffer automatically. Use the UI to start it.")
    app.run(host="127.0.0.1", port=5000, debug=False)
