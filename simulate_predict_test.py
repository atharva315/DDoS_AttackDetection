import glob
import joblib
import pandas as pd
import numpy as np
import os

# Find a model file
pkls = glob.glob('*.pkl')
if not pkls:
    print('No .pkl models found in current dir')
    raise SystemExit(1)

# prefer logistic if present
p = None
for name in ['logistic_DDoS.pkl','rf_DDoS.pkl','gb_DDoS.pkl']:
    if name in pkls:
        p = name
        break
if p is None:
    p = pkls[0]

print('Using model file:', p)
load = joblib.load(p)
if isinstance(load, dict) and 'pipeline' in load:
    model = load['pipeline']
    feature_names = load.get('feature_names')
else:
    model = load
    feature_names = None

# try to derive feature names
if feature_names is None:
    try:
        if hasattr(model, 'named_steps'):
            clf = model.named_steps.get('clf')
            if clf is not None and hasattr(clf, 'feature_names_in_'):
                feature_names = list(clf.feature_names_in_)
    except Exception:
        pass

if feature_names is None:
    feature_names = [
        'PKT_SIZE', 'NUMBER_OF_PKT', 'NUMBER_OF_BYTE', 'PKT_DELAY_NODE',
        'PKT_RATE', 'BYTE_RATE', 'PKT_AVG_SIZE', 'UTILIZATION', 'PKT_TYPE_ENCODED'
    ]

print('Feature names used:', feature_names)

# create a normal and an attack-like sample
normal = {f: 0 for f in feature_names}
attack = {f: 0 for f in feature_names}
for f in feature_names:
    if f == 'PKT_SIZE':
        normal[f] = 400
        attack[f] = 1200
    elif f == 'NUMBER_OF_PKT':
        normal[f] = 5
        attack[f] = 1000
    elif f == 'NUMBER_OF_BYTE':
        normal[f] = normal.get('PKT_SIZE',400)*normal.get('NUMBER_OF_PKT',5)
        attack[f] = attack.get('PKT_SIZE',1200)*attack.get('NUMBER_OF_PKT',1000)
    elif f == 'PKT_DELAY_NODE':
        normal[f] = 0.02
        attack[f] = 0.001
    elif f == 'PKT_RATE':
        normal[f] = 10
        attack[f] = 2000
    elif f == 'BYTE_RATE':
        normal[f] = normal['PKT_RATE']*normal['PKT_SIZE']
        attack[f] = attack['PKT_RATE']*attack['PKT_SIZE']
    elif f == 'PKT_AVG_SIZE':
        normal[f] = 300
        attack[f] = 1100
    elif f == 'UTILIZATION':
        normal[f] = 0.05
        attack[f] = 0.98
    elif f == 'PKT_TYPE_ENCODED':
        normal[f] = 0
        attack[f] = 3

# DataFrame
df = pd.DataFrame([normal, attack], columns=feature_names)
print(df)

# Predict
try:
    preds = model.predict(df)
    print('Predictions:', preds)
except Exception as e:
    print('Predict failed:', e)
    try:
        preds = model.predict(df.values)
        print('Predictions (via values):', preds)
    except Exception as e2:
        print('Predict via values also failed:', e2)

# Probabilities if available
try:
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(df)
        print('Probabilities:\n', probs)
    else:
        print('Model has no predict_proba')
except Exception as e:
    print('predict_proba failed:', e)
