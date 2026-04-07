"""
JEE Rank Predictor — Flask Backend
Serves the frontend + ML prediction API
"""
from flask import Flask, render_template, jsonify, request
from models import JEEPredictor

app = Flask(__name__)

# Train models once at startup
print("[*] Training models...")
predictor = JEEPredictor()
print("[OK] Models ready!")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    marks = int(data.get('marks', 150))
    year = int(data.get('year', 2025))
    total_candidates = int(data.get('total_candidates', 1400000))
    result = predictor.predict(marks, year, total_candidates)
    return jsonify(result)


@app.route('/api/metrics')
def metrics():
    return jsonify({
        'regression': predictor.reg_metrics,
        'classification': predictor.cls_metrics,
    })


@app.route('/api/year-candidates')
def year_candidates():
    return jsonify(predictor.year_candidates)


@app.route('/api/scatter')
def scatter():
    year = int(request.args.get('year', 2025))
    return jsonify(predictor.get_scatter_data(year))


@app.route('/api/trend')
def trend():
    marks = int(request.args.get('marks', 150))
    return jsonify(predictor.get_trend_data(marks))


@app.route('/api/explorer')
def explorer():
    year = int(request.args.get('year', 2025))
    marks_min = int(request.args.get('marks_min', 0))
    marks_max = int(request.args.get('marks_max', 300))
    return jsonify(predictor.get_explorer_data(year, marks_min, marks_max))


@app.route('/api/heatmap')
def heatmap():
    return jsonify(predictor.get_heatmap_data())


import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
