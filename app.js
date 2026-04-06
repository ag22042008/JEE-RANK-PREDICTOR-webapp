/* ══════════════════════════════════════════════
   JEE Rank Predictor — Frontend Logic
   ══════════════════════════════════════════════ */

const PLOT_LAYOUT = {
    plot_bgcolor: 'rgba(12,12,20,0.8)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Inter, sans-serif', color: '#8b8b9e' },
    margin: { l: 55, r: 30, t: 55, b: 50 },
};
const GRID = 'rgba(255,255,255,0.04)';
const CONFIG = { responsive: true, displayModeBar: false };

let yearCandidatesMap = {};
let currentPrediction = null;

// ─── Init ───
document.addEventListener('DOMContentLoaded', async () => {
    setupSliders();
    setupTabs();
    await loadYearCandidates();
    await predict();
    await loadMetrics();
    setupExplorer();
});

// ─── Sliders ───
function setupSliders() {
    const marks = document.getElementById('marks-slider');
    const marksVal = document.getElementById('marks-value');
    const year = document.getElementById('year-slider');
    const yearVal = document.getElementById('year-value');

    marks.addEventListener('input', () => { marksVal.textContent = marks.value; });
    year.addEventListener('input', () => {
        yearVal.textContent = year.value;
        const c = yearCandidatesMap[year.value];
        if (c) document.getElementById('candidates-input').value = c;
    });

    document.getElementById('predict-btn').addEventListener('click', predict);

    // Also predict on Enter in candidates input
    document.getElementById('candidates-input').addEventListener('keydown', e => {
        if (e.key === 'Enter') predict();
    });
}

// ─── Tabs ───
function setupTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
        });
    });
}

// ─── Load year→candidates map ───
async function loadYearCandidates() {
    try {
        const res = await fetch('/api/year-candidates');
        yearCandidatesMap = await res.json();
        const yr = document.getElementById('year-slider').value;
        if (yearCandidatesMap[yr]) {
            document.getElementById('candidates-input').value = yearCandidatesMap[yr];
        }
    } catch (e) { console.error('Failed to load year candidates:', e); }
}

// ─── Predict ───
async function predict() {
    const btn = document.getElementById('predict-btn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoader = btn.querySelector('.btn-loader');
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline';

    const marks = parseInt(document.getElementById('marks-slider').value);
    const year = parseInt(document.getElementById('year-slider').value);
    const candidates = parseInt(document.getElementById('candidates-input').value);

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ marks, year, total_candidates: candidates })
        });
        const data = await res.json();
        currentPrediction = data;
        updateCards(data);
        await Promise.all([
            renderScatter(marks, year, data),
            renderBar(data),
            renderGauge(data.percentile),
            renderTrend(marks),
        ]);
    } catch (e) {
        console.error('Prediction failed:', e);
    }

    btnText.style.display = 'inline';
    btnLoader.style.display = 'none';
}

// ─── Update prediction cards ───
function fmt(n) { return n.toLocaleString(); }

function updateCards(d) {
    document.getElementById('rf-rank').textContent = fmt(d.rf_rank);
    document.getElementById('rf-range').textContent = `${fmt(d.rf_lower)} – ${fmt(d.rf_upper)}`;
    document.getElementById('pct-value').textContent = d.percentile.toFixed(2) + '%';
    document.getElementById('lr-rank').textContent = fmt(d.lr_rank);

    const badge = document.getElementById('cat-badge');
    badge.innerHTML = `<span class="category-badge ${d.category_css}">${d.category_icon} ${d.category}</span>`;
}

// ─── Charts ───
async function renderScatter(marks, year, pred) {
    try {
        const res = await fetch(`/api/scatter?year=${year}`);
        const data = await res.json();

        const trace1 = {
            x: data.map(d => d.Marks), y: data.map(d => d.Rank),
            mode: 'markers', name: 'Historical',
            marker: {
                size: 6, opacity: 0.6,
                color: data.map(d => d.Percentile),
                colorscale: [[0,'#f43f5e'],[0.5,'#8b5cf6'],[1,'#06b6d4']],
                colorbar: { title: { text: 'Pctl', font: { color: '#8b8b9e', size: 11 } },
                            tickfont: { color: '#8b8b9e', size: 10 }, bgcolor: 'rgba(0,0,0,0)', borderwidth: 0 }
            },
            hovertemplate: '<b>Marks:</b> %{x}<br><b>Rank:</b> %{y:,}<extra></extra>'
        };
        const trace2 = {
            x: [marks], y: [pred.rf_rank],
            mode: 'markers+text', name: 'You',
            marker: { size: 16, color: '#f59e0b', symbol: 'diamond', line: { width: 2, color: '#fbbf24' } },
            text: ['You'], textposition: 'top center', textfont: { color: '#f59e0b', size: 12 },
            hovertemplate: '<b>Your Marks:</b> %{x}<br><b>Rank:</b> %{y:,}<extra></extra>'
        };

        const layout = {
            ...PLOT_LAYOUT,
            title: { text: 'Marks vs Rank Distribution', font: { size: 15, color: '#f0f0f5' }, x: 0.02 },
            xaxis: { title: 'Marks', color: '#8b8b9e', gridcolor: GRID },
            yaxis: { title: 'Rank', color: '#8b8b9e', gridcolor: GRID, autorange: 'reversed' },
            legend: { orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'right', x: 1,
                      font: { size: 11, color: '#8b8b9e' }, bgcolor: 'rgba(0,0,0,0)' },
            height: 400,
        };
        Plotly.newPlot('scatter-chart', [trace1, trace2], layout, CONFIG);
    } catch (e) { console.error('Scatter failed:', e); }
}

async function renderBar(pred) {
    const models = [
        { name: 'RF Regressor', rank: pred.rf_rank, color: '#6366f1' },
        { name: 'Poly LR', rank: pred.lr_rank, color: '#f43f5e' },
        { name: 'RF Lower', rank: pred.rf_lower, color: '#10b981' },
        { name: 'RF Upper', rank: pred.rf_upper, color: '#06b6d4' },
    ];
    const traces = models.map(m => ({
        x: [m.rank], y: [m.name], orientation: 'h', type: 'bar',
        marker: { color: m.color, opacity: 0.85 },
        text: [`  ${fmt(m.rank)}`], textposition: 'outside',
        textfont: { color: m.color, size: 13, family: 'JetBrains Mono' },
        showlegend: false,
        hovertemplate: `<b>${m.name}</b><br>Rank: ${fmt(m.rank)}<extra></extra>`
    }));
    const layout = {
        ...PLOT_LAYOUT,
        title: { text: 'Model Predictions', font: { size: 15, color: '#f0f0f5' }, x: 0.02 },
        xaxis: { title: 'Predicted Rank', color: '#8b8b9e', gridcolor: GRID },
        yaxis: { color: '#8b8b9e', categoryorder: 'array',
                 categoryarray: ['RF Upper','RF Lower','Poly LR','RF Regressor'] },
        margin: { ...PLOT_LAYOUT.margin, l: 110, r: 80 },
        height: 400, bargap: 0.35,
    };
    Plotly.newPlot('bar-chart', traces, layout, CONFIG);
}

async function renderGauge(pct) {
    const trace = {
        type: 'indicator', mode: 'gauge+number+delta', value: pct,
        number: { font: { size: 42, color: '#f0f0f5', family: 'JetBrains Mono' }, suffix: '%' },
        delta: { reference: 90, increasing: { color: '#10b981' }, decreasing: { color: '#f43f5e' } },
        gauge: {
            axis: { range: [0,100], tickwidth: 1, tickcolor: '#5a5a6e', tickfont: { color: '#5a5a6e', size: 10 } },
            bar: { color: '#6366f1', thickness: 0.3 },
            bgcolor: 'rgba(255,255,255,0.03)', borderwidth: 0,
            steps: [
                { range: [0,50], color: 'rgba(244,63,94,0.1)' },
                { range: [50,80], color: 'rgba(245,158,11,0.1)' },
                { range: [80,95], color: 'rgba(6,182,212,0.1)' },
                { range: [95,100], color: 'rgba(16,185,129,0.1)' },
            ],
            threshold: { line: { color: '#f59e0b', width: 3 }, thickness: 0.8, value: pct },
        },
        title: { text: 'Your Percentile Score', font: { size: 14, color: '#8b8b9e' } },
    };
    const layout = {
        ...PLOT_LAYOUT, margin: { l: 30, r: 30, t: 80, b: 30 }, height: 380,
    };
    Plotly.newPlot('gauge-chart', [trace], layout, CONFIG);
}

async function renderTrend(marks) {
    try {
        const res = await fetch(`/api/trend?marks=${marks}`);
        const data = await res.json();
        if (data.length === 0) {
            document.getElementById('trend-chart').innerHTML =
                '<p style="color:#8b8b9e;text-align:center;padding:3rem;">Not enough data for this marks range.</p>';
            return;
        }
        const trace = {
            x: data.map(d => d.Year), y: data.map(d => d.avg_rank),
            mode: 'lines+markers', name: 'Avg Rank',
            line: { color: '#8b5cf6', width: 3, shape: 'spline' },
            marker: { size: 8, color: '#8b5cf6', line: { width: 2, color: '#a78bfa' } },
            fill: 'tozeroy', fillcolor: 'rgba(139,92,246,0.08)',
            hovertemplate: '<b>Year:</b> %{x}<br><b>Avg Rank:</b> %{y:,.0f}<extra></extra>'
        };
        const layout = {
            ...PLOT_LAYOUT,
            title: { text: `Rank Trend for ~${marks} Marks (±5)`, font: { size: 15, color: '#f0f0f5' }, x: 0.02 },
            xaxis: { title: 'Year', color: '#8b8b9e', gridcolor: GRID, dtick: 1 },
            yaxis: { title: 'Average Rank', color: '#8b8b9e', gridcolor: GRID, autorange: 'reversed' },
            height: 380, showlegend: false,
        };
        Plotly.newPlot('trend-chart', [trace], layout, CONFIG);
    } catch (e) { console.error('Trend failed:', e); }
}

// ─── Model metrics ───
async function loadMetrics() {
    try {
        const res = await fetch('/api/metrics');
        const m = await res.json();
        renderRegTable(m.regression);
        renderRFCards(m.regression);
        renderClsCards(m.classification);
        renderClsTable(m.classification);
    } catch (e) { console.error('Metrics failed:', e); }
}

function renderRegTable(r) {
    const tbody = document.querySelector('#reg-table tbody');
    tbody.innerHTML = `
        <tr><td style="font-family:Inter;font-weight:600;">Polynomial LR</td>
            <td>${r.lr_r2}</td><td>${r.lr_adj_r2}</td><td>${fmt(r.lr_mae)}</td><td>${fmt(r.lr_rmse)}</td></tr>
        <tr><td style="font-family:Inter;font-weight:600;">Random Forest</td>
            <td>${r.rf_r2}</td><td>${r.rf_adj_r2}</td><td>${fmt(r.rf_mae)}</td><td>${fmt(r.rf_rmse)}</td></tr>
    `;
}

function makeCard(label, value, colorClass) {
    return `<div class="pred-card ${colorClass}" style="text-align:center;">
        <div class="pred-label ${colorClass}">${label}</div>
        <div class="pred-value" style="font-size:1.7rem;">${value}</div>
    </div>`;
}

function renderRFCards(r) {
    const el = document.getElementById('rf-metric-cards');
    el.innerHTML =
        makeCard('RF R² (Train)', r.rf_r2_train, 'indigo') +
        makeCard('RF R² (Test)', r.rf_r2_test, 'cyan') +
        makeCard('CI Coverage', (r.coverage * 100).toFixed(1) + '%', 'emerald') +
        makeCard('Avg Range Width', fmt(r.avg_range_width), 'rose');
}

function renderClsCards(c) {
    const el = document.getElementById('cls-metric-cards');
    el.innerHTML =
        makeCard('Test Accuracy', (c.xgb_acc * 100).toFixed(1) + '%', 'indigo') +
        makeCard('Train Accuracy', (c.xgb_train_acc * 100).toFixed(1) + '%', 'cyan') +
        makeCard('Cross-Val (5-fold)', (c.cv_mean * 100).toFixed(1) + '% ± ' + c.cv_std.toFixed(3), 'emerald');
}

function renderClsTable(c) {
    const tbody = document.querySelector('#cls-table tbody');
    tbody.innerHTML = c.class_report.map(r =>
        `<tr><td style="font-family:Inter;font-weight:500;">${r.category}</td>
             <td>${r.precision}</td><td>${r.recall}</td><td>${r.f1}</td><td>${r.support}</td></tr>`
    ).join('');
}

// ─── Explorer ───
function setupExplorer() {
    const sel = document.getElementById('explorer-year');
    const years = Object.keys(yearCandidatesMap).sort((a,b) => b-a);
    sel.innerHTML = years.map(y => `<option value="${y}">${y}</option>`).join('');

    document.getElementById('explorer-btn').addEventListener('click', loadExplorerData);
    loadExplorerData();
    loadHeatmap();
}

async function loadExplorerData() {
    const year = document.getElementById('explorer-year').value;
    const min = document.getElementById('explorer-min').value;
    const max = document.getElementById('explorer-max').value;

    try {
        const res = await fetch(`/api/explorer?year=${year}&marks_min=${min}&marks_max=${max}`);
        const data = await res.json();

        document.getElementById('explorer-info').innerHTML =
            `Showing <strong>${data.length}</strong> records for year <strong>${year}</strong>`;

        const tbody = document.querySelector('#explorer-table tbody');
        tbody.innerHTML = data.map(r =>
            `<tr><td>${r.Marks}</td><td>${r.Percentile.toFixed(2)}</td><td>${fmt(r.Rank)}</td>
                 <td>${fmt(r.Total_Candidates)}</td><td>${r.Category}</td></tr>`
        ).join('');
    } catch (e) { console.error('Explorer failed:', e); }
}

async function loadHeatmap() {
    try {
        const res = await fetch('/api/heatmap');
        const data = await res.json();

        const trace = {
            z: data.z, x: data.x, y: data.y, type: 'heatmap',
            colorscale: [[0,'#0a0a0f'],[0.2,'#1e1b4b'],[0.4,'#4338ca'],[0.6,'#6366f1'],[0.8,'#a78bfa'],[1,'#e0e7ff']],
            hovertemplate: '<b>Marks:</b> %{y}<br><b>Year:</b> %{x}<br><b>Log Avg Rank:</b> %{z:.1f}<extra></extra>',
            colorbar: { title: { text: 'Log(Rank)', font: { color: '#8b8b9e' } },
                        tickfont: { color: '#8b8b9e' }, bgcolor: 'rgba(0,0,0,0)', borderwidth: 0 },
        };
        const layout = {
            ...PLOT_LAYOUT,
            title: { text: 'Average Rank Heatmap (Log Scale)', font: { size: 15, color: '#f0f0f5' }, x: 0.02 },
            xaxis: { title: 'Year', color: '#8b8b9e' },
            yaxis: { title: 'Marks Range', color: '#8b8b9e', autorange: 'reversed' },
            margin: { ...PLOT_LAYOUT.margin, l: 80 }, height: 450,
        };
        Plotly.newPlot('heatmap-chart', [trace], layout, CONFIG);
    } catch (e) { console.error('Heatmap failed:', e); }
}
