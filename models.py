"""
JEE Rank Predictor — ML Models Module
Extracted from notebook: Random Forest Regressor + XGBoost Classifier + SMOTE
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, classification_report)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


def get_category(ratio):
    """Notebook insight: Category based on RankRatio = Rank / Total_Candidates"""
    if ratio <= 0.005:
        return 'Elite (Top 0.5%)'
    elif ratio <= 0.02:
        return 'Top Tier (0.5% - 2%)'
    elif ratio <= 0.05:
        return 'Highly Competitive (2% - 5%)'
    elif ratio <= 0.10:
        return 'Competitive (5% - 10%)'
    else:
        return 'Not Prepared (>10%)'


CATEGORY_META = {
    'Elite (Top 0.5%)':            {'css': 'cat-elite', 'icon': '🏆', 'color': '#10b981'},
    'Top Tier (0.5% - 2%)':       {'css': 'cat-top',   'icon': '⭐', 'color': '#06b6d4'},
    'Highly Competitive (2% - 5%)': {'css': 'cat-high', 'icon': '🔥', 'color': '#6366f1'},
    'Competitive (5% - 10%)':      {'css': 'cat-comp',  'icon': '💪', 'color': '#f59e0b'},
    'Not Prepared (>10%)':         {'css': 'cat-not',   'icon': '📚', 'color': '#f43f5e'},
}


class JEEPredictor:
    def __init__(self, csv_path="jee_marks_percentile_rank_2009_2026.csv"):
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip()
        self.df = self.df[self.df['Rank'] > 0].copy()
        self.df['RankRatio'] = self.df['Rank'] / self.df['Total_Candidates']
        self.df['Category'] = self.df['RankRatio'].apply(get_category)

        self.year_candidates = self.df.groupby('Year')['Total_Candidates'].first().to_dict()

        self.sc_reg = StandardScaler()
        self.sc_cls = StandardScaler()
        self.encoder = LabelEncoder()
        self.poly = PolynomialFeatures(degree=3)

        self.reg_metrics = {}
        self.cls_metrics = {}

        self._train_regression()
        self._train_classification()

    # ═══════════════════════════════════════════
    # REGRESSION: RF (best) + Poly LR (baseline)
    # Notebook: "Tree-based models handle skewed/outlier data locally through splits"
    # ═══════════════════════════════════════════
    def _train_regression(self):
        X = self.df[['Year', 'Marks', 'Total_Candidates']]
        Y = np.log1p(self.df['Rank'])

        X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_tr_sc = self.sc_reg.fit_transform(X_tr)
        X_te_sc = self.sc_reg.transform(X_te)

        # Poly LR baseline
        X_tr_poly = self.poly.fit_transform(X_tr_sc)
        X_te_poly = self.poly.transform(X_te_sc)
        self.lr = LinearRegression()
        self.lr.fit(X_tr_poly, Y_tr)
        y_lr = np.expm1(self.lr.predict(X_te_poly))

        # Random Forest (notebook's best)
        self.rf = RandomForestRegressor(
            n_estimators=300, max_depth=5, min_samples_split=3,
            min_samples_leaf=5, random_state=42
        )
        self.rf.fit(X_tr_sc, Y_tr)
        y_rf = np.expm1(self.rf.predict(X_te_sc))

        y_actual = np.expm1(Y_te)

        # Confidence interval coverage (notebook: 10-90 percentile)
        all_preds = np.array([t.predict(X_te_sc) for t in self.rf.estimators_])
        lo = np.expm1(np.percentile(all_preds, 10, axis=0))
        hi = np.expm1(np.percentile(all_preds, 90, axis=0))
        coverage = float(np.mean((y_actual >= lo) & (y_actual <= hi)))
        avg_width = float((hi - lo).mean())

        n = len(y_actual)
        p_lr = X_te_poly.shape[1]
        p_rf = X_te_sc.shape[1]
        lr_r2 = r2_score(y_actual, y_lr)
        rf_r2 = r2_score(y_actual, y_rf)

        self.reg_metrics = {
            'lr_r2': round(lr_r2, 4),
            'lr_adj_r2': round(1 - (1 - lr_r2) * (n-1) / (n-p_lr-1), 4),
            'lr_mae': round(float(mean_absolute_error(y_actual, y_lr)), 0),
            'lr_rmse': round(float(np.sqrt(mean_squared_error(y_actual, y_lr))), 0),
            'rf_r2': round(rf_r2, 4),
            'rf_adj_r2': round(1 - (1 - rf_r2) * (n-1) / (n-p_rf-1), 4),
            'rf_mae': round(float(mean_absolute_error(y_actual, y_rf)), 0),
            'rf_rmse': round(float(np.sqrt(mean_squared_error(y_actual, y_rf))), 0),
            'rf_r2_train': round(float(self.rf.score(X_tr_sc, Y_tr)), 4),
            'rf_r2_test': round(float(self.rf.score(X_te_sc, Y_te)), 4),
            'coverage': round(coverage, 4),
            'avg_range_width': round(avg_width, 0),
        }

    # ═══════════════════════════════════════════
    # CLASSIFICATION: XGBoost + SMOTE
    # Notebook: "slightly lower training accuracy due to SMOTE → better generalization"
    # ═══════════════════════════════════════════
    def _train_classification(self):
        X = self.df[['Year', 'Marks', 'Rank']]
        Y = self.encoder.fit_transform(self.df['Category'])

        X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_tr_sc = self.sc_cls.fit_transform(X_tr)
        X_te_sc = self.sc_cls.transform(X_te)

        smote = SMOTE(random_state=42)
        X_tr_sm, Y_tr_sm = smote.fit_resample(X_tr_sc, Y_tr)

        self.xgb = XGBClassifier(
            n_estimators=200, learning_rate=0.02, max_depth=3,
            subsample=0.8, random_state=42, use_label_encoder=False,
            eval_metric='mlogloss'
        )
        self.xgb.fit(X_tr_sm, Y_tr_sm)

        y_pred = self.xgb.predict(X_te_sc)
        report = classification_report(Y_te, y_pred, output_dict=True,
                                       target_names=self.encoder.classes_)
        cv = cross_val_score(self.xgb, X_tr_sm, Y_tr_sm, cv=5)

        # Per-class metrics for frontend table
        class_rows = []
        for cat in self.encoder.classes_:
            if cat in report:
                r = report[cat]
                class_rows.append({
                    'category': cat,
                    'precision': round(r['precision'], 3),
                    'recall': round(r['recall'], 3),
                    'f1': round(r['f1-score'], 3),
                    'support': int(r['support']),
                })

        self.cls_metrics = {
            'xgb_acc': round(float(accuracy_score(Y_te, y_pred)), 4),
            'xgb_train_acc': round(float(self.xgb.score(X_tr_sm, Y_tr_sm)), 4),
            'cv_mean': round(float(cv.mean()), 4),
            'cv_std': round(float(cv.std()), 4),
            'class_report': class_rows,
        }

    # ─── Prediction API ───
    def predict(self, marks, year, total_candidates):
        inp = np.array([[year, marks, total_candidates]])
        inp_sc = self.sc_reg.transform(inp)

        # RF rank
        rf_log = self.rf.predict(inp_sc)[0]
        rf_rank = max(1, int(np.expm1(rf_log)))

        # Confidence interval from all 300 trees
        tree_preds = np.array([t.predict(inp_sc)[0] for t in self.rf.estimators_])
        rf_lower = max(1, int(np.expm1(np.percentile(tree_preds, 10))))
        rf_upper = max(1, int(np.expm1(np.percentile(tree_preds, 90))))

        # Poly LR rank
        inp_poly = self.poly.transform(inp_sc)
        lr_rank = max(1, int(np.expm1(self.lr.predict(inp_poly)[0])))

        # Percentile
        pct = max(0.0, min(100.0, 100 * (1 - rf_rank / total_candidates)))

        # Category via XGBoost
        inp_cls = np.array([[year, marks, rf_rank]])
        inp_cls_sc = self.sc_cls.transform(inp_cls)
        cat_enc = self.xgb.predict(inp_cls_sc)[0]
        cat_label = self.encoder.inverse_transform([cat_enc])[0]
        cat_meta = CATEGORY_META.get(cat_label, CATEGORY_META['Competitive (5% - 10%)'])

        return {
            'rf_rank': rf_rank,
            'rf_lower': rf_lower,
            'rf_upper': rf_upper,
            'lr_rank': lr_rank,
            'percentile': round(pct, 2),
            'category': cat_label,
            'category_css': cat_meta['css'],
            'category_icon': cat_meta['icon'],
            'category_color': cat_meta['color'],
        }

    def get_explorer_data(self, year, marks_min, marks_max):
        f = self.df[
            (self.df['Year'] == year) &
            (self.df['Marks'].between(marks_min, marks_max))
        ].sort_values('Marks', ascending=False)
        return f[['Marks','Percentile','Rank','Total_Candidates','Category']].to_dict('records')

    def get_scatter_data(self, year):
        d = self.df[self.df['Year'].between(year-2, year+2)]
        if len(d) == 0:
            d = self.df
        return d[['Marks','Rank','Percentile']].to_dict('records')

    def get_trend_data(self, marks):
        d = self.df[self.df['Marks'].between(marks-5, marks+5)]
        t = d.groupby('Year').agg(avg_rank=('Rank','mean')).reset_index()
        return t.to_dict('records')

    def get_heatmap_data(self):
        h = self.df.copy()
        h['bucket'] = pd.cut(h['Marks'], bins=range(0,310,30),
                             labels=[f"{i}-{i+29}" for i in range(0,300,30)])
        pivot = h.groupby(['bucket','Year'])['Rank'].mean().unstack(fill_value=0)
        return {
            'z': np.log1p(pivot.values).tolist(),
            'x': [str(c) for c in pivot.columns],
            'y': [str(b) for b in pivot.index],
        }
