"""
ml_engine.py — LMS Student Performance Prediction Engine
=========================================================
Chuyên gia: Data Scientist / ML Engineer — Learning Analytics

Kiến trúc:
  1. load_data()             → Nạp và validate DataFrame
  2. preprocess_data()       → Time-series aggregation chuyên sâu (sum, mean, std, trend)
  3. train_models()          → Huấn luyện kép: Regressor (điểm) + Classifier (đậu/rớt)
  4. get_feature_importance()→ SHAP-style importance toàn bộ 15 features
  5. predict_student()       → Inference: điểm + phân phối xác suất + cluster

Dataset columns (18 fields):
  student_id, week, login_count, time_spent_minutes, video_views, forum_posts,
  forum_reads, materials_downloaded, quiz_attempts, quiz_score,
  assignments_submitted, assignments_on_time, practice_exercises_done,
  messages_sent, attendance_virtual, midterm_score,
  final_exam_score (target), pass_fail (label)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               GradientBoostingRegressor)
from sklearn.model_selection import (GridSearchCV, train_test_split,
                                     StratifiedKFold)
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, precision_score, recall_score,
                              cohen_kappa_score, r2_score,
                              mean_absolute_error, mean_squared_error)
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────────────────────────────────────
# CÁC HẰNG SỐ CẤU HÌNH
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'login_count', 'time_spent_minutes', 'video_views', 'forum_posts',
    'forum_reads', 'materials_downloaded', 'quiz_attempts', 'quiz_score',
    'assignments_submitted', 'assignments_on_time', 'practice_exercises_done',
    'messages_sent', 'attendance_virtual', 'midterm_score',
]
# Cột tổng hợp theo SUM (tần suất/hoạt động, có ý nghĩa khi cộng dồn)
SUM_COLS = [
    'login_count', 'time_spent_minutes', 'video_views', 'forum_posts',
    'forum_reads', 'materials_downloaded', 'quiz_attempts',
    'assignments_submitted', 'assignments_on_time', 'practice_exercises_done',
    'messages_sent', 'attendance_virtual',
]
# Cột tổng hợp theo MEAN (điểm số / tỉ lệ, có ý nghĩa khi lấy trung bình)
MEAN_COLS = ['quiz_score', 'midterm_score']

TARGET_REG = 'final_exam_score'
TARGET_CLF = 'pass_fail'
ID_COL     = 'week_col'  # tên cột thời gian


# ═════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════
def load_data(source) -> pd.DataFrame:
    """
    Nạp dữ liệu từ file path (str) hoặc DataFrame.
    Validate cấu trúc cột tối thiểu cần thiết.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, str):
        if source.endswith('.csv'):
            df = pd.read_csv(source)
        else:
            df = pd.read_excel(source)
    else:
        raise TypeError("source phải là DataFrame hoặc đường dẫn file str.")

    required = {'student_id', TARGET_REG, TARGET_CLF}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {missing}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 2. PREPROCESS DATA — Time-series Aggregation Chuyên Sâu
# ═════════════════════════════════════════════════════════════════════════════
def _compute_trend(series: pd.Series) -> float:
    """
    Tính hệ số góc (slope) của đường hồi quy tuyến tính qua các tuần.
    Dương → xu hướng tăng; Âm → xu hướng giảm.
    """
    n = len(series)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    y = series.values.astype(float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return 0.0
    try:
        slope = np.polyfit(x[mask], y[mask], 1)[0]
        return float(slope)
    except Exception:
        return 0.0


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tổng hợp dữ liệu time-series theo từng student_id.

    Chiến lược aggregation:
    ┌─────────────────────────────────┬──────────────────────────────┐
    │ Loại cột                        │ Phép tổng hợp                │
    ├─────────────────────────────────┼──────────────────────────────┤
    │ Hoạt động (login, time, ...)    │ sum + mean + std + trend     │
    │ Điểm số (quiz_score, midterm)   │ mean + max + min + trend     │
    │ Target (final_exam, pass_fail)  │ first (giá trị cố định)      │
    └─────────────────────────────────┴──────────────────────────────┘

    Trả về DataFrame 1 dòng/sinh viên với ~50+ engineered features.
    """
    df = df.copy()

    # Xác định các cột có trong dữ liệu
    sum_cols  = [c for c in SUM_COLS  if c in df.columns]
    mean_cols = [c for c in MEAN_COLS if c in df.columns]
    extra_num = [c for c in df.select_dtypes(include=np.number).columns
                 if c not in sum_cols + mean_cols + ['student_id']
                 and c not in [TARGET_REG] and c not in ['week']]

    records = []
    for sid, grp in df.groupby('student_id'):
        grp = grp.sort_values('week') if 'week' in grp.columns else grp
        row = {'student_id': sid}

        # ── Targets (lấy giá trị đầu tiên — giống nhau cho mọi tuần) ──
        if TARGET_REG in grp.columns:
            row[TARGET_REG] = grp[TARGET_REG].iloc[0]
        if TARGET_CLF in grp.columns:
            row[TARGET_CLF] = grp[TARGET_CLF].iloc[0]

        # ── SUM columns: tổng + trung bình + độ lệch chuẩn + trend ──
        for col in sum_cols:
            s = grp[col].dropna()
            row[f'{col}_sum']   = s.sum()
            row[f'{col}_mean']  = s.mean() if len(s) else np.nan
            row[f'{col}_std']   = s.std()  if len(s) > 1 else 0.0
            row[f'{col}_trend'] = _compute_trend(s)

        # ── MEAN columns: trung bình + max + min + trend ──
        for col in mean_cols:
            s = grp[col].dropna()
            row[f'{col}_mean']  = s.mean() if len(s) else np.nan
            row[f'{col}_max']   = s.max()  if len(s) else np.nan
            row[f'{col}_min']   = s.min()  if len(s) else np.nan
            row[f'{col}_trend'] = _compute_trend(s)

        records.append(row)

    df_agg = pd.DataFrame(records)
    return df_agg


# ═════════════════════════════════════════════════════════════════════════════
# 3. TRAIN MODELS — Dual Modeling Pipeline
# ═════════════════════════════════════════════════════════════════════════════
def _get_feature_matrix(df_agg: pd.DataFrame):
    """Trích xuất X (features) và y_reg, y_clf từ DataFrame đã aggregate."""
    drop_cols = ['student_id', TARGET_REG, TARGET_CLF]
    X = df_agg.drop(columns=[c for c in drop_cols if c in df_agg.columns])
    y_reg = df_agg[TARGET_REG] if TARGET_REG in df_agg.columns else None
    y_clf = df_agg[TARGET_CLF] if TARGET_CLF in df_agg.columns else None
    return X, y_reg, y_clf


def train_models(df_agg: pd.DataFrame) -> dict:
    """
    Pipeline huấn luyện đầy đủ:
    A) Tiền xử lý (Impute → Scale)
    B) SMOTE (cân bằng nhãn đậu/rớt)
    C) Hồi quy (GradientBoosting + RandomForest — chọn tốt nhất theo R²)
    D) Phân loại (5 thuật toán — chọn tốt nhất theo Accuracy)
    E) K-Means Clustering (sinh viên → nhóm học lực)

    Trả về dict chứa toàn bộ artifacts cần thiết.
    """
    X_raw, y_reg, y_clf = _get_feature_matrix(df_agg)
    feature_names = X_raw.columns.tolist()

    # ── Impute + Scale ──
    imputer = SimpleImputer(strategy='mean')
    X_imp   = pd.DataFrame(imputer.fit_transform(X_raw), columns=feature_names)
    scaler  = StandardScaler()
    X_sc    = pd.DataFrame(scaler.fit_transform(X_imp), columns=feature_names)

    # ── Label Encode y_clf ──
    le = LabelEncoder()
    y_enc = le.fit_transform(y_clf.fillna('Unknown'))

    artifacts = {
        'imputer': imputer, 'scaler': scaler, 'le': le,
        'feature_names': feature_names,
        'all_clf_reports': {}, 'all_reg_reports': {},
    }

    # ════════════════════════════
    # A) REGRESSION  (Dự báo điểm)
    # ════════════════════════════
    X_rtr, X_rte, y_rtr, y_rte = train_test_split(
        X_sc, y_reg, test_size=0.3, random_state=42)

    reg_candidates = {
        'RandomForest': GridSearchCV(
            RandomForestRegressor(random_state=42),
            {'n_estimators': [100, 200], 'max_depth': [None, 5, 10]},
            cv=min(3, len(X_rtr)), scoring='r2', n_jobs=-1),
        'GradientBoosting': GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]},
            cv=min(3, len(X_rtr)), scoring='r2', n_jobs=-1),
    }

    best_reg, best_r2, best_reg_name = None, -np.inf, ""
    for name, model in reg_candidates.items():
        try:
            model.fit(X_rtr, y_rtr)
            est = model.best_estimator_ if hasattr(model, 'best_estimator_') else model
            r2  = r2_score(y_rte, est.predict(X_rte))
            mae = mean_absolute_error(y_rte, est.predict(X_rte))
            rmse = np.sqrt(mean_squared_error(y_rte, est.predict(X_rte)))
            artifacts['all_reg_reports'][name] = {'r2': r2, 'mae': mae, 'rmse': rmse}
            if r2 > best_r2:
                best_r2, best_reg, best_reg_name = r2, est, name
        except Exception as e:
            artifacts['all_reg_reports'][name] = {'error': str(e)}

    artifacts['best_reg']      = best_reg
    artifacts['best_reg_name'] = best_reg_name
    artifacts['reg_r2']        = best_r2
    artifacts['reg_mae']       = artifacts['all_reg_reports'].get(best_reg_name, {}).get('mae', 0)
    artifacts['reg_rmse']      = artifacts['all_reg_reports'].get(best_reg_name, {}).get('rmse', 0)

    # Feature importances từ RF Regression
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_sc, y_reg)
    feat_imp_series = pd.Series(rf_reg.feature_importances_, index=feature_names)
    artifacts['feat_importances'] = feat_imp_series.sort_values(ascending=False).to_dict()
    artifacts['feat_importances_rf'] = rf_reg  # để tính xác suất phân phối

    # ════════════════════════════
    # B) CLASSIFICATION (Đậu/Rớt)
    # ════════════════════════════
    # SMOTE
    min_cnt = pd.Series(y_enc).value_counts().min()
    n_nb    = min(5, min_cnt - 1) if min_cnt > 1 else 1
    X_sm, y_sm = (SMOTE(random_state=42, k_neighbors=n_nb).fit_resample(X_sc, y_enc)
                  if n_nb >= 1 else (X_sc, y_enc))

    cv = StratifiedKFold(n_splits=min(5, pd.Series(y_sm).value_counts().min()),
                         shuffle=True, random_state=42)
    X_ctr, X_cte, y_ctr, y_cte = train_test_split(
        X_sm, y_sm, test_size=0.3, random_state=42, stratify=y_sm)

    clf_candidates = {
        'CART':               (DecisionTreeClassifier(random_state=42),
                               {'max_depth': [None, 3, 5, 10], 'criterion': ['gini', 'entropy']}),
        'SVM':                (SVC(probability=True, random_state=42),
                               {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42),
                               {'C': [0.1, 1, 10]}),
        'KNN':                (KNeighborsClassifier(),
                               {'n_neighbors': [3, 5, 7], 'metric': ['euclidean', 'cosine']}),
        'Naive Bayes':        (GaussianNB(), {}),
    }

    best_clf, best_acc, best_clf_name = None, -1, ""
    for name, (est, params) in clf_candidates.items():
        try:
            if params:
                gs = GridSearchCV(est, params, cv=2, scoring='accuracy', n_jobs=-1)
                gs.fit(X_ctr, y_ctr)
                fitted = gs.best_estimator_
            else:
                est.fit(X_ctr, y_ctr)
                fitted = est

            y_pred  = fitted.predict(X_cte)
            acc     = accuracy_score(y_cte, y_pred)
            f1      = f1_score(y_cte, y_pred, average='weighted')
            kappa   = cohen_kappa_score(y_cte, y_pred)
            cm      = confusion_matrix(y_cte, y_pred).tolist()
            try:
                auc = roc_auc_score(y_cte, fitted.predict_proba(X_cte)[:, 1])
            except Exception:
                auc = None

            artifacts['all_clf_reports'][name] = {
                'accuracy': acc, 'f1': f1, 'kappa': kappa,
                'auc': auc, 'confusion_matrix': cm, 'model': fitted
            }
            if acc > best_acc:
                best_acc, best_clf, best_clf_name = acc, fitted, name
        except Exception as e:
            artifacts['all_clf_reports'][name] = {'error': str(e)}

    artifacts['best_clf']      = best_clf
    artifacts['best_clf_name'] = best_clf_name
    artifacts['clf_acc']       = best_acc
    artifacts['clf_f1']        = artifacts['all_clf_reports'].get(best_clf_name, {}).get('f1', 0)
    artifacts['clf_kappa']     = artifacts['all_clf_reports'].get(best_clf_name, {}).get('kappa', 0)
    artifacts['clf_auc']       = artifacts['all_clf_reports'].get(best_clf_name, {}).get('auc')
    artifacts['smote_size']    = len(X_sm)

    # ════════════════════════════
    # C) K-MEANS CLUSTERING
    # ════════════════════════════
    from sklearn.metrics import davies_bouldin_score
    db_scores, inertias, k_range = [], [], range(2, min(7, len(X_sc)))
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X_sc)
        inertias.append(km.inertia_)
        db_scores.append(davies_bouldin_score(X_sc, lbl))
    best_k = list(k_range)[int(np.argmin(db_scores))] if db_scores else 3
    km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km_best.fit_predict(X_sc)

    artifacts['kmeans']              = km_best
    artifacts['cluster_k']          = best_k
    artifacts['cluster_k_range']    = list(k_range)
    artifacts['cluster_inertias']   = inertias
    artifacts['cluster_db_scores']  = db_scores
    artifacts['cluster_labels']     = cluster_labels.tolist()

    return artifacts


# ═════════════════════════════════════════════════════════════════════════════
# 4. GET FEATURE IMPORTANCE — Toàn bộ 15 features + giải thích ý nghĩa
# ═════════════════════════════════════════════════════════════════════════════

# Từ điển giải thích ngữ nghĩa từng nhóm feature
FEATURE_MEANINGS = {
    'login_count':               'Tần suất truy cập LMS — phản ánh mức độ chủ động trong học tập',
    'time_spent_minutes':        'Tổng thời gian học — thước đo nỗ lực đầu tư thực sự',
    'video_views':               'Lượt xem bài giảng — mức độ tiêu thụ nội dung học thuật',
    'forum_posts':               'Bài đăng diễn đàn — thể hiện tư duy phản biện và tương tác',
    'forum_reads':               'Lượt đọc diễn đàn — học từ đồng nghiệp (peer learning)',
    'materials_downloaded':      'Tài liệu tải về — mức độ chuẩn bị bài học chủ động',
    'quiz_attempts':             'Số lần thử quiz — sự kiên trì và tự đánh giá',
    'quiz_score':                'Điểm quiz — hiệu suất học tập tức thời qua kiểm tra nhỏ',
    'assignments_submitted':     'Bài tập đã nộp — mức độ hoàn thành nhiệm vụ học thuật',
    'assignments_on_time':       'Nộp đúng hạn — kỷ luật và quản lý thời gian',
    'practice_exercises_done':   'Bài thực hành — luyện tập chủ động ngoài yêu cầu',
    'messages_sent':             'Tin nhắn gửi — tương tác với giảng viên/bạn học',
    'attendance_virtual':        'Tham gia học trực tuyến — sự hiện diện và cam kết',
    'midterm_score':             'Điểm giữa kỳ — dự báo mạnh nhất cho kết quả cuối kỳ',
}


def get_feature_importance(artifacts: dict) -> pd.DataFrame:
    """
    Trả về DataFrame xếp hạng TOÀN BỘ features theo mức độ đóng góp.
    Bao gồm:
    - importance_score: trọng số từ RandomForest (0~1, tổng = 1)
    - importance_pct:   % đóng góp
    - meaning:          giải thích ngữ nghĩa
    - base_feature:     tên feature gốc (trước khi aggregate)
    """
    fi_dict = artifacts.get('feat_importances', {})
    if not fi_dict:
        return pd.DataFrame()

    rows = []
    for feat, score in fi_dict.items():
        # Lấy tên feature gốc (bỏ hậu tố _sum, _mean, _std, _trend, _max, _min)
        base = feat
        for suffix in ['_sum', '_mean', '_std', '_trend', '_max', '_min']:
            if feat.endswith(suffix):
                base = feat[:-len(suffix)]
                break
        meaning = FEATURE_MEANINGS.get(base, f'Chỉ số phụ của {base}')
        rows.append({
            'Feature (Engineered)': feat,
            'Feature Gốc':         base,
            'Importance Score':    round(score, 6),
            'Đóng góp (%)':        round(score * 100, 2),
            'Ý nghĩa':             meaning,
        })

    df_fi = pd.DataFrame(rows).sort_values('Importance Score', ascending=False)
    df_fi['Xếp hạng'] = range(1, len(df_fi) + 1)
    return df_fi.reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# 5. PREDICT STUDENT — Inference Đơn + Phân phối Xác suất Điểm
# ═════════════════════════════════════════════════════════════════════════════
def predict_student(student_weekly_data: pd.DataFrame, artifacts: dict) -> dict:
    """
    Dự đoán kết quả cho MỘT sinh viên từ dữ liệu tuần ghi nhận.

    Trả về dict:
    ┌────────────────────────┬─────────────────────────────────────────────────┐
    │ Khóa                   │ Ý nghĩa                                         │
    ├────────────────────────┼─────────────────────────────────────────────────┤
    │ predicted_score        │ Điểm cuối kỳ dự báo (float)                    │
    │ score_prob_margin      │ Xác suất điểm rơi vào ±0.5 của predicted_score │
    │ score_distribution     │ Dict {ngưỡng: xác suất} (4.0, 5.0, ..., 9.0)  │
    │ pass_fail_status       │ 'Pass' / 'Fail'                                 │
    │ pass_fail_confidence   │ Xác suất của status (float 0~1)                │
    │ cluster                │ Nhóm học lực ('Nhóm A', 'B', 'C', ...)         │
    │ feature_values         │ Vec feature đã aggregate (dict)                 │
    └────────────────────────┴─────────────────────────────────────────────────┘
    """
    # 1. Aggregate dữ liệu tuần
    if 'student_id' not in student_weekly_data.columns:
        student_weekly_data = student_weekly_data.copy()
        student_weekly_data['student_id'] = '__single__'
    df_tmp = pd.concat([student_weekly_data,
                        pd.DataFrame({'student_id': ['__single__'],
                                      'final_exam_score': [0.0],
                                      'pass_fail': ['Pass']})])
    # Chỉ aggregate dữ liệu của sinh viên này
    df_agg_row = preprocess_data(student_weekly_data.assign(
        final_exam_score=0.0, pass_fail='Pass'))

    X_raw, _, _ = _get_feature_matrix(df_agg_row)

    # 2. Align với feature names từ training
    feature_names = artifacts['feature_names']
    for missing_col in set(feature_names) - set(X_raw.columns):
        X_raw[missing_col] = 0.0
    X_raw = X_raw[feature_names]

    # 3. Transform
    imputer = artifacts['imputer']
    scaler  = artifacts['scaler']
    X_imp   = pd.DataFrame(imputer.transform(X_raw), columns=feature_names)
    X_sc    = pd.DataFrame(scaler.transform(X_imp), columns=feature_names)

    # 4. Regression → Điểm dự báo
    best_reg = artifacts['best_reg']
    pred_score = float(best_reg.predict(X_sc)[0])
    pred_score = max(0.0, min(10.0, pred_score))  # Clip [0, 10]

    # 5. Phân phối xác suất điểm — dùng cây trong RF
    rf_reg = artifacts.get('feat_importances_rf', best_reg)
    tree_preds = []
    if hasattr(rf_reg, 'estimators_'):
        tree_preds = [t.predict(X_sc.values)[0] for t in rf_reg.estimators_]
    else:
        tree_preds = [pred_score] * 100

    # Xác suất đạt ±0.5 điểm
    in_margin = sum(1 for p in tree_preds if abs(p - pred_score) <= 0.5)
    score_prob_margin = in_margin / len(tree_preds)

    # Phân phối xác suất theo ngưỡng điểm (4.0, 5.0, ..., 9.0)
    thresholds = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    score_distribution = {}
    for thr in thresholds:
        prob = sum(1 for p in tree_preds if p >= thr) / len(tree_preds)
        score_distribution[f'>= {thr}'] = round(prob, 4)

    # 6. Classification → Đậu/Rớt
    best_clf = artifacts['best_clf']
    le       = artifacts['le']
    pred_enc = best_clf.predict(X_sc)[0]
    status   = le.inverse_transform([pred_enc])[0]
    conf = float(np.max(best_clf.predict_proba(X_sc)[0])) \
           if hasattr(best_clf, 'predict_proba') else 1.0

    # 7. Cluster sinh viên
    cluster_map = {0: 'Nhóm A (Xuất sắc)', 1: 'Nhóm B (Khá)', 2: 'Nhóm C (Yếu)'}
    cluster_id  = int(artifacts['kmeans'].predict(X_sc)[0])
    cluster_lbl = cluster_map.get(cluster_id, f'Nhóm {cluster_id}')

    return {
        'predicted_score':      round(pred_score, 2),
        'score_prob_margin':    round(score_prob_margin, 4),
        'score_distribution':   score_distribution,
        'pass_fail_status':     status,
        'pass_fail_confidence': round(conf, 4),
        'cluster':              cluster_lbl,
        'cluster_id':           cluster_id,
        'feature_values':       dict(zip(feature_names, X_raw.values[0])),
    }


# ═════════════════════════════════════════════════════════════════════════════
# WRAPPER CLASS — Tương thích ngược với app.py cũ
# ═════════════════════════════════════════════════════════════════════════════
class StudentPerformancePredictor:
    """
    Wrapper OOP để app.py có thể gọi theo interface cũ,
    đồng thời expose các hàm module mới.
    """

    def __init__(self):
        self.artifacts: dict = {}
        self.df_agg: pd.DataFrame = pd.DataFrame()

    def aggregate_student_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self.df_agg = preprocess_data(df)
        return self.df_agg

    def train_and_evaluate_models(self, df_agg: pd.DataFrame) -> dict:
        self.artifacts = train_models(df_agg)
        return {
            'best_clf_name': self.artifacts['best_clf_name'],
            'clf_acc':       self.artifacts['clf_acc'],
            'clf_f1':        self.artifacts['clf_f1'],
            'clf_kappa':     self.artifacts['clf_kappa'],
            'clf_auc':       self.artifacts['clf_auc'],
            'reg_r2':        self.artifacts['reg_r2'],
            'reg_mae':       self.artifacts['reg_mae'],
            'reg_rmse':      self.artifacts['reg_rmse'],
            'smote_size':    self.artifacts['smote_size'],
            'feat_importances': self.artifacts['feat_importances'],
            'cluster_k':        self.artifacts['cluster_k'],
            'cluster_k_range':  self.artifacts['cluster_k_range'],
            'cluster_inertias': self.artifacts['cluster_inertias'],
            'cluster_db_scores':self.artifacts['cluster_db_scores'],
            'all_model_reports': {k: {m: v for m, v in r.items() if m != 'model'}
                                  for k, r in self.artifacts['all_clf_reports'].items()},
        }

    def predict(self, student_features: dict) -> dict:
        """Interface cũ: nhận dict feature đã aggregate → trả kết quả chuẩn."""
        if not self.artifacts:
            raise ValueError("Chưa train model. Gọi train_and_evaluate_models trước.")
        feat_names = self.artifacts['feature_names']
        imputer    = self.artifacts['imputer']
        scaler     = self.artifacts['scaler']
        le         = self.artifacts['le']
        best_reg   = self.artifacts['best_reg']
        best_clf   = self.artifacts['best_clf']

        # Tạo vector từ dict (căn chỉnh thứ tự cột)
        row = {f: student_features.get(f, np.nan) for f in feat_names}
        X   = pd.DataFrame([row], columns=feat_names)
        X_imp = pd.DataFrame(imputer.transform(X), columns=feat_names)
        X_sc  = pd.DataFrame(scaler.transform(X_imp), columns=feat_names)

        # Regression
        pred_score = float(best_reg.predict(X_sc)[0])
        pred_score = max(0.0, min(10.0, pred_score))

        rf_reg = self.artifacts.get('feat_importances_rf', best_reg)
        tree_preds = ([t.predict(X_sc.values)[0] for t in rf_reg.estimators_]
                      if hasattr(rf_reg, 'estimators_') else [pred_score] * 100)
        score_prob = sum(1 for p in tree_preds if abs(p - pred_score) <= 0.5) / len(tree_preds)

        # Classification
        pred_enc = best_clf.predict(X_sc)[0]
        status   = le.inverse_transform([pred_enc])[0]
        conf     = float(np.max(best_clf.predict_proba(X_sc)[0])) \
                   if hasattr(best_clf, 'predict_proba') else 1.0

        # Score distribution
        thresholds = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        score_dist = {f'>= {t}': round(
            sum(1 for p in tree_preds if p >= t) / len(tree_preds), 4)
            for t in thresholds}

        # Cluster
        cluster_map = {0: 'Nhóm A (Xuất sắc)', 1: 'Nhóm B (Khá)', 2: 'Nhóm C (Yếu)'}
        try:
            cid = int(self.artifacts['kmeans'].predict(X_sc)[0])
        except Exception:
            cid = 0
        cluster_lbl = cluster_map.get(cid, f'Nhóm {cid}')

        # Feature Importances
        feat_imp = self.artifacts.get('feat_importances', {})

        return {
            'Predicted_Score':    round(pred_score, 2),
            'Score_Probability':  round(score_prob, 4),
            'Score_Distribution': score_dist,
            'Prediction_Status':  status,
            'Status_Probability': round(conf, 4),
            'FeatureImportance':  feat_imp,
            'Cluster':            cluster_lbl,
            'Cluster_ID':         cid,
        }

    def get_full_feature_importance(self) -> pd.DataFrame:
        """Trả về bảng importance đầy đủ + giải thích ngữ nghĩa."""
        return get_feature_importance(self.artifacts)

    def generate_explanation_prompt(self, student_id, aggregated_dict,
                                    predicted_score, score_prob, status,
                                    status_prob, feature_importances, cluster):
        fi_df = get_feature_importance(self.artifacts)
        top_features = fi_df.head(5)[['Feature Gốc', 'Đóng góp (%)', 'Ý nghĩa']].to_string(index=False)

        behavior_data = "\n".join([f"- {k}: {v:.2f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in aggregated_dict.items() if k not in ['student_id', 'pass_fail', 'final_exam_score']])

        prompt = f"""You are an Academic Advisor. Based on the ML system's output below, write a concise advisory report for the lecturer. Use Markdown formatting.

**Student:** {student_id} | **Prediction:** {status} ({status_prob*100:.1f}% confidence) | **Predicted Score:** {predicted_score}/10

**LMS Behavior Data:**
{behavior_data}

**Top Predictive Factors (ML model):**
{top_features}

**Write a short report with exactly 3 sections:**
1. 🎯 **Overall Assessment** (1-2 sentences on risk/potential level)
2. 🔍 **Root Cause Analysis** (cite specific numbers from behavior data and match to top factors)
3. 💡 **Recommended Actions** (2-3 concrete interventions for the lecturer)

Rules: Only use provided numbers. No greeting. Professional, empathetic tone. Vietnamese language."""
        return prompt.strip()


if __name__ == '__main__':
    # Kiểm tra nhanh với Mock Data
    mock_df = pd.DataFrame({
        'student_id':             ['S1','S1','S2','S2','S3','S3','S4','S4'],
        'week':                   [1,2,1,2,1,2,1,2],
        'login_count':            [5,7,1,2,8,9,2,1],
        'time_spent_minutes':     [120,150,30,40,200,210,35,25],
        'video_views':            [3,4,1,1,6,7,1,0],
        'forum_posts':            [1,2,0,0,3,4,0,0],
        'forum_reads':            [5,6,1,2,8,9,1,1],
        'materials_downloaded':   [2,3,0,1,4,5,0,0],
        'quiz_attempts':          [2,3,1,1,3,3,1,1],
        'quiz_score':             [8.0,8.5,4.0,4.5,9.0,9.5,4.0,3.5],
        'assignments_submitted':  [1,1,0,1,1,1,0,0],
        'assignments_on_time':    [1,1,0,0,1,1,0,0],
        'practice_exercises_done':[2,3,0,1,4,5,0,0],
        'messages_sent':          [1,2,0,0,2,3,0,0],
        'attendance_virtual':     [2,2,0,1,2,2,0,0],
        'midterm_score':          [8.0,8.0,4.0,4.0,9.0,9.0,3.5,3.5],
        'final_exam_score':       [8.5,8.5,4.0,4.0,9.2,9.2,3.5,3.5],
        'pass_fail':              ['Pass','Pass','Fail','Fail','Pass','Pass','Fail','Fail'],
    })
    p = StudentPerformancePredictor()
    agg = p.aggregate_student_data(mock_df)
    print(f"Aggregated: {agg.shape[1]} features cho {len(agg)} sinh viên")
    m = p.train_and_evaluate_models(agg)
    print(f"Best CLF: {m['best_clf_name']} - Acc: {m['clf_acc']:.2f}")
    print(f"Best REG: RMSE={m['reg_rmse']:.2f}, R²={m['reg_r2']:.2f}")
    fi_df = p.get_full_feature_importance()
    print(f"\nTop 5 Features:\n{fi_df[['Feature (Engineered)','Đóng góp (%)','Ý nghĩa']].head(5).to_string()}")
