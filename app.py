import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from google import genai
from streamlit_option_menu import option_menu
from ml_engine import StudentPerformancePredictor
import io

st.set_page_config(page_title="LMS Analytics", page_icon="📚", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }
    .stApp { background-color: #F8FAFC; }
    footer {visibility: hidden;}
    .block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; padding-left: 3rem !important; padding-right: 3rem !important; }
    [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E2E8F0; }
    .main-header { font-size: 26px; font-weight: 700; color: #0F172A; }
    div.stButton > button[kind="primary"] { background-color: #1E3A8A; color: white; border: none; border-radius: 6px; font-weight: 600; width: 100%; height: 44px; }
    div.stButton > button[kind="primary"]:hover { background-color: #1E40AF; }
    div[data-testid="metric-container"] { background-color: #FFFFFF; border-radius: 10px; padding: 16px 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-top: 4px solid #1E3A8A; }
    div[data-testid="stMetricLabel"] { color: #64748B; font-weight: 500; font-size: 13px; }
    /* Ẩn instructions drag-drop, giữ lại nút Browse files */
    [data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
    [data-testid="stFileUploaderDropzone"] { border: none !important; padding: 0 !important; background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──
if 'predictor' not in st.session_state:
    st.session_state.update({'predictor': StudentPerformancePredictor(), 'model_trained': False,
                              'df_raw': pd.DataFrame(), 'df_agg': pd.DataFrame(), 'train_metrics': {}})

# ── Sidebar ──
with st.sidebar:
    st.markdown("<h2 style='text-align:center;color:#1E3A8A;font-size:1.4em'>📚 Learning Performance Analysis & Prediction</h2><br>", unsafe_allow_html=True)
    st.markdown("""
    <style>
        div[role='radiogroup'] > label > div:first-child { display: none; }
        div[role='radiogroup'] label p { font-size: 16px; font-weight: 500; padding: 10px 14px; margin: 4px 0; border-radius: 8px; transition: 0.2s; color: #334155; }
        div[role='radiogroup'] label:hover p { background-color: #F1F5F9; color: #1E3A8A; }
        div[role='radiogroup'] label[data-checked="true"] p { background-color: #1E3A8A; color: white; border-left: 4px solid #3B82F6; }
    </style>
    """, unsafe_allow_html=True)
    
    _selection = st.radio(
        "Navigation",
        options=["🏠 Home", "📊 Charts & Reporting", "👤 Student Details"],
        label_visibility="collapsed"
    )
    
    menu_map = {"🏠 Home": "Home", "📊 Charts & Reporting": "Charts & Reporting", "👤 Student Details": "Student Details"}
    menu = menu_map[_selection]
    if not st.session_state['df_raw'].empty:
        st.markdown("---")
        if st.button("🔄 Reset Data", use_container_width=True):
            st.session_state.update({'model_trained': False, 'df_raw': pd.DataFrame()})
            st.session_state['predictor'] = StudentPerformancePredictor()
            st.rerun()

df_raw = st.session_state['df_raw']
predictor = st.session_state['predictor']

# ════════════════════════════════════════════
# TRANG CHỦ
# ════════════════════════════════════════════
if menu == "Home":
    if df_raw.empty:
        st.markdown("<h1 style='text-align:center;color:#0F172A;font-weight:800'>👋 Student Performance Analysis & Prediction System</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#64748B;font-size:1.2em'>Upload LMS activity history for AI automatic classification, passing probability prediction, and intervention solutions.</p>", unsafe_allow_html=True)
        _, c2, _ = st.columns([1, 6, 1])
        with c2:
            st.markdown("<div style='background:#fff;padding:2rem;border-radius:12px;box-shadow:0 4px 6px rgba(0,0,0,.08);border:2px dashed #CBD5E1'>", unsafe_allow_html=True)
            uf = st.file_uploader("Drag and drop CSV/Excel file here", type=['csv','xlsx'])
            if uf:
                try:
                    df = pd.read_csv(uf) if uf.name.endswith('.csv') else pd.read_excel(uf)
                    st.session_state['df_raw'] = df; st.rerun()
                except Exception as e: st.error(e)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<br><hr>", unsafe_allow_html=True)
            st.subheader("💡 Required File Structure")
            st.table(pd.DataFrame({
                'Column Name': ['student_id','week','pass_fail','final_exam_score','login_count','time_spent_minutes','quiz_score'],
                'Meaning': ['Student ID','Study week (1,2,3...)','Actual Pass/Fail (target label)','Exam score (target regression)','Login count','Total time spent (mins)','Quiz score']
            }))
            if st.button("🚀 Try with Mock Data"):
                st.session_state['df_raw'] = pd.DataFrame({
                    'student_id': ['S01','S01','S02','S02','S03','S03','S04','S04','S05','S05','S06','S06'],
                    'week': [1,2]*6,
                    'login_count': [5,7,1,2,6,8,2,0,8,7,1,1],
                    'time_spent_minutes': [120,150,30,45,180,200,40,10,160,140,25,20],
                    'video_views': [3,4,1,1,5,6,1,0,4,5,0,1],
                    'forum_posts': [1,2,0,0,2,3,0,0,1,2,0,0],
                    'quiz_score': [8.5,9.0,4.0,4.5,9.5,9.0,5.0,4.0,8.0,8.5,3.0,4.0],
                    'midterm_score': [8.0,8.0,4.5,4.5,9.0,9.0,4.0,4.0,7.5,7.5,3.5,3.5],
                    'final_exam_score': [8.5,8.5,4.0,4.0,9.2,9.2,3.5,3.5,8.0,8.0,3.0,3.0],
                    'pass_fail': ['Pass','Pass','Fail','Fail','Pass','Pass','Fail','Fail','Pass','Pass','Fail','Fail']
                })
                st.rerun()
    else:
        st.markdown("<p class='main-header'>Home</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        try:
            # ── HIỂN THỊ TOÀN BỘ DỮ LIỆU ĐÃ IMPORT ──
            st.subheader(f"📂 Imported Data ({len(df_raw):,} rows · {df_raw.shape[1]} columns)")
            st.dataframe(df_raw, use_container_width=True, height=320)
            st.caption(f"Columns: {', '.join(df_raw.columns.tolist())}")
            st.markdown("<br>", unsafe_allow_html=True)

            df_agg = predictor.aggregate_student_data(df_raw)
            st.session_state['df_agg'] = df_agg

            if not st.session_state['model_trained']:
                with st.spinner("Analyzing data and training AI models..."):
                    metrics = predictor.train_and_evaluate_models(df_agg)
                    st.session_state['model_trained'] = True
                    st.session_state['train_metrics'] = metrics

            if st.session_state['model_trained']:
                m = st.session_state['train_metrics']
                st.success("✅ Completed!")

                # ── ANALYSIS GRID ──
                st.subheader("📋 Student Performance Analysis Grid")
                st.caption("AI-powered grade predictions and pass/fail risk assessment for every student. Download as Excel below.")
                grid_rows = []
                for _, row_data in df_agg.iterrows():
                    sid = row_data['student_id']
                    feat_in = {k: v for k, v in row_data.items()
                               if k not in ['student_id', 'pass_fail', 'final_exam_score']}
                    try:
                        res = predictor.predict(feat_in)
                        score_dist = res.get('Score_Distribution', {})
                        actual = row_data.get('final_exam_score', None)
                        grid_rows.append({
                            'Student ID': sid,
                            'Predicted Score': res['Predicted_Score'],
                            'Score Prob. (±0.5)': f"{res['Score_Probability']*100:.1f}%",
                            'Classification': res['Prediction_Status'],
                            'Confidence': f"{res['Status_Probability']*100:.1f}%",
                            'K-Means Cluster': res['Cluster'],
                            'Prob. ≥ 5.0': f"{score_dist.get('>= 5.0',0)*100:.0f}%",
                            'Prob. ≥ 7.0': f"{score_dist.get('>= 7.0',0)*100:.0f}%",
                            'Prob. ≥ 8.0': f"{score_dist.get('>= 8.0',0)*100:.0f}%",
                        })
                    except Exception:
                        grid_rows.append({'Student ID': sid, 'Predicted Score': 'Error'})

                df_grid = pd.DataFrame(grid_rows)
                st.session_state['df_grid'] = df_grid

                # Highlight Pass/Fail
                def highlight_status(row):
                    color = '#DCFCE7' if row.get('Classification','') == 'Pass' else '#FEE2E2'
                    return [f'background-color:{color}'] * len(row)
                st.dataframe(df_grid.style.apply(highlight_status, axis=1),
                             use_container_width=True, height=400)

                # ── EXPORT EXCEL ──
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                    df_grid.to_excel(writer, sheet_name='Analysis Results', index=False)
                    df_raw.to_excel(writer, sheet_name='Raw Data', index=False)
                    df_agg.to_excel(writer, sheet_name='Aggregated Data', index=False)
                    # Sheet model comparison
                    report_rows = []
                    for mname, mdata in m.get('all_model_reports', {}).items():
                        if 'error' not in mdata:
                            report_rows.append({'Model': mname,
                                                'Accuracy (%)': round(mdata.get('accuracy',0)*100,1),
                                                'F1-Score (%)': round(mdata.get('f1',0)*100,1),
                                                "Cohen's Kappa": round(mdata.get('kappa',0),3),
                                                'AUC-ROC (%)': round(mdata.get('auc',0)*100,1) if mdata.get('auc') else None})
                    pd.DataFrame(report_rows).to_excel(writer, sheet_name='Algorithm Comparison', index=False)
                buf.seek(0)
                st.download_button(
                    label="📥 Download Excel Report (4 sheets)",
                    data=buf,
                    file_name="LMS_Analytics_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"Lỗi: {e}")
            import traceback; st.code(traceback.format_exc())

# ════════════════════════════════════════════
# BIỂU ĐỒ BÁO CÁO (CHARTS & REPORTING)
# ════════════════════════════════════════════
elif menu == "Charts & Reporting":
    st.markdown("<p class='main-header'>📊 Course Analytics Dashboard</p><br>", unsafe_allow_html=True)
    if not st.session_state['model_trained']:
        st.warning("Please run the AI Analysis on the Home page first.")
    else:
        m      = st.session_state['train_metrics']
        df_agg = st.session_state['df_agg']

        # Chart 1: Pass/Fail Pie Chart
        st.markdown("### 1. Overall Pass/Fail Distribution & Trend")
        c1_left, c1_right = st.columns([2, 1])
        with c1_left:
            vc = df_agg['pass_fail'].value_counts().reset_index()
            vc.columns = ['Status','Count']
            fig1 = px.pie(vc, names='Status', values='Count', hole=0.4,
                          color='Status', color_discrete_map={'Pass':'#1E3A8A','Fail':'#94A3B8'})
            st.plotly_chart(fig1, use_container_width=True)
        with c1_right:
            st.info("**Trend & Meaning:**\n\nShows the proportion of students who passed vs failed. If the failure rate is exceptionally high (e.g. >30%), this suggests the curriculum might be too difficult or that early-intervention systems lag. A balanced or high passing rate generally indicates healthy student engagement across the cohort.")
        st.markdown("<hr>", unsafe_allow_html=True)

        # Chart 2: Elbow Method
        st.markdown("### 2. Student Behavioral Clustering")
        c2_left, c2_right = st.columns([2, 1])
        with c2_left:
            k_range = m.get('cluster_k_range', [])
            inertias = m.get('cluster_inertias', [])
            if k_range and inertias:
                fig2 = px.line(x=k_range, y=inertias, markers=True,
                               title=f"Optimal K = {m.get('cluster_k',3)}",
                               labels={'x':'Number of Clusters (K)','y':'Inertia (WCSS)'})
                fig2.add_vline(x=m.get('cluster_k',3), line_dash="dash", line_color="#1E3A8A")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.write("Clustering data not available.")
        with c2_right:
            st.info("**Trend & Meaning:**\n\nThe Elbow Method automatically partitions your students based on their behaviors (logins, time spent, quiz metrics). Grouping into K clusters typically surfaces natural tiers such as 'Highly Engaged', 'Average', and 'At-Risk/Struggling'. Tracking how students migrate between these clusters week-over-week reveals whether the class is drifting toward disengagement.")
        st.markdown("<hr>", unsafe_allow_html=True)

        # Chart 3: Feature Importance
        st.markdown("### 3. Key Drivers of Academic Success")
        c3_left, c3_right = st.columns([2, 1])
        with c3_left:
            fi = m.get('feat_importances', {})
            if fi:
                df_fi = pd.DataFrame(list(fi.items()), columns=['Feature','Importance']).sort_values('Importance', ascending=True)
                fig3 = px.bar(df_fi, x='Importance', y='Feature', orientation='h',
                              color='Importance', color_continuous_scale='Blues', text_auto='.3f')
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.write("Feature importance data not available.")
        with c3_right:
            st.info("**Trend & Meaning:**\n\nGenerated by the Random Forest model, this reveals exactly *which* student activities predict the final grade. Often, 'quiz_score', 'time_spent_minutes_trend', or 'forum_participation_recent' end up at the top. This empowers educators to know what specific behaviors to encourage for maximum impact on learning outcomes.")
        st.markdown("<hr>", unsafe_allow_html=True)

        # Chart 4: Scatter Plot
        st.markdown("### 4. Behavior vs. Performance Correlation")
        c4_left, c4_right = st.columns([2, 1])
        with c4_left:
            # Dynamically find a valid numerical feature for the Y axis
            num_cols = [c for c in df_agg.columns if pd.api.types.is_numeric_dtype(df_agg[c]) and c not in ['student_id', 'final_exam_score', 'pass_fail']]
            tc = 'time_spent_minutes' if 'time_spent_minutes' in num_cols else (num_cols[0] if num_cols else df_agg.columns[-1])
            y_label = tc.replace('_', ' ').title()
            
            fig4 = px.scatter(df_agg, x='final_exam_score', y=tc, color='final_exam_score', hover_name='student_id',
                              labels={'final_exam_score': 'Final Exam Score (1-10)', tc: f'{y_label} (Ascending)'},
                              color_continuous_scale='RdYlGn', range_color=[1, 10])
            fig4.update_xaxes(range=[0, 10.5])
            st.plotly_chart(fig4, use_container_width=True)
        with c4_right:
            st.info("**Trend & Meaning:**\n\nPlots the Final Exam Score (1-10 on the X-axis) against a primary behavior metric like Study Time (Y-axis). This reveals the actual score distribution and visually confirms whether spending more time strictly pushes a student further to the right (higher score).")
        st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════
# CHI TIẾT SINH VIÊN (STUDENT DETAILS)
# ════════════════════════════════════════════
elif menu == "Student Details":
    st.markdown("<p class='main-header'>👤 In-Depth Student Analysis</p><br>", unsafe_allow_html=True)
    if not st.session_state['model_trained']:
        st.warning("Please run the AI Analysis on the Home page first.")
    else:
        df_agg   = st.session_state['df_agg']
        stud_id  = st.selectbox("📌 Select Student:", df_agg['student_id'].unique().tolist())

        if stud_id:
            row = df_agg[df_agg['student_id'] == stud_id].iloc[0].to_dict()
            feat_in = {k: v for k, v in row.items()
                       if k not in ['student_id', 'pass_fail', 'final_exam_score']}

            try:
                res = predictor.predict(feat_in)
            except Exception as e:
                st.error(f"Lỗi dự đoán: {e}"); st.stop()

            # ── 1. KẾT QUẢ DỰ ĐOÁN ──
            st.subheader("1. 🎯 AI Prediction Results")
            is_pass = res['Prediction_Status'].lower() == 'pass'
            accent  = "#1E3A8A" if is_pass else "#DC2626"
            icon    = "✅" if is_pass else "❌"
            st.markdown(f"""
            <div style="background:#fff;padding:24px;border-radius:12px;border-left:6px solid {accent};box-shadow:0 2px 8px rgba(0,0,0,.08);margin-bottom:16px">
                <h3 style="margin:0;color:{accent}">{icon} Prediction: {res['Prediction_Status'].upper()} &nbsp;|&nbsp; Predicted Score: <b>{res['Predicted_Score']}/10</b></h3>
            </div>
            """, unsafe_allow_html=True)

            # ── AI ADVISOR (right below Prediction Results) ──
            if st.button("✨ Cố vấn bằng AI", type="primary", use_container_width=False):
                import os
                api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
                if not api_key:
                    st.error("🔑 Vui lòng điền GEMINI_API_KEY trong file secrets.toml")
                else:
                    prompt = predictor.generate_explanation_prompt(
                        stud_id, row, res['Predicted_Score'], res['Score_Probability'],
                        res['Prediction_Status'], res['Status_Probability'],
                        res['FeatureImportance'], res.get('Cluster', 'N/A')
                    )
                    with st.spinner("AI đang phân tích..."):
                        try:
                            from google import genai
                            client = genai.Client(api_key=api_key)
                            response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
                            st.success("✅ Phân tích xong!")
                            st.code(response.text, language="markdown")
                        except Exception:
                            try:
                                response = client.models.generate_content(model='gemini-2.0-flash-exp', contents=prompt)
                                st.success("✅ Phân tích xong!")
                                st.code(response.text, language="markdown")
                            except Exception as e2:
                                st.error(f"Lỗi AI: {e2}")

            st.subheader("2. 📈 Score Probability Distribution")
            st.caption("Over 100 decision trees voted to calculate the chance of reaching score thresholds.")
            c_dist1, c_dist2 = st.columns([2, 1])
            with c_dist1:
                score_dist = res.get('Score_Distribution', {})
                if score_dist:
                    df_dist = pd.DataFrame(list(score_dist.items()), columns=['Score Threshold', 'Probability'])
                    df_dist['Probability (%)'] = df_dist['Probability'] * 100
                    fig_dist = px.bar(df_dist, x='Score Threshold', y='Probability (%)',
                                      title="Probability per Score Threshold", text_auto='.1f',
                                      color='Probability (%)', color_continuous_scale='Blues')
                    fig_dist.update_layout(coloraxis_showscale=False, yaxis_range=[0, 105])
                    st.plotly_chart(fig_dist, use_container_width=True)
            with c_dist2:
                p50 = score_dist.get('>= 5.0', 0) * 100
                p70 = score_dist.get('>= 7.0', 0) * 100
                st.markdown(f"""
                <div style="background:#F0FDF4;padding:16px;border-radius:10px;border:1px solid #86EFAC;margin-top:1rem">
                  <p style="margin:0;color:#166534;font-weight:700">Prediction Summary</p>
                  <p style="margin:4px 0 0;font-size:2em;font-weight:800;color:#15803D">{res['Predicted_Score']}<span style="font-size:0.5em;color:#6B7280">/10</span></p>
                  <hr style="border-color:#D1FAE5;margin:8px 0">
                  <p style="margin:0;font-size:0.9em">Confidence ±0.5: <b>{res['Score_Probability']*100:.1f}%</b></p>
                  <p style="margin:4px 0 0;font-size:0.9em">Probability ≥ 5.0: <b>{p50:.0f}%</b></p>
                  <p style="margin:4px 0 0;font-size:0.9em">Probability ≥ 7.0: <b>{p70:.0f}%</b></p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── 3. FEATURE IMPORTANCE FULL ──
            st.subheader("3. 🔬 Actionable Feature Importance")
            st.caption("Every behavior is rated by its impact on the final exam score—even the least important variables.")
            fim = res['FeatureImportance']
            if isinstance(fim, dict):
                df_fim = pd.DataFrame(list(fim.items()), columns=['Feature', 'Weight'])
                df_fim['Weight (%)'] = df_fim['Weight'] * 100
                df_fim = df_fim.sort_values('Weight (%)', ascending=True)
                fig_fim = px.bar(df_fim.tail(15), x='Weight (%)', y='Feature', orientation='h',
                                 title="Top Drivers of Final Exam Score",
                                 color='Weight (%)', color_continuous_scale='Blues', text_auto='.1f')
                fig_fim.update_layout(coloraxis_showscale=False, height=450)
                st.plotly_chart(fig_fim, use_container_width=True)

            # ── 4. SO SÁNH VỚI LỚP ──
            st.subheader("4. 📡 Radar Chart: Student vs Class Average")
            # Dynamically select all available numerical features for a comprehensive comparison
            all_num_cols = [c for c in feat_in if isinstance(feat_in[c], (int, float)) and c in df_agg.columns]
            avail_cols = all_num_cols
            
            if avail_cols:
                class_means  = df_agg[avail_cols].mean().to_dict()
                student_vals = [feat_in[c] for c in avail_cols]
                class_vals   = [class_means.get(c, 0) for c in avail_cols]
                radar_cats   = [c.replace('_', ' ').title() for c in avail_cols]
                max_vals     = [max(sv, cv, 1e-9) for sv, cv in zip(student_vals, class_vals)]
                s_norm = [sv / mv for sv, mv in zip(student_vals, max_vals)]
                c_norm = [cv / mv for cv, mv in zip(class_vals, max_vals)]
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(r=s_norm + [s_norm[0]], theta=radar_cats + [radar_cats[0]],
                                                    fill='toself', name=f'Student {stud_id}', line_color='#1E3A8A'))
                fig_radar.add_trace(go.Scatterpolar(r=c_norm + [c_norm[0]], theta=radar_cats + [radar_cats[0]],
                                                    fill='toself', name='Class Avg', line_color='#94A3B8', opacity=0.6))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                        title="Student's standing relative to the rest of the class")
                st.plotly_chart(fig_radar, use_container_width=True)

