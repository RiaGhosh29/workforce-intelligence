import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Workforce Intelligence", page_icon="🧠", layout="wide")

@st.cache_resource
def load_model():
    model        = joblib.load('xgb_attrition_model.pkl')
    preprocessor = joblib.load('preprocessor_xgb.pkl')
    features     = joblib.load('feature_names.pkl')
    shap_vals    = np.load('shap_values_test.npy')
    return model, preprocessor, features, shap_vals

@st.cache_data
def load_data():
    scored       = pd.read_csv('workforce_intelligence_scored.csv')
    displacement = pd.read_csv('displacement_scores.csv')
    cost_summary = pd.read_csv('cost_simulation_summary.csv')
    return scored, displacement, cost_summary

model, preprocessor, features, shap_vals = load_model()
scored, displacement, cost_summary       = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.title("🧠 AI Workforce Intelligence")
st.sidebar.markdown("Ria Ghosh & Izabella Martirosyan  \nUniversidad Europea de Madrid 2025–26")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "🔮 Module 1 — Attrition Risk",
    "💰 Module 2 — Cost Simulator",
    "🤖 Module 3 — AI Displacement",
    "🎯 Module 4 — 2×2 Matrix"
])

# ══════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("🧠 AI-Assisted Workforce Intelligence")
    st.markdown("**Predictive Attrition · Layoff Cost Simulation · AI Displacement Scoring**")
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Employees Scored", f"{len(scored):,}")
    col2.metric("High Attrition Risk", f"{(scored['Attrition_Probability'] >= 0.50).sum()}")
    col3.metric("Avg Attrition Probability", f"{scored['Attrition_Probability'].mean():.1%}")
    if 'Quadrant' in scored.columns:
        col4.metric("Double Exposure", f"{(scored['Quadrant']=='Double Exposure').sum()}")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Module 1** — XGBoost model predicts who is likely to leave and explains why via SHAP.")
        st.info("**Module 2** — Financial simulator models the cost of attrition and planned restructuring.")
    with col2:
        st.info("**Module 3** — Composite score rates each role's exposure to AI automation.")
        st.info("**Module 4** — Agile 2×2 matrix combines all three outputs for sprint-based HR planning.")

# ══════════════════════════════════════════════════════════════════════
elif page == "🔮 Module 1 — Attrition Risk":
    st.title("🔮 Module 1 — Attrition Risk Prediction")
    st.markdown("XGBoost classifier with SHAP explainability. Target AUC ≥ 0.80 (Objective 1).")
    st.markdown("---")

    threshold = st.slider("Risk threshold", 0.1, 0.9, 0.5, 0.05)
    high_risk = scored[scored['Attrition_Probability'] >= threshold]

    col1, col2, col3 = st.columns(3)
    col1.metric("High-risk employees", len(high_risk))
    col2.metric("% of workforce", f"{len(high_risk)/len(scored)*100:.1f}%")
    col3.metric("Avg risk probability", f"{scored['Attrition_Probability'].mean():.3f}")

    col1, col2 = st.columns(2)
    with col1:
        risk_counts = scored['Risk_Tier'].value_counts().reset_index()
        risk_counts.columns = ['Risk Tier','Count']
        fig = px.bar(risk_counts, x='Risk Tier', y='Count',
                     color='Risk Tier',
                     color_discrete_map={'High':'#ED7D31','Medium':'#FFC000','Low':'#4472C4'},
                     title="Employees by Attrition Risk Tier")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.histogram(scored, x='Attrition_Probability', nbins=30,
                            title="Distribution of Attrition Probabilities",
                            color_discrete_sequence=['#4472C4'])
        fig2.add_vline(x=threshold, line_dash='dash', line_color='red')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Top 10 Highest Risk Employees")
    display_cols = ['Attrition_Probability','Risk_Tier']
    for c in ['Department','JobRole','MonthlyIncome','JobLevel']:
        if c in scored.columns:
            display_cols.append(c)
    st.dataframe(
        scored.nlargest(10, 'Attrition_Probability')[display_cols]
        .style.format({'Attrition_Probability':'{:.3f}'}),
        use_container_width=True
    )

    st.markdown("### SHAP Feature Importance")
    try:
        mean_shap = pd.Series(
            np.abs(shap_vals).mean(axis=0), index=features
        ).sort_values(ascending=True).tail(15)
        fig_s = px.bar(mean_shap, orientation='h',
                       title="Top 15 Features by Mean |SHAP| Value",
                       labels={'value':'Mean |SHAP|','index':'Feature'})
        fig_s.update_traces(marker_color='#4472C4')
        st.plotly_chart(fig_s, use_container_width=True)
        st.caption("Higher SHAP value = stronger influence on attrition prediction. "
                   "Source: Lundberg and Lee (2017); Lundberg et al. (2020).")
    except Exception as e:
        st.warning(f"SHAP chart unavailable: {e}")

# ══════════════════════════════════════════════════════════════════════
elif page == "💰 Module 2 — Cost Simulator":
    st.title("💰 Module 2 — Layoff Cost Simulator")
    st.markdown("Parameterised financial engine grounded in Cascio (2006) and SHRM (2022) benchmarks.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "Scenario 1 — Voluntary Attrition",
        "Scenario 2 — Planned RIF",
        "Scenario 3 — Combined"
    ])

    with tab1:
        st.markdown("### Voluntary Attrition Cost Projection")
        st.markdown("Estimates total replacement cost for employees flagged as high attrition risk.")
        col1, col2 = st.columns(2)
        with col1:
            recruit_pct   = st.slider("Recruitment cost (% of annual salary)", 10, 40, 15) / 100
            assess_pct    = st.slider("Assessment cost (% of annual salary)",   2,  15,  5) / 100
            onboard_pct   = st.slider("Onboarding cost (% of annual salary)",   5,  25, 10) / 100
        with col2:
            rampup_months  = st.slider("Productivity ramp-up (months)",  1, 12, 3)
            knowledge_mult = st.slider("Knowledge transfer multiplier",  0.1, 1.5, 0.5)
            threshold_s1   = st.slider("Attrition risk threshold",       0.3, 0.8, 0.5, 0.05)

        hr = scored[scored['Attrition_Probability'] >= threshold_s1]
        if len(hr) > 0 and 'MonthlyIncome' in hr.columns:
            total = (
                (hr['MonthlyIncome']*12*recruit_pct) +
                (hr['MonthlyIncome']*12*assess_pct) +
                (hr['MonthlyIncome']*12*onboard_pct) +
                (hr['MonthlyIncome']*rampup_months) +
                (hr['MonthlyIncome']*knowledge_mult)
            ).sum()
            col1, col2, col3 = st.columns(3)
            col1.metric("High-risk headcount", len(hr))
            col2.metric("Total estimated cost", f"${total:,.0f}")
            col3.metric("Average per person",   f"${total/len(hr):,.0f}")
        else:
            st.info("Adjust threshold or check MonthlyIncome is in the scored dataset.")

    with tab2:
        st.markdown("### Planned Reduction-in-Force Simulation")
        rif_pct      = st.slider("Target headcount reduction (%)", 1, 30, 10) / 100
        sev_wks      = st.slider("Severance weeks per year of service", 1, 6, 2)
        n_cut        = max(1, int(len(scored) * rif_pct))
        st.metric("Employees affected", n_cut)
        if 'MonthlyIncome' in scored.columns and 'YearsAtCompany' in scored.columns:
            sample       = scored.nsmallest(n_cut, 'Attrition_Probability')
            severance    = (sample['MonthlyIncome'] * sample['YearsAtCompany'] * sev_wks / 4).sum()
            knowledge    = (sample['MonthlyIncome'] * 0.5).sum()
            morale       = (sample['MonthlyIncome'] * 0.15 * 2).sum()
            rif_total    = severance + knowledge + morale
            col1, col2, col3 = st.columns(3)
            col1.metric("Total RIF cost",   f"${rif_total:,.0f}")
            col2.metric("Severance",         f"${severance:,.0f}")
            col3.metric("Indirect costs",    f"${knowledge+morale:,.0f}")

    with tab3:
        st.markdown("### Combined Risk Scenario")
        try:
            s1 = float(cost_summary['scenario_1_total'].values[0])
            s2 = float(cost_summary['scenario_2_total'].values[0])
            s3 = s1 + s2
            df_s = pd.DataFrame({
                'Scenario': ['Voluntary Attrition','Planned RIF','Combined Total'],
                'Cost':     [s1, s2, s3]
            })
            fig = px.bar(df_s, x='Scenario', y='Cost',
                         color='Scenario',
                         color_discrete_sequence=['#4472C4','#ED7D31','#C00000'],
                         title="Total Workforce Cost Exposure by Scenario")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            col1, col2, col3 = st.columns(3)
            col1.metric("Voluntary attrition", f"${s1:,.0f}")
            col2.metric("Planned RIF",          f"${s2:,.0f}")
            col3.metric("Combined total",        f"${s3:,.0f}")
        except:
            st.info("Run Colab notebook Section 5 to generate cost simulation results.")

# ══════════════════════════════════════════════════════════════════════
elif page == "🤖 Module 3 — AI Displacement":
    st.title("🤖 Module 3 — AI Displacement Risk")
    st.markdown("Role-level composite score (0–100) combining O\\*NET task data and Frey & Osborne (2017).")
    st.markdown("---")

    adoption = st.slider("Company AI Adoption Speed", 0.5, 1.5, 1.0, 0.1,
                         help="1.0 = industry average. Higher = faster AI adoption.")
    disp = displacement.copy()
    disp['Adjusted Score'] = (disp['Displacement Score'] * adoption).clip(0,100).round(1)
    disp['Risk Tier'] = disp['Adjusted Score'].apply(
        lambda x: 'High' if x > 65 else ('Medium' if x >= 35 else 'Low')
    )
    disp_sorted = disp.sort_values('Adjusted Score', ascending=True)

    fig = px.bar(disp_sorted, x='Adjusted Score', y='JobRole',
                 orientation='h', color='Risk Tier',
                 color_discrete_map={'High':'#ED7D31','Medium':'#FFC000','Low':'#4472C4'},
                 text='Adjusted Score',
                 title="AI Displacement Risk Score by Job Role (0–100)")
    fig.add_vline(x=65, line_dash='dash', line_color='red',    opacity=0.5, annotation_text="High threshold")
    fig.add_vline(x=35, line_dash='dash', line_color='orange', opacity=0.5, annotation_text="Low threshold")
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Scoring Methodology")
    st.markdown("""
| Input | Weight | Source |
|-------|--------|--------|
| Task repetitiveness | 40% | O\\*NET task composition data (US Dept of Labor) |
| AI substitutability | 40% | Frey & Osborne (2017) automation probabilities |
| Dept AI adoption speed | 20% | McKinsey Global Institute (2023) sector analysis |

**Risk tiers:** High > 65 · Medium 35–65 · Low < 35
    """)

    st.markdown("### Role Mapping Table")
    st.dataframe(disp[['JobRole','Displacement Score','Risk Tier',
                        'Automation Prob','Task Repetitiveness']],
                 use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
elif page == "🎯 Module 4 — 2×2 Matrix":
    st.title("🎯 Module 4 — Strategic 2×2 Prioritisation Matrix")
    st.markdown("Agile-aligned MoSCoW prioritisation — attrition risk × AI displacement risk.")
    st.caption("Based on Clegg and Barker (1994) MoSCoW method and McMackin & Heffernan (2021) Agile HR framework.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        at = st.slider("Attrition risk threshold",    0.2, 0.8, 0.5, 0.05)
    with col2:
        dt = st.slider("Displacement score threshold", 20,  80,  50,  5)

    plot_df = scored.copy()

    if 'AI_Displacement_Score' not in plot_df.columns:
        score_lookup = dict(zip(displacement['JobRole'], displacement['Displacement Score']))
        plot_df['AI_Displacement_Score'] = plot_df.get('JobRole', pd.Series()).map(score_lookup).fillna(50)

    def quadrant(row):
        ha = row['Attrition_Probability']    >= at
        hd = row['AI_Displacement_Score'] >= dt
        if ha and hd:   return 'Double Exposure'
        elif ha:         return 'Retention Priority'
        elif hd:         return 'Restructuring Candidate'
        else:            return 'Stable'

    plot_df['Quadrant'] = plot_df.apply(quadrant, axis=1)

    color_map = {
        'Double Exposure'        : '#C00000',
        'Retention Priority'     : '#ED7D31',
        'Restructuring Candidate': '#FFC000',
        'Stable'                 : '#4472C4'
    }

    hover = {c: True for c in
             ['Attrition_Probability','AI_Displacement_Score','Quadrant'] +
             [c for c in ['Department','JobRole','MonthlyIncome'] if c in plot_df.columns]}

    fig = px.scatter(plot_df,
                     x='Attrition_Probability',
                     y='AI_Displacement_Score',
                     color='Quadrant',
                     color_discrete_map=color_map,
                     hover_data=hover,
                     title="Strategic Workforce Prioritisation Matrix",
                     labels={'Attrition_Probability':'Attrition Risk  →',
                             'AI_Displacement_Score':'AI Displacement Risk  →'},
                     height=580)
    fig.add_vline(x=at, line_dash='dash', line_color='grey', opacity=0.6)
    fig.add_hline(y=dt, line_dash='dash', line_color='grey', opacity=0.6)
    fig.update_traces(marker=dict(size=9, opacity=0.8))
    fig.update_layout(plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Action Plan by Quadrant")
    q_counts = plot_df['Quadrant'].value_counts()
    actions = {
        'Double Exposure'        : ('🔴 Must-have',    'Immediate combined retention and reskilling action'),
        'Retention Priority'     : ('🟠 Should-have',  'Retention investment — compensation review, career development'),
        'Restructuring Candidate': ('🟡 Should-have',  'Reskilling — longer-horizon capability redirection'),
        'Stable'                 : ('🔵 Could-have',   'Monitor at next quarterly planning cycle'),
    }
    for q, (moscow, action) in actions.items():
        n = q_counts.get(q, 0)
        pct = n / len(plot_df) * 100
        st.markdown(f"**{q}** — {n} employees ({pct:.1f}%) — {moscow} — {action}")

st.markdown("---")
st.caption("AI-Assisted Workforce Intelligence | Ria Ghosh & Izabella Martirosyan | "
           "Universidad Europea de Madrid 2025–26 | "
           "Supervisor: Prof. Ximena Alejandra Valente Hervier")
