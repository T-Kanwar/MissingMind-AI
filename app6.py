import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency, shapiro, skew, kurtosis
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Missing Data Intelligence Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-title {
        font-size: 2.4rem; font-weight: 700; color: #b4b4d9;
        border-bottom: 3px solid #4f8ef7; padding-bottom: 10px; margin-bottom: 6px;
    }
    .sub-title { font-size: 1.0rem; color: #555; margin-bottom: 30px; }
    .section-header {
        font-size: 1.4rem; font-weight: 600; color: #1a1a2e;
        background: linear-gradient(90deg, #eef2ff, transparent);
        padding: 10px 16px; border-left: 4px solid #4f8ef7;
        border-radius: 4px; margin: 30px 0 16px 0;
    }
    .insight-box {
        background: #f0f7ff; border: 1px solid #bdd5ff;
        border-radius: 8px; padding: 16px 20px; margin: 12px 0;
    }
    .insight-box li { margin: 6px 0; color: #1a3a6e; font-size: 0.92rem; }
    .theory-box {
        background: #fafafa; border: 1px solid #e0e0e0;
        border-radius: 8px; padding: 16px 20px; margin: 12px 0;
    }
    .theory-box h4 { color: #333; margin-bottom: 8px; }
    .theory-box p  { color: #555; font-size: 0.91rem; line-height: 1.6; }
    .metric-card {
        background: white; border-radius: 10px; padding: 18px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;
    }
    .metric-card .val { font-size: 2rem; font-weight: 700; color: #4f8ef7; }
    .metric-card .lbl { font-size: 0.82rem; color: #666; margin-top: 4px; }
    .badge-mcar { background:#d4edda; color:#155724; padding:3px 10px; border-radius:12px; font-size:0.82rem; font-weight:600; }
    .badge-mar  { background:#fff3cd; color:#856404; padding:3px 10px; border-radius:12px; font-size:0.82rem; font-weight:600; }
    .badge-mnar { background:#f8d7da; color:#721c24; padding:3px 10px; border-radius:12px; font-size:0.82rem; font-weight:600; }
    /* Per-column analysis */
    .col-stat-card {
        background: white; border-radius: 10px; padding: 14px 18px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.07); text-align: center;
    }
    .col-stat-card .cv { font-size: 1.5rem; font-weight: 700; color: #1a1a2e; }
    .col-stat-card .ck { font-size: 0.75rem; color: #888; margin-top: 3px;
        text-transform: uppercase; letter-spacing: .05em; }
    .verdict-mcar  { background:#edfaf3; border:2px solid #89d9ac; border-radius:10px; padding:14px 18px; }
    .verdict-mar   { background:#fffaeb; border:2px solid #f0cc7a; border-radius:10px; padding:14px 18px; }
    .verdict-mnar  { background:#fff0ed; border:2px solid #f5a898; border-radius:10px; padding:14px 18px; }
    .verdict-clean { background:#edfaf3; border:2px solid #89d9ac; border-radius:10px; padding:14px 18px; }
    .strat-chip { display:inline-block; padding:4px 14px; border-radius:20px;
        font-size:0.82rem; font-weight:600; margin:3px 3px; }
    .chip-green  { background:#d4edda; color:#155724; border:1px solid #89d9ac; }
    .chip-yellow { background:#fff3cd; color:#856404; border:1px solid #f0cc7a; }
    .chip-red    { background:#f8d7da; color:#721c24; border:1px solid #f5a898; }
    .chip-blue   { background:#dce3ff; color:#2a3da0; border:1px solid #bdc8f5; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS — global analysis
# ─────────────────────────────────────────────
def severity(pct):
    if pct < 5:  return "Low"
    if pct < 20: return "Moderate"
    return "High"

def identify_columns(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return num_cols, cat_cols

def missing_summary(df, num_cols, cat_cols):
    rows = []
    for col in df.columns:
        mc  = df[col].isnull().sum()
        pct = mc / len(df) * 100
        dtype = "Numerical" if col in num_cols else "Categorical"
        rows.append({"Column": col, "Data Type": dtype,
                     "Missing Count": mc, "Missing %": round(pct, 2)})
    result = pd.DataFrame(rows).sort_values("Missing %", ascending=False).reset_index(drop=True)
    # ── CHANGE: filter out rows with zero missing values ──
    result = result[result["Missing Count"] > 0].reset_index(drop=True)
    return result

def plot_missing_heatmap(df):
    missing_cols = [c for c in df.columns if df[c].isnull().any()]
    if not missing_cols:
        return None
    sorted_cols = sorted(missing_cols, key=lambda c: df[c].isnull().mean(), reverse=True)
    mask_df = df[sorted_cols].isnull().astype(int)
    fig, ax = plt.subplots(figsize=(max(10, len(sorted_cols) * 0.7), 5))
    sns.heatmap(mask_df.T, cmap="Blues", cbar=True,
                yticklabels=sorted_cols, xticklabels=False, linewidths=0, ax=ax)
    ax.set_title("Missing Value Heatmap (rows × columns)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Rows (observations)", fontsize=10)
    ax.set_ylabel("Columns", fontsize=10)
    plt.tight_layout()
    return fig

def plot_missingness_correlation(df):
    missing_cols = [c for c in df.columns if df[c].isnull().any()]
    if len(missing_cols) < 2:
        return None
    miss_bin = df[missing_cols].isnull().astype(int)
    corr = miss_bin.corr()
    fig, ax = plt.subplots(figsize=(max(7, len(missing_cols) * 0.9), max(6, len(missing_cols) * 0.8)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                mask=mask, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Missingness Correlation Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df, num_cols):
    valid = [c for c in num_cols if df[c].isnull().mean() < 1.0]
    if len(valid) < 2:
        return None, pd.DataFrame()
    corr = df[valid].corr()
    strong = (corr.abs() > 0.5) & (corr != 1.0)
    if not strong.any().any():
        return None, pd.DataFrame()
    fig, ax = plt.subplots(figsize=(max(8, len(valid) * 0.9), max(7, len(valid) * 0.8)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    display_corr = corr.where(corr.abs() > 0.5)
    sns.heatmap(display_corr, annot=False, cmap="RdYlGn", center=0,
                mask=mask, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Strong Correlations (|r| > 0.5) — Numerical Features", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    pairs = []
    seen = set()
    for i, c1 in enumerate(corr.columns):
        for j, c2 in enumerate(corr.columns):
            if i >= j: continue
            v = corr.loc[c1, c2]
            if abs(v) > 0.5:
                key = tuple(sorted([c1, c2]))
                if key not in seen:
                    seen.add(key)
                    # ── CHANGE: added Correlation % column ──
                    pairs.append({
                        "Column A": c1,
                        "Column B": c2,
                        "Correlation": round(v, 4),
                        "Correlation %": f"{round(v * 100, 2)}%",
                    })
    corr_table = pd.DataFrame(pairs).sort_values("Correlation", key=abs, ascending=False).reset_index(drop=True)
    return fig, corr_table

def diagnose_mechanism(df, col, num_cols):
    miss_mask  = df[col].isnull().astype(int)
    predictors = [c for c in df.columns if c != col and df[c].isnull().mean() < 0.9]
    if not predictors or miss_mask.sum() < 5:
        return "MNAR", "Insufficient data to test; assumed MNAR."
    mcar_p_vals = []
    for p in predictors:
        if p in num_cols and df[p].dropna().nunique() > 1:
            try:
                binned = pd.qcut(df[p].fillna(df[p].median()), q=4, duplicates="drop", labels=False)
                ct = pd.crosstab(binned, miss_mask)
                if ct.shape[0] > 1 and ct.shape[1] > 1:
                    _, p_val, _, _ = chi2_contingency(ct)
                    mcar_p_vals.append(p_val)
            except Exception:
                pass
    if mcar_p_vals and np.mean(mcar_p_vals) > 0.05:
        return "MCAR", (f"Chi-square tests show no significant dependency "
                        f"(avg p={np.mean(mcar_p_vals):.3f} > 0.05). Missingness appears random.")
    try:
        X_pred = df[predictors].copy()
        for c in X_pred.select_dtypes(include="object").columns:
            X_pred[c] = X_pred[c].astype("category").cat.codes
        X_pred = X_pred.fillna(X_pred.median(numeric_only=True))
        scaler  = StandardScaler()
        X_scaled = scaler.fit_transform(X_pred)
        lr = LogisticRegression(max_iter=300, solver="lbfgs")
        lr.fit(X_scaled, miss_mask)
        score    = lr.score(X_scaled, miss_mask)
        baseline = max(miss_mask.mean(), 1 - miss_mask.mean())
        if score > baseline + 0.05:
            return "MAR", (f"Logistic Regression predicts missingness with accuracy {score:.2%} "
                           f"(baseline {baseline:.2%}). Missingness is related to observed variables.")
    except Exception:
        pass
    return "MNAR", "Missingness not explained by observed data. Likely related to the missing value itself — assumed MNAR."

def detect_outliers_iqr(series):
    s = series.dropna()
    if len(s) < 4: return 0
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    return int(((s < Q1 - 1.5 * IQR) | (s > Q3 + 1.5 * IQR)).sum())

def variance_impact(series):
    s = series.dropna()
    if len(s) < 2: return 0.0, 0.0, 0.0
    var_before = float(s.var())
    var_after  = float(series.fillna(s.mean()).var())
    return round(var_before, 4), round(var_after, 4), round(var_before - var_after, 4)


# ─────────────────────────────────────────────
# HELPERS — per-column deep analysis
# ─────────────────────────────────────────────
def stat_card(label, value, color="#1a1a2e"):
    return (f'<div class="col-stat-card">'
            f'<div class="cv" style="color:{color};">{value}</div>'
            f'<div class="ck">{label}</div></div>')

def plot_numerical_column(df, col):
    s_original = df[col].dropna()
    s_imputed = df[col].fillna(s_original.mean())
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7)) 
    fig.suptitle(f"Deep Distribution Analysis — {col}", fontsize=16, fontweight="bold", y=1.05)

    sns.kdeplot(s_original, ax=axes[0], color="#4f8ef7", linewidth=3, label="Original (Before)", fill=True, alpha=0.2)
    sns.kdeplot(s_imputed, ax=axes[0], color="#e07b54", linewidth=3, label="Mean Imputed (After)", linestyle="--")
    axes[0].set_title("Distribution Shift: Original vs. Imputed", fontsize=14)
    axes[0].legend()

    box_data = pd.DataFrame({
        "Value": pd.concat([s_original, s_imputed]),
        "Type": ["Original"] * len(s_original) + ["Imputed"] * len(s_imputed)
    })
    sns.boxplot(data=box_data, x="Type", y="Value", ax=axes[1], palette=["#dce3ff", "#fce4d6"])
    axes[1].set_title("Variance & Outlier Comparison", fontsize=14)
    
    plt.tight_layout()
    return fig

def plot_categorical_column(df, col, top_n=10):
    s_original = df[col].dropna()
    s_imputed = df[col].fillna(s_original.mode()[0] if not s_original.empty else "N/A")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"Categorical Frequency Analysis — {col}", fontsize=16, fontweight="bold", y=1.05)

    orig_counts = s_original.value_counts().head(top_n)
    imp_counts = s_imputed.value_counts().head(top_n)
    
    compare_df = pd.DataFrame({"Original": orig_counts, "Imputed (Mode)": imp_counts}).fillna(0)
    compare_df.plot(kind='barh', ax=axes[0], color=["#4f8ef7", "#e07b54"], width=0.8)
    axes[0].set_title(f"Top {top_n} Categories: Original vs Mode Imputed", fontsize=14)
    axes[0].invert_yaxis()

    top_pie = imp_counts.head(8)
    axes[1].pie(top_pie, labels=top_pie.index.astype(str), autopct="%1.1f%%", 
                startangle=140, colors=plt.cm.Pastel1.colors, wedgeprops={'edgecolor': 'white'})
    axes[1].set_title("Final Proportion (After Imputation)", fontsize=14)

    plt.tight_layout()
    return fig

def plot_missing_vs_features(df, col):
    num_others = [c for c in df.select_dtypes(include=[np.number]).columns
                  if c != col and df[c].isnull().mean() < 0.95]
    if not num_others:
        return None
    means_present = df[df[col].notna()][num_others].mean()
    means_missing = df[df[col].isnull()][num_others].mean()
    diff_df = pd.DataFrame({"Present": means_present, "Missing": means_missing}).dropna().head(12)
    if diff_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(max(8, len(diff_df) * 0.9), 4))
    x = np.arange(len(diff_df)); w = 0.35
    ax.bar(x - w/2, diff_df["Present"], w, label="Present rows", color="#4f8ef7", alpha=0.85)
    ax.bar(x + w/2, diff_df["Missing"], w, label="Missing rows",  color="#e07b54", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(diff_df.index, rotation=35, ha="right", fontsize=9)
    ax.set_title(f"Feature Means — Rows where '{col}' is Present vs Missing",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean value")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig

def strategy_chips_html(mech, miss_pct, col_type):
    chips = []
    if mech == "CLEAN":
        return '<span class="strat-chip chip-green">✅ No action needed — column is complete</span>'
    if miss_pct > 50:
        chips.append(("⚠ Consider Dropping Column (>50% missing)", "chip-red"))
    if mech == "MCAR":
        if miss_pct < 5:
            chips.append(("Listwise Deletion (safe)", "chip-green"))
        chips.append(("Median Imputation" if col_type == "Numerical" else "Mode Imputation", "chip-green"))
    if mech == "MAR":
        chips.append(("KNN Imputation", "chip-blue"))
        chips.append(("Iterative Imputer (MICE)", "chip-blue"))
        chips.append(("Group-wise Imputation", "chip-blue"))
        if miss_pct >= 10:
            chips.append(("Create Missing Indicator (≥10% MAR)", "chip-yellow"))
    if mech == "MNAR":
        chips.append(("⚠ Create Missing Indicator FIRST (mandatory)", "chip-red"))
        chips.append(("Constant / Domain-Specific Value", "chip-yellow"))
        chips.append(("Sensitivity Analysis Required", "chip-yellow"))
    return " ".join(f'<span class="strat-chip {cls}">{lbl}</span>' for lbl, cls in chips)

def render_per_column_analysis(df, col, num_cols, cat_cols, mechanism_results):
    miss_count = int(df[col].isnull().sum())
    miss_pct   = round(df[col].isnull().mean() * 100, 2)
    total_rows = len(df)
    present    = total_rows - miss_count
    col_type   = "Numerical" if col in num_cols else "Categorical"

    mech_info   = mechanism_results.get(col, {})
    mech        = mech_info.get("mechanism", "N/A")
    mech_reason = mech_info.get("reason", "Run the global diagnosis section above first.")
    sev         = severity(miss_pct) if miss_pct > 0 else "None"

    miss_color = "#dc2626" if miss_pct >= 20 else "#d97706" if miss_pct >= 5 else "#16a34a"
    sev_color  = "#dc2626" if sev == "High" else "#d97706" if sev == "Moderate" else "#16a34a"
    mech_color = {"MCAR": "#155724", "MAR": "#856404", "MNAR": "#721c24"}.get(mech, "#444")

    st.markdown(f"#### 🔍 Deep Analysis — `{col}` &nbsp;·&nbsp; {col_type}", unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: st.markdown(stat_card("Total Rows", f"{total_rows:,}"), unsafe_allow_html=True)
    with m2: st.markdown(stat_card("Present",    f"{present:,}"),    unsafe_allow_html=True)
    with m3: st.markdown(stat_card("Missing",    f"{miss_pct}%",  miss_color), unsafe_allow_html=True)
    with m4: st.markdown(stat_card("Severity",   sev,             sev_color),  unsafe_allow_html=True)
    with m5: st.markdown(stat_card("Mechanism",  mech,            mech_color), unsafe_allow_html=True)
    st.markdown("")

    if col_type == "Numerical":
        s = df[col].dropna()
        if len(s) > 1:
            col_skew = float(skew(s))
            col_kurt = float(kurtosis(s))
            Q1, Q3   = float(s.quantile(0.25)), float(s.quantile(0.75))
            IQR      = Q3 - Q1
            n_out    = detect_outliers_iqr(df[col])
            vb, va, vi = variance_impact(df[col])
            out_pct  = n_out / max(len(s), 1)

            r1 = st.columns(4)
            for (lbl, val), col_ui in zip(
                [("Mean", f"{s.mean():.4g}"), ("Median", f"{s.median():.4g}"),
                 ("Std Dev", f"{s.std():.4g}"), ("Variance", f"{s.var():.4g}")], r1):
                with col_ui: st.markdown(stat_card(lbl, val), unsafe_allow_html=True)

            st.markdown("")
            r2 = st.columns(4)
            for (lbl, val), col_ui in zip(
                [("Min", f"{s.min():.4g}"), ("Max", f"{s.max():.4g}"),
                 ("Skewness", f"{col_skew:.3f}"), ("Kurtosis", f"{col_kurt:.3f}")], r2):
                with col_ui: st.markdown(stat_card(lbl, val), unsafe_allow_html=True)

            st.markdown("")
            r3 = st.columns(4)
            out_color = "#dc2626" if out_pct > 0.15 else "#d97706" if out_pct > 0.05 else "#16a34a"
            for (lbl, val, clr), col_ui in zip(
                [("Q1", f"{Q1:.4g}", "#1a1a2e"),
                 ("Q3", f"{Q3:.4g}", "#1a1a2e"),
                 ("IQR", f"{IQR:.4g}", "#1a1a2e"),
                 ("Outliers (IQR)", str(n_out), out_color)], r3):
                with col_ui: st.markdown(stat_card(lbl, val, clr), unsafe_allow_html=True)

            if len(s) <= 5000:
                try:
                    _, p_norm = shapiro(s.sample(min(len(s), 5000), random_state=0))
                    norm_txt = f"✅ Normal (p={p_norm:.4f})" if p_norm > 0.05 else f"⚠ Not Normal (p={p_norm:.4f})"
                    st.caption(f"📐 Shapiro-Wilk normality test: {norm_txt}")
                except Exception:
                    pass

            st.markdown("")
            fig_dist = plot_numerical_column(df, col)
            st.pyplot(fig_dist); plt.close(fig_dist)

            st.markdown("**Variance Impact of Mean Imputation (simulated)**")
            vc = st.columns(3)
            delta_color = "#dc2626" if abs(vi)/max(vb,1e-9) > 0.3 else "#d97706" if abs(vi)/max(vb,1e-9) > 0.1 else "#16a34a"
            with vc[0]: st.markdown(stat_card("Variance (before)", f"{vb:.4g}"), unsafe_allow_html=True)
            with vc[1]: st.markdown(stat_card("Variance (after)",  f"{va:.4g}"), unsafe_allow_html=True)
            with vc[2]: st.markdown(stat_card("Δ Variance", f"{vi:.4g}", delta_color), unsafe_allow_html=True)

            pct_chg = abs(vi) / max(vb, 1e-9) * 100
            if pct_chg >= 30:
                st.warning(f"⚠ Variance drops by {pct_chg:.1f}% after mean imputation — over-smoothing risk. Use median or model-based imputation.")
            elif pct_chg >= 10:
                st.info(f"ℹ Variance drops by {pct_chg:.1f}% — acceptable, but monitor distribution shape.")
            else:
                st.success(f"✅ Variance change is small ({pct_chg:.1f}%) — mean imputation is statistically safe here.")

    else:
        s       = df[col].dropna()
        n_unique = s.nunique()
        mode_val = str(s.mode().iloc[0]) if len(s) > 0 else "N/A"
        mode_cnt = int((s == s.mode().iloc[0]).sum()) if len(s) > 0 else 0
        mode_pct = round(mode_cnt / max(len(s), 1) * 100, 1)

        r1 = st.columns(4)
        for (lbl, val), col_ui in zip(
            [("Unique Values", n_unique), ("Mode", mode_val[:12]),
             ("Mode Count", f"{mode_cnt:,}"), ("Mode Freq %", f"{mode_pct}%")], r1):
            with col_ui: st.markdown(stat_card(lbl, str(val)), unsafe_allow_html=True)

        st.markdown("")
        freq_table = s.value_counts().reset_index()
        freq_table.columns = ["Value", "Count"]
        freq_table["% of Present"] = (freq_table["Count"] / len(s) * 100).round(2)

        tab_chart, tab_table = st.tabs(["📊 Frequency Chart", "📋 Frequency Table"])
        with tab_chart:
            fig_cat = plot_categorical_column(df, col)
            st.pyplot(fig_cat); plt.close(fig_cat)
        with tab_table:
            st.dataframe(freq_table, use_container_width=True, hide_index=True)

    st.markdown("")

    if miss_count > 0:
        st.markdown("**How Missingness Relates to Other Features**")
        fig_pat = plot_missing_vs_features(df, col)
        if fig_pat:
            st.pyplot(fig_pat); plt.close(fig_pat)
            st.caption("Large differences between blue (present) and orange (missing) bars signal MAR behavior.")
        else:
            st.info("No other numerical features available for pattern comparison.")

    st.markdown("")

    verdict_cls = {"MCAR": "verdict-mcar", "MAR": "verdict-mar",
                   "MNAR": "verdict-mnar"}.get(mech, "verdict-clean")
    mech_icon  = {"MCAR": "🟢", "MAR": "🟡", "MNAR": "🔴"}.get(mech, "✅")
    mech_label = {"MCAR": "Missing Completely At Random (MCAR)",
                  "MAR":  "Missing At Random (MAR)",
                  "MNAR": "Missing Not At Random (MNAR)",
                  "N/A":  "No Missing Values"}.get(mech, mech)

    st.markdown(
        f'<div class="{verdict_cls}"><strong>{mech_icon} {mech_label}</strong><br>'
        f'<span style="font-size:0.9rem;color:#444;">{mech_reason}</span></div>',
        unsafe_allow_html=True)

    chips_html = strategy_chips_html(mech, miss_pct, col_type)
    if chips_html:
        st.markdown("")
        st.markdown("**Recommended Strategies**")
        st.markdown(chips_html, unsafe_allow_html=True)

    pointer = {
        "MCAR": ("📍 See **Step 2 → MCAR card** → **Step 3 (Threshold)** in the flowchart. "
                 "Missing% <5% → listwise deletion is safe. 5–20% → statistical imputation."),
        "MAR":  ("📍 See **Step 2 → MAR card** → **Step 4 → Advanced Imputation** in the flowchart. "
                 "KNN / MICE preferred. Create a missing indicator if missing% ≥10%."),
        "MNAR": ("📍 See **Step 2 → MNAR card** → **Step 4 → MNAR Strategy** in the flowchart. "
                 "**Create the missing indicator FIRST**, then use constant or sensitivity analysis."),
        "N/A":  "📍 No action needed — this column is complete. Proceed to feature engineering.",
    }.get(mech, "")
    if pointer:
        st.markdown("")
        st.info(pointer)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Missing Data Suite")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    st.markdown("---")
    st.markdown("**How to use**")
    st.markdown("""
1. Upload a CSV file
2. Select your target column
3. Explore each global analysis section
4. Use **Per-Column Analysis** to deep-dive any column
5. Review the final diagnosis table
    """)
    st.markdown("---")
    st.caption("Built with Streamlit · pandas · scikit-learn · scipy")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🔬 Missing Data Intelligence Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Comprehensive dataset diagnostics · missing mechanism detection · outlier & variance analysis</div>', unsafe_allow_html=True)

if uploaded_file is None:
    st.info("👈 Upload a CSV file from the sidebar to begin analysis.")
    st.stop()

# ── Load & preprocess ─────────────────────────────────────────────────────
df_raw = pd.read_csv(uploaded_file)
id_cols = [c for c in df_raw.columns if c.strip().lower() in ("id", "index", "row", "rowid", "row_id")]
if id_cols:
    df_raw.drop(columns=id_cols, inplace=True)
    st.toast(f"Auto-removed non-informative column(s): {id_cols}", icon="🗑️")

# ── Dataset Preview ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Dataset Preview</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
total_missing = df_raw.isnull().sum().sum()
pct_miss = round(total_missing / df_raw.size * 100, 1)
with c1: st.markdown(f'<div class="metric-card"><div class="val">{df_raw.shape[0]:,}</div><div class="lbl">Rows</div></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="metric-card"><div class="val">{df_raw.shape[1]:,}</div><div class="lbl">Columns</div></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="metric-card"><div class="val">{total_missing:,}</div><div class="lbl">Total Missing Cells</div></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="metric-card"><div class="val">{pct_miss}%</div><div class="lbl">Overall Missing Rate</div></div>', unsafe_allow_html=True)
st.dataframe(df_raw.head(10), use_container_width=True)

# ── Target selection ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">🎯 Target Column Selection</div>', unsafe_allow_html=True)
target_col = st.selectbox("Select target column (y):", df_raw.columns.tolist())
df = df_raw.copy()
X  = df.drop(columns=[target_col])
y  = df[target_col]
num_cols, cat_cols = identify_columns(X)
st.markdown(f"- **Target column:** `{target_col}`\n"
            f"- **Feature columns:** {X.shape[1]} total — {len(num_cols)} numerical, {len(cat_cols)} categorical")

# ── Missing Data Summary ──────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Missing Data Summary</div>', unsafe_allow_html=True)
miss_df = missing_summary(X, num_cols, cat_cols)
if miss_df.empty:
    st.success("✅ No missing values detected in features!")
else:
    def color_missing(val):
        if isinstance(val, float):
            if val >= 20: return "background-color: #f8d7da; color: #721c24;"
            if val >= 5:  return "background-color: #fff3cd; color: #856404;"
            if val > 0:   return "background-color: #d4edda; color: #155724;"
        return ""
    st.dataframe(miss_df.style.applymap(color_missing, subset=["Missing %"]),
                 use_container_width=True, hide_index=True)

# ── Missing Pattern Analysis ──────────────────────────────────────────────
st.markdown('<div class="section-header">🔍 Missing Pattern Analysis</div>', unsafe_allow_html=True)
tab_hm, tab_corr = st.tabs(["Missing Heatmap", "Missingness Correlation"])
with tab_hm:
    fig = plot_missing_heatmap(X)
    if fig: st.pyplot(fig); plt.close(fig)
    else:   st.info("No missing values to display.")
with tab_corr:
    fig2 = plot_missingness_correlation(X)
    if fig2:
        st.pyplot(fig2); plt.close(fig2)
        st.caption("Near +1: columns tend to be missing together. Near -1: rarely missing simultaneously.")
    else:
        st.info("Need at least 2 columns with missing values for this chart.")

# ── Correlation Analysis ──────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Correlation Analysis (Numerical Features)</div>', unsafe_allow_html=True)
fig3, corr_table = plot_correlation_heatmap(X, num_cols)
if fig3:
    st.pyplot(fig3); plt.close(fig3)
    st.markdown("**Strong Correlation Pairs (|r| > 0.5)**")
    if not corr_table.empty:
        st.dataframe(corr_table, use_container_width=True, hide_index=True)
    else:
        st.info("No strong pairs found.")
else:
    st.info("Not enough numerical columns or no strong correlations found.")

# ── Train-Test Split ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">✂️ Train-Test Split (80 / 20)</div>', unsafe_allow_html=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc1, sc2 = st.columns(2)
with sc1: st.markdown(f"**Training Set**\n- X_train: `{X_train.shape}`\n- y_train: `{y_train.shape}`")
with sc2: st.markdown(f"**Test Set**\n- X_test: `{X_test.shape}`\n- y_test: `{y_test.shape}`")

# ── Missing Mechanism Diagnosis ───────────────────────────────────────────
st.markdown('<div class="section-header">🧪 Missing Data Mechanism Diagnosis</div>', unsafe_allow_html=True)
missing_feature_cols = [c for c in X.columns if X[c].isnull().any()]

if not missing_feature_cols:
    st.success("No missing values in feature columns — nothing to diagnose.")
    mechanism_results = {}
else:
    with st.spinner("Running MCAR (Chi-square) and MAR (Logistic Regression) tests…"):
        mechanism_results = {}
        for col in missing_feature_cols:
            mech, reason = diagnose_mechanism(X, col, num_cols)
            mechanism_results[col] = {"mechanism": mech, "reason": reason}

    badge_map = {"MCAR": "badge-mcar", "MAR": "badge-mar", "MNAR": "badge-mnar"}
    for col, res in mechanism_results.items():
        mech = res["mechanism"]
        pct  = round(X[col].isnull().mean() * 100, 2)
        with st.expander(f"🔎 **{col}** — {mech}  |  {pct}% missing"):
            st.markdown(f'<span class="{badge_map[mech]}">{mech}</span>&nbsp;&nbsp;{res["reason"]}',
                        unsafe_allow_html=True)

# ── Outlier Detection & Variance Impact ──────────────────────────────────
st.markdown('<div class="section-header">⚡ Outlier Detection & Variance Impact</div>', unsafe_allow_html=True)
outlier_data = {}
for col in num_cols:
    n_out = detect_outliers_iqr(X[col])
    vb, va, vi = variance_impact(X[col])
    outlier_data[col] = {
        "Missing %": round(X[col].isnull().mean() * 100, 2),
        "Outliers (IQR)": n_out,
        "Variance (before impute)": vb,
        "Variance (after mean impute)": va,
        "Variance Impact (Δ)": vi,
    }
if outlier_data:
    out_df = (pd.DataFrame(outlier_data).T.reset_index()
              .rename(columns={"index": "Column"})
              .sort_values("Outliers (IQR)", ascending=False))

    def color_outliers(val):
        if isinstance(val, (int, float)):
            if val > 50: return "background-color: #f8d7da; color: #721c24;"
            if val > 10: return "background-color: #fff3cd; color: #856404;"
        return ""

    st.dataframe(out_df.style.applymap(color_outliers, subset=["Outliers (IQR)"]),
                 use_container_width=True, hide_index=True)
else:
    st.info("No numerical columns available for outlier analysis.")

# ── Final Diagnosis Table ─────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Final Diagnosis Table</div>', unsafe_allow_html=True)
diag_rows = []
for col in X.columns:
    mp   = round(X[col].isnull().mean() * 100, 2)
    mech = mechanism_results.get(col, {}).get("mechanism", "N/A") if col in missing_feature_cols else "N/A"
    diag_rows.append({
        "Column": col, "Missing %": mp,
        "Mechanism": mech, "Severity": severity(mp) if mp > 0 else "None",
        "Outliers": outlier_data.get(col, {}).get("Outliers (IQR)", "—"),
        "Variance Impact (Δ)": outlier_data.get(col, {}).get("Variance Impact (Δ)", "—"),
    })
diag_df = pd.DataFrame(diag_rows).sort_values("Missing %", ascending=False).reset_index(drop=True)

sev_colors  = {"High": ("background-color: #f8d7da;", "color: #721c24;"), 
               "Moderate": ("background-color: #fff3cd;", "color: #856404;"), 
               "Low": ("background-color: #d4edda;", "color: #155724;")}
mech_colors = {"MCAR": ("background-color: #d4edda;", "color: #155724;"), 
               "MAR":  ("background-color: #fff3cd;", "color: #856404;"), 
               "MNAR": ("background-color: #f8d7da;", "color: #721c24;")}

def color_diag_row(row):
    mech_style = " ".join(mech_colors.get(row['Mechanism'], ("", "")))
    sev_style  = " ".join(sev_colors.get(row['Severity'], ("", "")))
    return ["", "", mech_style, sev_style, "", ""]

st.dataframe(
    diag_df.style.apply(color_diag_row, axis=1),
    use_container_width=True,
    hide_index=True
)

# ── PER-COLUMN DEEP ANALYSIS ──────────────────────────────────────────────
st.markdown('<div class="section-header">🔬 Per-Column Deep Analysis</div>', unsafe_allow_html=True)
st.markdown("""
Select any column from the dropdown to run a full statistical analysis:
distribution charts, outlier breakdown, variance impact simulation,
missing mechanism verdict, and tailored strategy recommendations.
""")

col_label_to_name = {}
for col in X.columns:
    mp          = round(X[col].isnull().mean() * 100, 1)
    type_lbl    = "Num" if col in num_cols else "Cat"
    mech_lbl    = mechanism_results.get(col, {}).get("mechanism", "—") if col in missing_feature_cols else "complete"
    label       = f"{col}  [{type_lbl} · {mp}% missing · {mech_lbl}]"
    col_label_to_name[label] = col

chosen_label = st.selectbox(
    "Select a column to analyse in detail:",
    options=["— choose a column —"] + list(col_label_to_name.keys()),
    key="deep_col_select"
)

if chosen_label != "— choose a column —":
    chosen_col = col_label_to_name[chosen_label]
    with st.spinner(f"Analysing `{chosen_col}`…"):
        st.markdown("---")
        render_per_column_analysis(
            df=X,
            col=chosen_col,
            num_cols=num_cols,
            cat_cols=cat_cols,
            mechanism_results=mechanism_results,
        )
        st.markdown("---")


# ── Data Analysis Insights ────────────────────────────────────────────────
st.markdown('<div class="section-header">💡 Data Analysis Insights</div>', unsafe_allow_html=True)
high_miss = diag_df[diag_df["Missing %"] >= 20]["Column"].tolist()
mar_cols  = diag_df[diag_df["Mechanism"] == "MAR"]["Column"].tolist()
mnar_cols = diag_df[diag_df["Mechanism"] == "MNAR"]["Column"].tolist()
high_out  = [c for c in num_cols if outlier_data.get(c, {}).get("Outliers (IQR)", 0) > 10]

insights = [
    "Missing data must be understood <b>before</b> any imputation or modeling to avoid biased results.",
    (f"<b>{', '.join(high_miss)}</b> have ≥20% missing values — treat with caution or consider dropping."
     if high_miss else "No columns have critically high (≥20%) missing rates — dataset quality looks reasonable."),
    (f"Columns <b>{', '.join(mar_cols)}</b> show MAR behavior — KNN/MICE imputation is viable."
     if mar_cols else "No columns confirmed MAR."),
    (f"Columns <b>{', '.join(mnar_cols)}</b> are likely MNAR — create a missing indicator before imputing."
     if mnar_cols else "No columns flagged as MNAR."),
    (f"Columns <b>{', '.join(high_out)}</b> have many outliers — prefer median over mean imputation."
     if high_out else "Outlier counts appear manageable across numerical columns."),
    "Correlated missingness indicates data is likely <b>not MCAR</b> — jointly missing due to a common cause.",
    "MCAR is rare in real-world datasets. Most missingness in practice is MAR or MNAR.",
    "MNAR <b>cannot be confirmed statistically</b> from observed data alone — domain knowledge is essential.",
]
st.markdown('<div class="insight-box"><ul>' + "".join(f"<li>{i}</li>" for i in insights) + "</ul></div>",
            unsafe_allow_html=True)

# ── Theoretical Explanation ───────────────────────────────────────────────
st.markdown('<div class="section-header">📚 Theoretical Background</div>', unsafe_allow_html=True)
theories = [
    ("🔵 MCAR — Missing Completely At Random",
     "The probability of missingness is entirely independent of observed and unobserved data. "
     "Listwise deletion is unbiased under MCAR, though it reduces sample size."),
    ("🟡 MAR — Missing At Random",
     "Missingness depends on <i>observed</i> data but not on the missing value itself. "
     "Multiple imputation or FIML methods produce valid estimates under MAR."),
    ("🔴 MNAR — Missing Not At Random",
     "Missingness depends on the <i>unobserved value itself</i>. Cannot be detected from observed data. "
     "Requires sensitivity analysis and domain knowledge. Ignoring MNAR produces biased results."),
    ("📐 Why Chi-Square for MCAR Testing?",
     "Chi-square tests independence between the binary missingness indicator and binned numeric predictors. "
     "No significant association is consistent with MCAR, though this only confirms pairwise independence."),
    ("🤖 Why Logistic Regression for MAR Detection?",
     "LR models the binary missingness indicator as a function of all observed features. "
     "Accuracy substantially above the majority-class baseline indicates MAR."),
    ("📉 Why MNAR Cannot Be Confirmed Statistically",
     "MNAR depends on unobserved values — data we do not have. No statistical test on observed data "
     "can definitively confirm it. Domain reasoning about the data generation process is required."),
    ("📦 Outliers and Their Impact on Variance",
     "Outliers (>1.5×IQR) inflate variance and distort the mean. Mean imputation artificially collapses "
     "variance because all missing cells receive the same central value, masking true data spread."),
]
for title, body in theories:
    st.markdown(f'<div class="theory-box"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Missing Data Intelligence Suite · Built with Streamlit, pandas, scikit-learn, scipy, seaborn")