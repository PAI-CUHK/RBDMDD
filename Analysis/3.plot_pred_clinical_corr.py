# ============================================================
# RBD-MDD Correlation + Figure Script (ONE FILE)
# 1) Correlation tables vs TARGET ("Mean_RBD_prob")
#    - Primary: Spearman (robust)
#    - Sensitivity: Pearson
#    - Pairwise complete observations
#    - FDR (Benjamini–Hochberg) within each family
#    - Outputs (CSV + one Excel workbook):
#        Table1: Scale-level
#        Table2: HADS/BDI items (+ totals) + Top10
#        Table3: UPDRS items (+ totals/HY) + Top10
#        Table4: Model metrics
# 2) 1×3 combined scatter figure (UPDRS, BDI, HADS)
#    - Colored by group (MDD+RBD vs MDD-Only)
#    - Overall linear fit (red) + 95% CI band
#    - Pearson r & p in top-left
#    - BDI y-lim: -2..25 with ticks starting at 0
# ============================================================

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, pearsonr, t
from statsmodels.stats.multitest import multipletests
from PIL import Image


# -----------------------------
# Config
# -----------------------------
DATA_PATH = "RBDMDD_data_clean.xlsx"     # input xlsx
TARGET = "Mean_RBD_prob"                 # model score column

# outputs
CORR_OUT_DIR = "rbdmdd_corr_results"
FIG_OUT_DIR  = "rbdmdd_pretty_1x3"

os.makedirs(CORR_OUT_DIR, exist_ok=True)
os.makedirs(FIG_OUT_DIR, exist_ok=True)


# -----------------------------
# Helpers: Correlations
# -----------------------------
def _pairwise_series(df: pd.DataFrame, x: str, y: str) -> tuple[pd.Series, pd.Series]:
    sub = df[[x, y]].dropna()
    return sub[x], sub[y]

def _has_variance(s: pd.Series) -> bool:
    return s.nunique(dropna=True) >= 2

def corr_pair(df: pd.DataFrame, x: str, y: str) -> dict:
    xs, ys = _pairwise_series(df, x, y)
    n = len(xs)

    out = {
        "x": x, "y": y, "n": n,
        "spearman_rho": np.nan, "spearman_p": np.nan,
        "pearson_r": np.nan, "pearson_p": np.nan,
        "note": ""
    }

    if n < 3:
        out["note"] = "Too few paired observations"
        return out
    if (not _has_variance(xs)) or (not _has_variance(ys)):
        out["note"] = "No variance in x or y"
        return out

    rho, p_s = spearmanr(xs, ys)
    r, p_p = pearsonr(xs, ys)
    out.update({"spearman_rho": rho, "spearman_p": p_s, "pearson_r": r, "pearson_p": p_p})
    return out

def corr_many(df: pd.DataFrame, target: str, variables: list[str]) -> pd.DataFrame:
    rows = []
    for v in variables:
        if v == target:
            continue
        if v not in df.columns:
            rows.append({
                "x": target, "y": v, "n": 0,
                "spearman_rho": np.nan, "spearman_p": np.nan,
                "pearson_r": np.nan, "pearson_p": np.nan,
                "note": "Column not found"
            })
            continue
        rows.append(corr_pair(df, target, v))
    return pd.DataFrame(rows)

def add_fdr(df_corr: pd.DataFrame, p_col: str, out_col: str, sig_col: str, alpha: float = 0.05) -> pd.DataFrame:
    df2 = df_corr.copy()
    mask = np.isfinite(df2[p_col].values)
    df2[out_col] = np.nan
    df2[sig_col] = False
    if mask.sum() == 0:
        return df2
    rej, p_adj, *_ = multipletests(df2.loc[mask, p_col].values, alpha=alpha, method="fdr_bh")
    df2.loc[mask, out_col] = p_adj
    df2.loc[mask, sig_col] = rej
    return df2

def top_k(df_corr: pd.DataFrame, k: int = 10, sort_col: str = "spearman_p") -> pd.DataFrame:
    out = df_corr.copy()
    out = out[np.isfinite(out[sort_col])]
    return out.sort_values(sort_col, ascending=True).head(k)

def _pretty(df_corr: pd.DataFrame) -> pd.DataFrame:
    out = df_corr.rename(columns={"y": "variable"}).copy()
    cols_order = [
        "variable", "n",
        "spearman_rho", "spearman_p", "spearman_p_fdr", "spearman_sig_fdr",
        "pearson_r", "pearson_p",
        "note"
    ]
    for c in cols_order:
        if c not in out.columns:
            out[c] = np.nan if c not in ("note", "variable") else ""
    return out[cols_order]


# -----------------------------
# Helpers: Plotting (fit + 95% CI)
# -----------------------------
def add_linear_fit_with_ci(ax, x, y, color="red", alpha_fill=0.18, lw=2.2):
    """
    Add linear regression line and 95% CI band for mean prediction.
    """
    n = len(x)
    if n < 3 or len(np.unique(x)) <= 1:
        return

    a, b = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 200)
    yhat = a * xx + b

    y_pred = a * x + b
    resid = y - y_pred
    dof = n - 2
    if dof <= 0:
        ax.plot(xx, yhat, color=color, linewidth=lw)
        return

    s_err = np.sqrt(np.sum(resid**2) / dof)
    x_mean = np.mean(x)
    Sxx = np.sum((x - x_mean) ** 2)
    if Sxx <= 0:
        ax.plot(xx, yhat, color=color, linewidth=lw)
        return

    tcrit = t.ppf(0.975, dof)
    se_fit = s_err * np.sqrt(1 / n + (xx - x_mean) ** 2 / Sxx)
    ci_lower = yhat - tcrit * se_fit
    ci_upper = yhat + tcrit * se_fit

    ax.fill_between(xx, ci_lower, ci_upper, alpha=alpha_fill)
    ax.plot(xx, yhat, color=color, linewidth=lw)


# -----------------------------
# Load data
# -----------------------------
df = pd.read_excel(DATA_PATH)

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found. Columns: {list(df.columns)}")


# ============================================================
# PART A: Correlation Tables
# ============================================================
# Families
scale_vars = ["HADS_Dep", "HADS_Anx", "HADS_TOT", "BDI_TotalScore", "UPDRS_TOT", "HY"]

hads_items = [c for c in df.columns if re.fullmatch(r"HADS_\d+", str(c))]
bdi_items  = [c for c in df.columns if re.fullmatch(r"BDI_\d+", str(c))]
hads_bdi_all = hads_items + ["HADS_Dep", "HADS_Anx", "HADS_TOT"] + bdi_items + ["BDI_TotalScore"]
hads_bdi_all = [c for c in hads_bdi_all if c in df.columns]

updrs_items = [c for c in df.columns if str(c).startswith("re_updrs")]
updrs_all = updrs_items + ["UPDRS_TOT", "HY"]
updrs_all = [c for c in updrs_all if c in df.columns]

model_metrics = ["SD", "Agreement_rate", "Norm_entropy"]
model_metrics = [c for c in model_metrics if c in df.columns]

# Compute tables
tbl_scale = add_fdr(
    corr_many(df, TARGET, scale_vars),
    p_col="spearman_p", out_col="spearman_p_fdr", sig_col="spearman_sig_fdr", alpha=0.05
)

tbl_hads_bdi = add_fdr(
    corr_many(df, TARGET, hads_bdi_all),
    p_col="spearman_p", out_col="spearman_p_fdr", sig_col="spearman_sig_fdr", alpha=0.05
)
tbl_hads_bdi_top10 = top_k(tbl_hads_bdi, k=10, sort_col="spearman_p")

tbl_updrs = add_fdr(
    corr_many(df, TARGET, updrs_all),
    p_col="spearman_p", out_col="spearman_p_fdr", sig_col="spearman_sig_fdr", alpha=0.05
)
tbl_updrs_top10 = top_k(tbl_updrs, k=10, sort_col="spearman_p")

tbl_model = add_fdr(
    corr_many(df, TARGET, model_metrics),
    p_col="spearman_p", out_col="spearman_p_fdr", sig_col="spearman_sig_fdr", alpha=0.05
)

# Pretty
tbl_scale_pretty = _pretty(tbl_scale)
tbl_hads_bdi_pretty = _pretty(tbl_hads_bdi)
tbl_hads_bdi_top10_pretty = _pretty(tbl_hads_bdi_top10)
tbl_updrs_pretty = _pretty(tbl_updrs)
tbl_updrs_top10_pretty = _pretty(tbl_updrs_top10)
tbl_model_pretty = _pretty(tbl_model)

# Save CSVs
tbl_scale_pretty.to_csv(os.path.join(CORR_OUT_DIR, "table1_scale_level.csv"), index=False)
tbl_hads_bdi_pretty.to_csv(os.path.join(CORR_OUT_DIR, "table2_hads_bdi_all.csv"), index=False)
tbl_hads_bdi_top10_pretty.to_csv(os.path.join(CORR_OUT_DIR, "table2_hads_bdi_top10.csv"), index=False)
tbl_updrs_pretty.to_csv(os.path.join(CORR_OUT_DIR, "table3_updrs_all.csv"), index=False)
tbl_updrs_top10_pretty.to_csv(os.path.join(CORR_OUT_DIR, "table3_updrs_top10.csv"), index=False)
tbl_model_pretty.to_csv(os.path.join(CORR_OUT_DIR, "table4_model_metrics.csv"), index=False)

# Save one Excel workbook
xlsx_path = os.path.join(CORR_OUT_DIR, "all_correlation_tables.xlsx")
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    tbl_scale_pretty.to_excel(writer, sheet_name="Table1_ScaleLevel", index=False)
    tbl_hads_bdi_pretty.to_excel(writer, sheet_name="Table2_HADS_BDI_All", index=False)
    tbl_hads_bdi_top10_pretty.to_excel(writer, sheet_name="Table2_HADS_BDI_Top10", index=False)
    tbl_updrs_pretty.to_excel(writer, sheet_name="Table3_UPDRS_All", index=False)
    tbl_updrs_top10_pretty.to_excel(writer, sheet_name="Table3_UPDRS_Top10", index=False)
    tbl_model_pretty.to_excel(writer, sheet_name="Table4_ModelMetrics", index=False)

print(f"[OK] Correlation tables saved to: {CORR_OUT_DIR}")
print(f"[OK] Excel workbook: {xlsx_path}")


# ============================================================
# PART B: 1×3 Figure (UPDRS, BDI, HADS)
# ============================================================
group_map = {"RBD": "MDD+RBD", "Control": "MDD-Only"}
df["Group_label"] = df["Group"].map(group_map).fillna(df["Group"].astype(str))

XCOL = TARGET
X_LABEL = "Mean predicted probability of comorbid RBD in MDD"

panels = [
    ("UPDRS_TOT", "UPDRS", "Total UPDRS Score"),
    ("BDI_TotalScore", "BDI", "Total BDI Score"),
    ("HADS_TOT", "HADS", "Total HADS Score"),
]

def make_single_plot(ycol, title_short, y_label):
    sub = df[[XCOL, ycol, "Group_label"]].dropna()
    x = sub[XCOL].to_numpy()
    y = sub[ycol].to_numpy()

    if len(sub) >= 3 and len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
        r, p = pearsonr(x, y)
    else:
        r, p = np.nan, np.nan

    fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=240)

    for glab in ["MDD-Only", "MDD+RBD"]:
        ss = sub[sub["Group_label"] == glab]
        if len(ss) == 0:
            continue
        ax.scatter(
            ss[XCOL], ss[ycol],
            s=52, alpha=0.86,
            edgecolors="white", linewidths=0.6,
            label=glab
        )

    add_linear_fit_with_ci(ax, x, y, color="red", alpha_fill=0.18, lw=2.2)

    if np.isfinite(r) and np.isfinite(p):
        txt = f"Pearson r = {r:.2f}\np = {p:.3g}"
    else:
        txt = "Pearson r = NA\np = NA"

    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="black", alpha=1.0)
    )

    ax.set_title(title_short, fontsize=14, pad=10)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, fontsize=9, loc="lower left")

    if title_short == "BDI":
        ax.set_ylim(-2, 25)
        ax.set_yticks([0, 5, 10, 15, 20, 25])

    fig.tight_layout()
    out_path = os.path.join(FIG_OUT_DIR, f"panel_{title_short}.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path

panel_paths = [make_single_plot(*p) for p in panels]

# Combine to 1×3
imgs = [Image.open(p).convert("RGB") for p in panel_paths]
w = max(im.size[0] for im in imgs)
h = max(im.size[1] for im in imgs)

padded = []
for im in imgs:
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    xoff = (w - im.size[0]) // 2
    yoff = (h - im.size[1]) // 2
    canvas.paste(im, (xoff, yoff))
    padded.append(canvas)

combined = Image.new("RGB", (w * 3, h), (255, 255, 255))
combined.paste(padded[0], (0, 0))
combined.paste(padded[1], (w, 0))
combined.paste(padded[2], (w * 2, 0))

combined_path = os.path.join(FIG_OUT_DIR, "MeanRBDprob_correlations_1x3.png")
combined.save(combined_path, quality=95)

print(f"[OK] Figure saved to: {combined_path}")
