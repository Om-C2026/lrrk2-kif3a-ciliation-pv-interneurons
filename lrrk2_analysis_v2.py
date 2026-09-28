import sys
import numpy as np
import scanpy as sc
import pandas as pd
import os
import warnings
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

OUTDIR = "results_revised"
os.makedirs(OUTDIR, exist_ok=True)

path = sys.argv[1]
print("Loading " + path)
adata = sc.read_h5ad(path)
print(str(adata.n_obs) + " cells x " + str(adata.n_vars) + " genes")

# Check if data needs log normalization
max_val = float(adata.X.max())
print("Max expression value: " + str(round(max_val, 2)))
if max_val > 20:
    print("Data appears to be raw counts. Normalizing and log-transforming...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print("Max after log1p: " + str(round(float(adata.X.max()), 2)))
else:
    print("Data appears already log-normalized.")

# Per-cell-type median split (not global!)
print("")
print("DERIVING CILIA_BIN PER CELL TYPE (within-type median split)")
adata.obs["cilia_bin"] = "NA"
for cn in adata.obs["celltype"].unique():
    m = adata.obs["celltype"] == cn
    n = int(m.sum())
    if n < 6:
        continue
    scores = adata.obs.loc[m, "score_ciliation"].values.astype(float)
    med = float(np.median(scores))
    hi = scores >= med
    adata.obs.loc[m, "cilia_bin"] = np.where(hi, "Hi", "Lo")
    n_hi = int(hi.sum())
    n_lo = n - n_hi
    print("  " + cn.ljust(25) + " n=" + str(n).rjust(6)
          + " median=" + "{:.4f}".format(med)
          + " Hi=" + str(n_hi) + " Lo=" + str(n_lo))

# Sex confound check (global)
print("")
print("=" * 60)
print("SEX CONFOUND CHECK (global)")
print("=" * 60)
valid = adata.obs["cilia_bin"] != "NA"
ct = pd.crosstab(adata.obs.loc[valid, "cilia_bin"], adata.obs.loc[valid, "sex"])
print(ct)
chi2, pv, dof, exp = stats.chi2_contingency(ct)
print("Chi2=" + str(round(chi2, 1)) + " p=" + str(pv))
if pv < 0.05:
    print("*** GLOBAL SEX CONFOUND PRESENT ***")

# Per cell type sex check
print("")
print("Sex confound per cell type:")
for cn in ["Neuron", "PV_Interneuron"]:
    m = adata.obs["celltype"] == cn
    ct2 = pd.crosstab(adata.obs.loc[m, "cilia_bin"], adata.obs.loc[m, "sex"])
    print("")
    print(cn + ":")
    print(ct2)
    if ct2.shape[0] > 1 and ct2.shape[1] > 1:
        chi2b, pvb, _, _ = stats.chi2_contingency(ct2)
        print("Chi2=" + str(round(chi2b, 1)) + " p=" + "{:.2e}".format(pvb))
        if pvb < 0.05:
            print("*** CONFOUNDED ***")
        else:
            print("Not confounded within this cell type")

# Remove Y-chromosome genes
Y_GENES = [
    "USP9Y", "NLGN4Y", "UTY", "DDX3Y", "KDM5D",
    "EIF1AY", "RPS4Y1", "ZFY", "TMSB4Y", "PCDH11Y",
    "SRY", "AMELY", "TBL1Y",
]
yp = [g for g in Y_GENES if g in adata.var_names]
print("")
print("Removing " + str(len(yp)) + " Y-chrom genes")
af = adata[:, ~adata.var_names.isin(yp)].copy()

# Corrected DE per powered cell type
print("")
print("=" * 60)
print("CORRECTED DE (per-cell-type split, Y-chrom removed, log-normalized)")
print("=" * 60)

for cn in ["Neuron", "PV_Interneuron"]:
    m = af.obs["celltype"] == cn
    sub = af[m].copy()
    groups = sub.obs["cilia_bin"].value_counts()
    print("")
    print(cn + ": n=" + str(sub.n_obs))
    print("  Groups: " + str(groups.to_dict()))

    min_group = int(groups.min())
    if min_group < 10:
        print("  SKIPPING: smallest group has " + str(min_group) + " cells")
        continue

    sc.tl.rank_genes_groups(
        sub,
        groupby="cilia_bin",
        method="wilcoxon",
        corr_method="benjamini-hochberg",
    )
    result = sc.get.rank_genes_groups_df(sub, group=None)

    # Filter to just the Hi group comparison
    if "group" in result.columns:
        result = result[result["group"] == "Hi"]

    n_sig = int((result["pvals_adj"] < 0.05).sum())
    print("  Adj-significant genes (FDR < 0.05): " + str(n_sig))

    print("  Top 15 genes:")
    for _, r in result.head(15).iterrows():
        mk = "***" if r["pvals_adj"] < 0.05 else "   "
        nm = str(r["names"]).ljust(15)
        pa = "{:.2e}".format(r["pvals_adj"])
        lf = "{:+.3f}".format(r["logfoldchanges"])
        print("  " + mk + " " + nm + " padj=" + pa + " lfc=" + lf)

    outf = os.path.join(OUTDIR, "DE_" + cn + "_corrected.csv")
    result.to_csv(outf, index=False)
    print("  Saved: " + outf)

    # Volcano plot
    fig, ax = plt.subplots(figsize=(10, 8))
    nlp = -np.log10(result["pvals_adj"].clip(lower=1e-50))
    lfc = result["logfoldchanges"]
    padj = result["pvals_adj"]

    not_sig = padj >= 0.05
    sig_up = (padj < 0.05) & (lfc > 0)
    sig_dn = (padj < 0.05) & (lfc < 0)

    ax.scatter(lfc[not_sig], nlp[not_sig], c="#CCCCCC", s=8, alpha=0.4, label="NS")
    if sig_up.sum() > 0:
        ax.scatter(lfc[sig_up], nlp[sig_up], c="#E63946", s=20, alpha=0.7, label="Up")
    if sig_dn.sum() > 0:
        ax.scatter(lfc[sig_dn], nlp[sig_dn], c="#457B9D", s=20, alpha=0.7, label="Down")

    for _, r2 in result.nsmallest(5, "pvals_adj").iterrows():
        if r2["pvals_adj"] < 0.1:
            ax.annotate(
                r2["names"],
                (r2["logfoldchanges"], -np.log10(max(r2["pvals_adj"], 1e-50))),
                fontsize=8, ha="center",
            )

    ax.axhline(-np.log10(0.05), ls="--", color="gray", lw=0.8)
    ax.set_xlabel("log2 Fold Change", fontsize=12)
    ax.set_ylabel("-log10(adjusted p-value)", fontsize=12)
    ax.set_title(cn + " - Corrected DE\n(per-cell-type split, Y-chrom removed)", fontsize=13)
    ax.legend(fontsize=9)

    vf = os.path.join(OUTDIR, "volcano_" + cn + "_corrected.png")
    fig.savefig(vf, dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved: " + vf)

# Cilia-Shh correlations
print("")
print("=" * 60)
print("CILIA-SHH CORRELATIONS BY CELL TYPE")
print("=" * 60)
for cn in sorted(adata.obs["celltype"].unique()):
    m = adata.obs["celltype"] == cn
    n = int(m.sum())
    if n < 10:
        continue
    c = adata.obs.loc[m, "score_ciliation"].values.astype(float)
    s = adata.obs.loc[m, "score_shh"].values.astype(float)
    if np.std(c) > 0 and np.std(s) > 0:
        r, p = stats.pearsonr(c, s)
        print(
            "  " + cn.ljust(25)
            + " n=" + str(n).rjust(6)
            + " r=" + "{:+.4f}".format(r)
            + " p=" + "{:.2e}".format(p)
        )

print("")
print("DONE - check results_revised folder")
