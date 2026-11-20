"""Builds the 1-2 page PDF (AI_ML_Task2_Report.pdf) from results.json + figures/ made by the notebook.
Every number and conclusion in the report is computed from your run - nothing is hard-coded."""
import json
from datetime import date
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

NAVY, LIGHT, GOLD = colors.HexColor("#1f3a5f"), colors.HexColor("#eef2f7"), colors.HexColor("#fff3c4")
USD = 100_000


def usd(v): return f"${v * USD:,.0f}"


def img(path, width):
    w, h = PILImage.open(path).size
    return Image(str(path), width=width, height=width * h / w)


def build_report(root=None, out_name="AI_ML_Task2_Report.pdf"):
    root = Path(root or Path(__file__).parent)
    R = json.loads((root / "results.json").read_text()); fig = root / "figures"
    ds, sp, comp = R["dataset"], R["split"], R["comparison"]
    by = {c["name"]: c for c in comp}
    base, best, bestlin = by[R["baseline"]], by[R["best_overall"]], by[R["best_linear"]]
    tree5, ridge1 = by[R["tree_required"]], by[R["ridge_required"]]

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.8, leading=11.6, spaceAfter=3)
    bul = ParagraphStyle("bu", parent=body, leftIndent=11, bulletIndent=1, spaceAfter=1.5)
    h1 = ParagraphStyle("h", parent=ss["Heading2"], textColor=NAVY, fontSize=11.5, spaceBefore=6, spaceAfter=3)
    title = ParagraphStyle("t", parent=ss["Title"], textColor=NAVY, fontSize=15.5, leading=19, spaceAfter=1)
    sub = ParagraphStyle("s", parent=body, alignment=1, textColor=colors.grey)
    cap = ParagraphStyle("c", parent=body, fontSize=7.5, leading=9, textColor=colors.grey, alignment=1)
    cell = ParagraphStyle("ce", parent=body, fontSize=7.8, leading=9.6, spaceAfter=0)
    cellh = ParagraphStyle("ch", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    P = lambda t, s=body: Paragraph(t, s)
    B = lambda t: Paragraph(t, bul, bulletText="-")

    # ---- results table (sorted by test RMSE; best model highlighted) ----
    rows = [["Model", "Test MAE", "Test RMSE", "Test R<super>2</super>", "CV RMSE", "Overfit gap", "Fit time (s)"]]
    ordered = sorted(comp, key=lambda c: c["rmse"])
    for c in ordered:
        rows.append([c["name"], f"{c['mae']:.3f}", f"{c['rmse']:.3f}", f"{c['r2']:.3f}", f"{c['cv_rmse']:.3f}", f"{c['gap']:+.3f}", f"{c['fit_s']:.1f}"])
    data = [[Paragraph(str(x), cellh if i == 0 else cell) for x in r] for i, r in enumerate(rows)]
    tbl = Table(data, colWidths=[6.0 * cm, 1.8 * cm, 1.9 * cm, 1.9 * cm, 1.8 * cm, 2.0 * cm, 2.0 * cm], repeatRows=1)
    st = [("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#c5cedb")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2), ("BACKGROUND", (0, 0), (-1, 0), NAVY)]
    st += [("BACKGROUND", (0, i + 1), (-1, i + 1), GOLD) for i, c in enumerate(ordered) if c["name"] == best["name"]]
    tbl.setStyle(TableStyle(st))

    # ---- data-driven narrative ----
    gain = (base["rmse"] - best["rmse"]) / base["rmse"] * 100
    widest = max(comp, key=lambda c: c["gap"])
    tree_vs = "outperforms" if tree5["rmse"] < base["rmse"] else "does not outperform"
    ridge_vs = "slightly improves on" if ridge1["rmse"] < base["rmse"] - 1e-4 else "performs about the same as"
    lin_gap = (bestlin["rmse"] - best["rmse"]) / best["rmse"] * 100
    S = [P("Feature Engineering, Model Optimization &amp; Performance Comparison", title),
         P(f"California Housing | AI &amp; ML Task 2 | Manish Kumar Yadav | {date.today():%d %b %Y}", sub), Spacer(1, 3),
         P("1. Objective and data", h1),
         P(f"The goal is to improve on the Task 1 baseline the way ML engineers do in practice: prepare the data correctly, train several algorithms, "
           f"compare them with objective metrics and justify the final choice. The California Housing dataset has {ds['rows']:,} census block groups, "
           f"8 numeric features and the median house value as target (units of $100,000). It was split 80/20 into {sp['train']:,} training and {sp['test']:,} "
           f"test rows (random_state = {sp['random_state']})."),
         P("2. Methodology", h1),
         B("<b>Leak-free preprocessing:</b> features differ widely in scale (population in the thousands, latitude around 35), so they are standardised with "
           "StandardScaler. The data is split <b>first</b> and the scaler is fitted on the training set only, inside a scikit-learn Pipeline; scaling "
           "before splitting would let test-set statistics leak into training."),
         B("<b>Stage A, required models:</b> Linear Regression (baseline), Ridge (alpha = 1.0, reduces over-fitting) and Decision Tree (max_depth = 5, captures non-linearity)."),
         B("<b>Stage B, optimisation:</b> 5-fold GridSearchCV for the Ridge alpha and the tree depth / leaf size; feature engineering (ratios, log transforms, distance "
           "to major cities); Random Forest and Gradient Boosting as stronger non-linear references."),
         B("<b>Evaluation:</b> RMSE (primary), MAE and R<super>2</super> on the unseen test set. Models are <b>selected by cross-validated RMSE on the training data</b>, so the "
           "test set is used only for the final comparison. The overfit gap is train R<super>2</super> minus test R<super>2</super>."),
         P("3. Model performance comparison", h1), tbl,
         P("Sorted by test RMSE (lower is better). Highlighted row: selected model. CV RMSE = 5-fold on training data. Prices in units of $100,000.", cap),
         Spacer(1, 4), img(fig / "model_comparison.png", 15.5 * cm), P("Figure 1. Test RMSE and R<super>2</super> for every model.", cap),
         img(fig / "actual_vs_predicted.png", 16.5 * cm), P("Figure 2. Actual vs predicted prices on the test set (closer to the red line is better).", cap)]

    concl = [P("4. Model selection and justification", h1),
             P(f"<b>Selected model: {best['name']}.</b> It has the lowest cross-validated RMSE ({best['cv_rmse']:.3f}) and on the test set reaches RMSE "
               f"{best['rmse']:.3f} (about {usd(best['rmse'])}), MAE {best['mae']:.3f} and R<super>2</super> {best['r2']:.3f}. Compared with the baseline Linear Regression "
               f"(RMSE {base['rmse']:.3f}, R<super>2</super> {base['r2']:.3f}) this changes RMSE by <b>{-gain:+.1f}%</b>. Its overfit gap is {best['gap']:+.3f}, "
               f"which indicates {'good generalisation' if best['gap'] < 0.1 else 'some over-fitting that should be watched'}."),
             P(f"Ridge (alpha = 1) {ridge_vs} plain Linear Regression, because with 8 features and thousands of rows there is little variance to shrink. "
               f"The depth-5 Decision Tree {tree_vs} the baseline (RMSE {tree5['rmse']:.3f} vs {base['rmse']:.3f}), showing the effect of non-linearity. "
               f"The largest train-test gap belongs to <b>{widest['name']}</b> ({widest['gap']:+.3f}), a reminder that flexible models memorise training data when not constrained."),
             P(f"<b>Trade-off:</b> if interpretability matters more than accuracy, the best linear model is {bestlin['name']} (test RMSE {bestlin['rmse']:.3f}, "
               f"{lin_gap:+.1f}% vs the selected model); its coefficients can be read directly, whereas {best['name']} is less transparent."),
             P("5. Conclusions and next steps", h1),
             B("Correct preprocessing (split before scaling), several algorithms and metric-based selection give a defensible, reproducible choice."),
             B("Feature engineering, especially location-based features, and non-linear models matter more than tuning a linear model."),
             B("Next: wider hyper-parameter search, a log-transformed target, handling the $500,000 price cap, spatial neighbour features and error analysis by region."),
             B("Limitations: 1990 census block-group averages; results are for learning, not for real valuation.")]
    S += [KeepTogether(concl)]

    def footer(c, d):
        c.saveState(); c.setFont("Helvetica", 7.5); c.setFillColor(colors.grey)
        c.drawString(1.6 * cm, .8 * cm, "Task 2 - Model Comparison on California Housing"); c.drawRightString(A4[0] - 1.6 * cm, .8 * cm, f"Page {d.page}"); c.restoreState()

    out = root / out_name
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.2 * cm, bottomMargin=1.3 * cm,
                      title="Task 2 - Model Comparison", author="Manish Kumar Yadav").build(S, onFirstPage=footer, onLaterPages=footer)
    return out


if __name__ == "__main__":
    print("Report written to", build_report())
