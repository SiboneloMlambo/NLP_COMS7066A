from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
)

ROOT = Path(__file__).parent
OUT = ROOT / "output" / "pdf"
OUT.mkdir(parents=True, exist_ok=True)
PDF = OUT / "Extended_Abstract_Model_Collapse.pdf"
SUMMARY = pd.read_csv(ROOT / "outputs" / "submission" / "summary.csv")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="PaperTitle", parent=styles["Title"], fontName="Helvetica-Bold",
                          fontSize=15, leading=17, alignment=TA_CENTER, spaceAfter=4))
styles.add(ParagraphStyle(name="Authors", parent=styles["Normal"], fontSize=9, leading=11,
                          alignment=TA_CENTER, spaceAfter=7))
styles.add(ParagraphStyle(name="AbstractBody", parent=styles["BodyText"], fontSize=8.2, leading=10,
                          alignment=TA_JUSTIFY, leftIndent=7*mm, rightIndent=7*mm, spaceAfter=5))
styles.add(ParagraphStyle(name="H1paper", parent=styles["Heading1"], fontName="Helvetica-Bold",
                          fontSize=10, leading=11, spaceBefore=5, spaceAfter=2, keepWithNext=True))
styles.add(ParagraphStyle(name="BodyPaper", parent=styles["BodyText"], fontSize=8.4, leading=10.2,
                          alignment=TA_JUSTIFY, spaceAfter=3))
styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontSize=7.2, leading=8.5,
                          alignment=TA_JUSTIFY, spaceBefore=2, spaceAfter=4))
styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=7.6, leading=9.2,
                          alignment=TA_JUSTIFY, spaceAfter=3))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#B7C9D6"))
    canvas.line(18*mm, 13*mm, 192*mm, 13*mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#455A64"))
    canvas.drawString(18*mm, 8.5*mm, "COMS4054A / COMS7062A / COMS7066A - NLP Project 1")
    canvas.drawRightString(192*mm, 8.5*mm, str(doc.page))
    canvas.restoreState()


def section(title, body):
    return [Paragraph(title, styles["H1paper"]), Paragraph(body, styles["BodyPaper"])]


def metric(task, generation, field):
    row = SUMMARY[(SUMMARY.task == task) & (SUMMARY.generation == generation)].iloc[0]
    return row[field]


story = []
story += [
    Paragraph("Interpretability of Model Collapse in a Synthetic In-Context Learning Task", styles["PaperTitle"]),
    Paragraph("Sibonelo Mlambo<br/>University of the Witwatersrand - Natural Language Processing", styles["Authors"]),
    Paragraph("<b>Abstract.</b> We study recursive synthetic-label training in a two-layer causal Transformer performing an arbitrary symbol-value lookup task. A base task predicts one value; an extended task predicts the value, a subsequent symbol, and its mapped value. Thirty-six independently initialised models span two tasks, six generations, and three seeds. Clean-test first-value accuracy declined from 32.0% to 27.3% in the base condition and from 32.2% to 15.8% in the extended condition. The best induction-pattern attention score was stable in the base task (0.244 to 0.234) but declined in the extended task (0.259 to 0.175). These results support faster behavioural and mechanistic degradation under the extended recursive objective, while low exact-sequence accuracy and the use of fixed real contexts limit the strength and generality of the conclusion.", styles["AbstractBody"]),
]

story += section("1. Introduction", "In-context learning (ICL) allows a Transformer to infer a temporary mapping from examples in its prompt without parameter updates. Induction heads provide a proposed circuit-level account: information is moved from a preceding token and later retrieved when a matching token recurs [1,2]. Model collapse is the progressive distortion produced when synthetic outputs are recursively reused as training data [3,4]. We ask: <i>does recursive synthetic-label training degrade behavioural ICL and its associated attention patterns, and is degradation stronger when the output sequence is longer?</i>")

story += section("2. Method", "Each example sampled a fresh permutation between eight symbol tokens and eight value tokens. Its context contained six sampled symbol-value pairs and ended with a query symbol that occurred earlier. The base target contained the query's value. The extended target contained this value, a sampled next symbol, and that symbol's mapped value. We used 3,000/600/1,000 independently generated train/validation/test examples. Validation and test examples remained clean at every generation. Generation 0 learned real targets; each newly initialised model in Generations 1-5 learned targets sampled at temperature 1 from its predecessor. Thus the intervention isolates recursive label corruption while holding contexts fixed.")

eq = "L = -(1/N) &Sigma;<sub>i,t</sub> log p<sub>&theta;</sub>(y<sub>i,t</sub> | x<sub>i</sub>, y<sub>i,&lt;t</sub>)"
story.append(Paragraph(f"The model had two causal attention blocks, four heads, width 64, feed-forward width 128, learned positional embeddings and tied input-output embeddings. AdamW minimised {eq}. Early stopping used validation loss (maximum 25 epochs, patience 5). We repeated the complete pipeline with seeds 42, 123 and 2026. Metrics were clean-test cross-entropy, first-value and exact accuracy, generated-symbol entropy, and attention-pattern scores. The previous-token score averages attention to position <i>t-1</i>. The induction score measures attention from the final query to value positions immediately following earlier matching symbols. These scores identify candidates rather than demonstrate causality.", styles["BodyPaper"]))

chart = ROOT / "outputs" / "submission" / "main_results.png"
story += [Image(str(chart), width=158*mm, height=96*mm),
          Paragraph("<b>Figure 1.</b> Mean performance and attention-pattern scores across three seeds; bands show one standard deviation. Extended exact accuracy is near zero from Generation 0 and must be interpreted as a task-learning limitation.", styles["Caption"]),
          PageBreak()]

story += section("3. Results", "Table 1 summarises the endpoints. Base first-value accuracy decreased by 4.7 percentage points (14.6% relative), whereas extended first-value accuracy decreased by 16.4 points (50.9% relative). Loss increased from 1.632 to 1.955 for the base condition and from 1.975 to 2.166 for the extended condition. The extended induction-pattern score decreased by 32.4%; the base score changed by only 4.1%. Extended symbol entropy varied substantially across seeds and did not decrease monotonically, so the experiment does not support a simple claim of steadily shrinking output diversity.")

table_data = [["Task", "Metric", "Gen. 0 mean +/- SD", "Gen. 5 mean +/- SD"]]
for task, label in [("base", "Base"), ("extended", "Extended")]:
    for field, name in [("first_value_accuracy", "First-value acc."),
                        ("loss", "Test loss"),
                        ("induction_score", "Induction score")]:
        a = metric(task, 0, field + "_mean")
        a_sd = metric(task, 0, field + "_sd")
        b = metric(task, 5, field + "_mean")
        b_sd = metric(task, 5, field + "_sd")
        table_data.append([label, name, f"{a:.3f} +/- {a_sd:.3f}", f"{b:.3f} +/- {b_sd:.3f}"])
tbl = Table(table_data, colWidths=[23*mm, 43*mm, 48*mm, 48*mm], repeatRows=1)
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#123B5D")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
    ("FONTSIZE", (0,0), (-1,-1), 7.5),
    ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#9FB3C1")),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#EDF3F7")]),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("ALIGN", (2,1), (-1,-1), "CENTER"),
    ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
]))
story += [KeepTogether([tbl, Paragraph("<b>Table 1.</b> Clean-test endpoint results across three random seeds.", styles["Caption"])])]

story += section("4. Discussion", "The behavioural evidence is consistent with recursive error accumulation, with the longer output objective degrading more quickly. The mechanistic result is more specific: the strongest attention pattern compatible with induction weakened in the extended model while remaining broadly stable in the base model. This association suggests that recursive targets can disrupt the internal retrieval strategy, but it does not establish that the scored head causes the behavioural decline. Head ablation or activation patching would be required for a causal claim [2]. The extended model's extremely low exact accuracy at Generation 0 means it never robustly learned all three tokens; consequently, its later exact-accuracy trajectory cannot diagnose collapse. First-value accuracy remains comparable between tasks at Generation 0 and is therefore the more defensible shared measure.")

story += section("5. Limitations and conclusion", "This controlled experiment studies recursive labels rather than full recursive generation of both contexts and targets. The artificial vocabulary, small model, three seeds and five recursive steps restrict external validity. Symbol entropy was unstable, and selecting the maximum head score may be sensitive to head swapping across generations. Within these limits, the results show modest base-task degradation and substantially stronger extended-task degradation, accompanied by weakening induction-like attention only in the extended condition. Future work should increase Generation 0 exact accuracy, pre-register a head-matching procedure, include mixed real/synthetic ratios, and perform causal interventions.")

story += [Spacer(1, 4), Paragraph("Reproducibility", styles["H1paper"]),
          Paragraph("All configurations, seeds, data generation, checkpoints, raw metrics and plotting code accompany the submission. The complete study is reproduced with: <font name='Courier'>python run_study.py --config configs/submission.yaml</font>.", styles["Small"]),
          PageBreak(),
          Paragraph("References", styles["PaperTitle"])]
refs = [
    "[1] A. K. Singh et al. What Needs to Go Right for an Induction Head? A Mechanistic Study of In-Context Learning Circuits and Their Formation. ICML, 2024.",
    "[2] A. Conmy et al. Towards Automated Circuit Discovery for Mechanistic Interpretability. NeurIPS 36, 2023.",
    "[3] I. Shumailov et al. The Curse of Recursion: Training on Generated Data Makes Models Forget. arXiv:2305.17493, 2023.",
    "[4] I. Shumailov et al. AI Models Collapse When Trained on Recursively Generated Data. Nature 631, 755-759, 2024.",
    "[5] C. Olsson et al. In-context Learning and Induction Heads. arXiv:2209.11895, 2022.",
]
for ref in refs:
    story.append(Paragraph(ref, styles["BodyPaper"]))

story += [Spacer(1, 10), Paragraph("Contribution statement", styles["H1paper"]),
          Paragraph("Sibonelo Mlambo: experimental design, implementation, execution, analysis, visualisation and manuscript preparation (100%). This statement assumes an individual submission and must be updated if submitted by a group.", styles["BodyPaper"]),
          Spacer(1, 10), Paragraph("AI assistance statement", styles["H1paper"]),
          Paragraph("Generative AI was used to assist with code scaffolding, debugging, editing and document formatting. The author ran and checked the experiments, reviewed the generated material, and accepts responsibility for the correctness, originality and academic integrity of the submitted work.", styles["BodyPaper"]),
          PageBreak(), Paragraph("Ethics statement - submission action required", styles["PaperTitle"]),
          Paragraph("The assignment requires the exact NeurIPS 2024 ethics questionnaire supplied on Moodle and the Faculty of Science AI Ethics Statement. Those official files were not supplied with the assignment text available during preparation and therefore have not been invented or represented as completed here.", styles["BodyPaper"]),
          Paragraph("Before submission, append the completed official NeurIPS questionnaire after the references and before any supplementary material. Replace or supplement the AI assistance statement above with the exact Faculty wording required by the course. Failure to append the official questionnaire means the project will not be marked.", styles["BodyPaper"])]

doc = SimpleDocTemplate(str(PDF), pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                        topMargin=14*mm, bottomMargin=18*mm,
                        title="Interpretability of Model Collapse",
                        author="Sibonelo Mlambo")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(PDF)
