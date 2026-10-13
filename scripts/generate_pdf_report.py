"""Generate submission-ready 2-page PDF report for Structural Damage Recognition."""

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

WORKSPACE = Path(__file__).resolve().parent.parent
PDF_OUTPUT = WORKSPACE / "docs" / "report.pdf"
PLOTS_DIR = WORKSPACE / "outputs" / "plots"


def build_pdf():
    doc = SimpleDocTemplate(
        str(PDF_OUTPUT),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#1A365D"),
        alignment=1,  # Center
        spaceAfter=3,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#4A5568"),
        alignment=1,  # Center
        spaceAfter=6,
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=5,
        spaceAfter=2,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=10.2,
        textColor=colors.HexColor("#2D3748"),
        alignment=4,  # Justified
        spaceAfter=3,
    )

    bullet_style = ParagraphStyle(
        "BulletText",
        parent=body_style,
        leftIndent=10,
        spaceAfter=2,
    )

    table_text = ParagraphStyle(
        "TableText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=9,
        alignment=1,  # Center
    )

    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7,
        leading=9,
        textColor=colors.white,
        alignment=1,
    )

    story = []

    # Title & Header
    story.append(Paragraph("Structural Damage Recognition Using Transfer Learning", title_style))
    story.append(
        Paragraph(
            "<b>Course:</b> UE24CS352A — Machine Learning (Sec-C, Team C-22, Project 31)<br/>"
            "<b>Students:</b> Mirunjai Suresh Kumar (PES1UG24AM161) &nbsp;|&nbsp; Mohammed Sinan M T (PES1UG24AM165) &nbsp;|&nbsp; "
            "<b>Date:</b> October 2026",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E0"), spaceAfter=5))

    # Abstract Box
    abstract_text = (
        "<b>Abstract—</b> Rapid post-disaster structural integrity assessment is crucial for life safety and emergency resource allocation. "
        "Manual inspection is hazardous, subjective, and slow. We develop an automated computer vision pipeline for binary structural "
        "damage recognition (Damaged vs. Undamaged) on the benchmark PEER Hub ImageNet (Φ-Net) Task 2 dataset. Tailored for consumer "
        "hardware constraints, our approach employs an ImageNet-pretrained MobileNetV2 backbone (include_top=False) with disk-backed bottleneck feature caching. "
        "Evaluated on the official held-out test set of 1,460 images, the model achieves <b>84.25% test accuracy</b>, an <b>ROC-AUC of 0.9289</b>, "
        "and a <b>Damaged class recall of 85.91%</b> (F1 = 0.85). An interactive Streamlit web application provides real-time single-image inference."
    )
    abstract_table = Table(
        [[Paragraph(abstract_text, body_style)]],
        colWidths=[540],
    )
    abstract_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EDF2F7")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(abstract_table)
    story.append(Spacer(1, 4))

    # Section 1
    story.append(Paragraph("1. Problem Statement & Motivation", h1_style))
    story.append(
        Paragraph(
            "Following earthquakes and natural disasters, thousands of building facades, joints, columns, and walls require immediate "
            "structural screening. On-site assessment by licensed structural engineers requires weeks during catastrophe scenarios. "
            "Automated image-based damage classification provides rapid triage to prioritize emergency repairs and evacuation.",
            body_style,
        )
    )

    # Section 2
    story.append(Paragraph("2. Dataset & Image Preprocessing", h1_style))
    story.append(
        Paragraph(
            "We utilize <b>PEER Hub ImageNet (Φ-Net) Task 2: Damage State</b> (Gao & Mosalam, 2018), cropped to 224×224×3: "
            "<br/>• <b>Training Set:</b> 11,811 images (6,282 Damaged [53.2%], 5,529 Undamaged [46.8%]), stored as float32 arrays (~6.62 GiB). "
            "<br/>• <b>Official Held-Out Test Set:</b> 1,460 images (745 Damaged [51.0%], 715 Undamaged [49.0%]), ~838 MiB. "
            "<br/>• <b>Validation Split:</b> Stratified 80/20 train/val split (9,448 train / 2,363 val; seed 42). Contiguous block ordering requires strict stratified shuffling. "
            "<br/>• <b>Preprocessing Inversion:</b> Dataset arrays have Caffe BGR ImageNet means [103.939, 116.779, 123.68] subtracted. We invert them to standard RGB in [0, 255] via: "
            "<i>RGB = clip(Image_BGR + μ_BGR, 0, 255)[::-1]</i>, then scale to [-1, 1] via <i>(RGB / 127.5) - 1.0</i> for MobileNetV2. Uploaded JPG/PNG photographs are standard RGB and bypass mean restoration.",
            body_style,
        )
    )

    # Section 3
    story.append(Paragraph("3. Methodology & System Architecture", h1_style))
    story.append(
        Paragraph(
            "Due to physical CPU/RAM constraints (dual-core i3, 8 GB RAM), training full CNNs end-to-end causes system lockups. We formulate "
            "<b>transfer learning with disk-backed bottleneck caching</b>: "
            "<br/>• <b>Backbone:</b> Pretrained MobileNetV2 (include_top=False, 2,257,984 frozen parameters in convolutional blocks). "
            "<br/>• <b>Disk Caching:</b> Mini-batches of 8 are passed through the frozen extractor and GlobalAveragePooling2D, writing 1,280-dim feature vectors "
            "directly to disk via <i>np.lib.format.open_memmap</i> with atomic <i>.meta.json</i> progress sidecars. Working RAM remains below 1.5 GB. "
            "<br/>• <b>Classification Head:</b> Input(1280) → Dropout(0.4) → Dense(64, ReLU) → Dense(2, Softmax), totaling 82,114 trainable parameters (2,340,098 total model parameters). "
            "<br/>• <b>Training:</b> Adam (lr=0.001), Categorical Cross-Entropy, batch size 64, EarlyStopping (patience=5), ReduceLROnPlateau.",
            body_style,
        )
    )

    # Section 4: Results
    story.append(Paragraph("4. Experimental Results & Evaluation", h1_style))
    story.append(
        Paragraph(
            "The model converged in 19 epochs. Evaluation on the <b>1,460 official held-out test images</b> produced <b>84.25% accuracy</b> and <b>0.9289 ROC-AUC</b>.",
            body_style,
        )
    )

    # Table of Metrics
    metrics_data = [
        [
            Paragraph("Metric", table_header),
            Paragraph("Damaged (Class 0)", table_header),
            Paragraph("Undamaged (Class 1)", table_header),
            Paragraph("Macro Avg", table_header),
            Paragraph("Overall Test Set", table_header),
        ],
        [
            Paragraph("Precision", table_text),
            Paragraph("<b>0.8366 (83.7%)</b>", table_text),
            Paragraph("0.8489 (84.9%)", table_text),
            Paragraph("0.8428 (84.3%)", table_text),
            Paragraph("0.8426 (84.3%)", table_text),
        ],
        [
            Paragraph("Recall", table_text),
            Paragraph("<b>0.8591 (85.9%)</b>", table_text),
            Paragraph("0.8252 (82.5%)", table_text),
            Paragraph("0.8421 (84.2%)", table_text),
            Paragraph("0.8425 (84.2%)", table_text),
        ],
        [
            Paragraph("F1-Score", table_text),
            Paragraph("<b>0.8477 (84.8%)</b>", table_text),
            Paragraph("0.8369 (83.7%)", table_text),
            Paragraph("0.8423 (84.2%)", table_text),
            Paragraph("0.8424 (84.2%)", table_text),
        ],
        [
            Paragraph("Test Accuracy", table_text),
            Paragraph("—", table_text),
            Paragraph("—", table_text),
            Paragraph("—", table_text),
            Paragraph("<b>84.25% (1230 / 1460)</b>", table_text),
        ],
        [
            Paragraph("ROC-AUC", table_text),
            Paragraph("<b>0.9289</b>", table_text),
            Paragraph("—", table_text),
            Paragraph("—", table_text),
            Paragraph("<b>0.9289</b>", table_text),
        ],
    ]

    t_metrics = Table(metrics_data, colWidths=[90, 115, 115, 110, 110])
    t_metrics.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ])
    )
    story.append(t_metrics)
    story.append(Spacer(1, 4))

    # Confusion matrix & Plot images row
    cm_path = PLOTS_DIR / "confusion_matrix.png"
    curves_path = PLOTS_DIR / "training_curves.png"

    plot_elements = []
    if cm_path.exists() and curves_path.exists():
        img_cm = Image(str(cm_path), width=2.4 * inch, height=1.9 * inch)
        img_curves = Image(str(curves_path), width=4.5 * inch, height=1.9 * inch)
        img_table = Table([[img_cm, img_curves]], colWidths=[180, 360])
        img_table.setStyle(
            TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ])
        )
        plot_elements.append(img_table)
        caption = Paragraph(
            "<b>Figure 1:</b> (Left) Official Test Set Confusion Matrix [640 TP, 590 TN, 125 FP, 105 FN]. "
            "(Right) Training and validation loss & accuracy curves over 19 epochs.",
            ParagraphStyle("FigCap", parent=body_style, fontSize=7, leading=9, alignment=1),
        )
        plot_elements.append(caption)

    story.append(KeepTogether(plot_elements))
    story.append(Spacer(1, 4))

    # Confusion matrix explanation
    story.append(
        Paragraph(
            "<b>Confusion Matrix Analysis:</b> Actual Damaged (745 total): 640 correct (TP), 105 missed (FN, 14.1% miss rate). "
            "Actual Undamaged (715 total): 590 correct (TN), 125 false alarms (FP, 17.5% alarm rate). In civil safety engineering, "
            "recall (85.91%) is prioritized to minimize dangerous structural oversights. Plausible visual failure modes for false negatives "
            "include low-contrast hairline fissures and shadow occlusions; plausible visual failure modes for false positives include high-contrast "
            "window frames, conduit lines, and architectural brick seams.",
            body_style,
        )
    )

    # Section 5: Web Application & Conclusion
    story.append(Paragraph("5. Deployment Prototype, Limitations & Future Work", h1_style))
    story.append(
        Paragraph(
            "• <b>Streamlit Web Application:</b> A functional UI (<i>app.py</i>) was built with resource-cached model loading, responsive single-photograph "
            "inference, confidence gauge, and civil engineering safety disclaimer. "
            "<br/>• <b>Limitations:</b> Binary classification does not evaluate structural damage severity (e.g. slight vs. catastrophic) or identify failure modes (shear, flexure). "
            "Resolution is fixed at 224×224 px. Predictions must never replace certified civil engineering inspection. "
            "<br/>• <b>Future Work:</b> Multi-class damage grading across Φ-Net Tasks 3–6, Grad-CAM saliency heatmaps, and edge-device mobile deployment.",
            body_style,
        )
    )

    # References
    story.append(Paragraph("References", h1_style))
    refs = (
        "1. Gao, Y., & Mosalam, K. M. (2018). Deep Transfer Learning for Image‐Based Structural Damage Recognition. <i>Comp.-Aided Civ. & Inf. Eng.</i>, 33(9), 748–768.<br/>"
        "2. Gao, Y., & Mosalam, K. M. (2020). PEER Hub ImageNet: A Large-Scale Benchmark Data Set. <i>J. Struct. Eng.</i>, 146(10), 04020198.<br/>"
        "3. Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. <i>CVPR 2018</i>, 4510–4520.<br/>"
        "4. Ho, M., & Troncoso, J. (2018). Structural Damage Image Classification. <i>Stanford CS229 Project Report</i>."
    )
    story.append(Paragraph(refs, ParagraphStyle("Refs", parent=body_style, fontSize=6.5, leading=8.5)))

    doc.build(story)
    print(f"[SUCCESS] Generated submission PDF: {PDF_OUTPUT}")


if __name__ == "__main__":
    build_pdf()
