"""Standalone offline HTML/JSON/inline-SVG report generator.

Builds an offline ZIP package exclusively from persisted database records,
verified CV bundles, and verified finalization bundles.
No raw dataset rows, model binaries, external assets, or network calls are included.
Report generation never triggers model refitting or holdout partition access.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
from typing import Any
import zipfile

from apps.api.database.repository import ConflictError, JobRepository, NotFoundError, _now
from apps.api.services.leaderboard import _ensure_available
from ml.contracts import TaskType
from ml.evaluation.finalize import validate_final_result_payload
from ml.tracking.manifest import canonical_bytes, digest_bytes, file_digest


def _h(value: object) -> str:
    """HTML escape dynamic values."""
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _format_metric_val(val: float | None) -> str:
    if val is None or not math.isfinite(val):
        return "—"
    return f"{val:.4f}"


def _format_delta(delta: float | None) -> str:
    if delta is None or not math.isfinite(delta):
        return "—"
    prefix = "+" if delta > 0 else ""
    return f"{prefix}{delta:.4f}"


def _format_seconds(sec: float | None) -> str:
    if sec is None or not math.isfinite(sec):
        return "—"
    return f"{sec:.2f}s"


def _render_loss_curves_svg(candidate_family: str, folds: list[dict], width: int = 540, height: int = 240) -> str:
    """Render inline SVG loss curves for an MLP candidate across folds."""
    folds_with_curves = [f for f in folds if f.get("loss_curve")]
    if not folds_with_curves:
        return "<p class='text-muted'>No per-epoch loss curve data recorded for this model.</p>"

    svg_blocks = []
    left = 55
    right = 20
    top = 35
    bottom = 40
    plot_w = width - left - right
    plot_h = height - top - bottom

    for fold in folds_with_curves:
        fold_idx = fold.get("fold_index", 0)
        best_epoch = fold.get("best_epoch")
        epochs_trained = fold.get("epochs_trained")
        stop_reason = fold.get("stop_reason")
        curve = fold.get("loss_curve", [])
        if not curve:
            continue

        train_pts = [(p["epoch"], p["train_loss"]) for p in curve if p.get("train_loss") is not None and math.isfinite(p["train_loss"])]
        val_pts = [(p["epoch"], p["val_loss"]) for p in curve if p.get("val_loss") is not None and math.isfinite(p["val_loss"])]

        all_epochs = [p[0] for p in train_pts] + [p[0] for p in val_pts]
        all_losses = [p[1] for p in train_pts] + [p[1] for p in val_pts]
        if not all_losses or not all_epochs:
            continue

        min_e = min(all_epochs)
        max_e = max(all_epochs)
        if min_e == max_e:
            max_e = min_e + 1

        min_l = min(all_losses)
        max_l = max(all_losses)
        if min_l == max_l:
            min_l = max(0.0, min_l - 0.1)
            max_l = max_l + 0.1
        else:
            span = max_l - min_l
            min_l = max(0.0, min_l - span * 0.05)
            max_l = max_l + span * 0.05

        def get_x(e: int) -> float:
            return left + (e - min_e) / (max_e - min_e) * plot_w

        def get_y(l: float) -> float:
            return top + plot_h - ((l - min_l) / (max_l - min_l)) * plot_h

        train_poly = " ".join(f"{get_x(e):.1f},{get_y(l):.1f}" for e, l in train_pts)
        val_poly = " ".join(f"{get_x(e):.1f},{get_y(l):.1f}" for e, l in val_pts)

        # SVG elements
        grid_lines = []
        # 4 horizontal ticks
        for i in range(5):
            t_val = min_l + (max_l - min_l) * (i / 4.0)
            ty = top + plot_h - (i / 4.0) * plot_h
            grid_lines.append(f"<line x1='{left}' y1='{ty:.1f}' x2='{left + plot_w}' y2='{ty:.1f}' stroke='#e2e8f0' stroke-width='1'/>")
            grid_lines.append(f"<text x='{left - 8}' y='{ty + 4:.1f}' font-size='10' fill='#64748b' text-anchor='end'>{t_val:.3f}</text>")

        # X axis ticks
        e_step = max(1, (max_e - min_e) // 4)
        for e in range(min_e, max_e + 1, e_step):
            tx = get_x(e)
            grid_lines.append(f"<line x1='{tx:.1f}' y1='{top}' x2='{tx:.1f}' y2='{top + plot_h}' stroke='#f1f5f9' stroke-width='1'/>")
            grid_lines.append(f"<text x='{tx:.1f}' y='{top + plot_h + 16}' font-size='10' fill='#64748b' text-anchor='middle'>{e}</text>")

        best_epoch_marker = ""
        if best_epoch is not None and min_e <= best_epoch <= max_e:
            bx = get_x(best_epoch)
            best_epoch_marker = (
                f"<line x1='{bx:.1f}' y1='{top}' x2='{bx:.1f}' y2='{top + plot_h}' stroke='#16a34a' stroke-width='1.5' stroke-dasharray='3,3'/>"
                f"<text x='{bx:.1f}' y='{top - 8}' font-size='10' fill='#16a34a' text-anchor='middle' font-weight='bold'>Best: {best_epoch}</text>"
            )

        title_text = f"Fold {fold_idx} Loss Curve"
        meta_info = []
        if best_epoch is not None:
            meta_info.append(f"Best: {best_epoch}")
        if epochs_trained is not None:
            meta_info.append(f"Trained: {epochs_trained}")
        if stop_reason:
            meta_info.append(f"Stop: {stop_reason}")
        if meta_info:
            title_text += f" ({', '.join(meta_info)})"

        svg_content = f"""
        <div class="fold-chart-card">
          <h4>{_h(title_text)}</h4>
          <svg viewBox="0 0 {width} {height}" class="chart-svg">
            <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" rx="4"/>
            {''.join(grid_lines)}
            <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#94a3b8" stroke-width="1.5"/>
            <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#94a3b8" stroke-width="1.5"/>
            {best_epoch_marker}
            <polyline points="{train_poly}" fill="none" stroke="#2563eb" stroke-width="2"/>
            <polyline points="{val_poly}" fill="none" stroke="#ea580c" stroke-width="2" stroke-dasharray="4,3"/>
            <!-- Legend -->
            <rect x="{left + 10}" y="{top + 8}" width="12" height="3" fill="#2563eb"/>
            <text x="{left + 26}" y="{top + 12}" font-size="10" fill="#334155" font-weight="600">Train Loss</text>
            <rect x="{left + 100}" y="{top + 8}" width="12" height="3" fill="#ea580c"/>
            <text x="{left + 116}" y="{top + 12}" font-size="10" fill="#334155" font-weight="600">Val Loss</text>
            <text x="{left + plot_w / 2}" y="{height - 8}" font-size="11" fill="#64748b" text-anchor="middle">Epoch</text>
            <text x="16" y="{top + plot_h / 2}" font-size="11" fill="#64748b" text-anchor="middle" transform="rotate(-90 16 {top + plot_h / 2})">Loss</text>
          </svg>
        </div>
        """
        svg_blocks.append(svg_content)

    return "\n".join(svg_blocks)


def _render_leaderboard_svg(ranked: list[dict], primary_metric: str, width: int = 560) -> str:
    """Render horizontal bar chart for cross-validation leaderboard."""
    if not ranked:
        return ""

    row_h = 36
    left = 160
    right = 90
    top = 25
    bottom = 25
    plot_w = width - left - right
    plot_h = len(ranked) * row_h
    height = top + plot_h + bottom

    scores = [r["mean"] for r in ranked if r.get("mean") is not None and math.isfinite(r["mean"])]
    if not scores:
        return ""

    min_s = min(scores)
    max_s = max(scores)
    if min_s >= 0:
        min_val = 0.0
        max_val = max(1.0, max_s * 1.1) if max_s > 0 else 1.0
    else:
        min_val = min_s * 1.1
        max_val = max_s * 1.1 if max_s > 0 else 0.0

    if max_val == min_val:
        max_val = min_val + 1.0

    bars = []
    for idx, cand in enumerate(ranked):
        mean = cand.get("mean")
        std = cand.get("std", 0.0)
        family = cand.get("family", "unknown")
        delta = cand.get("baseline_delta")

        y = top + idx * row_h + 6
        bar_h = 22

        if mean is not None and math.isfinite(mean):
            bar_w = max(2.0, ((mean - min_val) / (max_val - min_val)) * plot_w)
            color = "#16a34a" if (delta is not None and delta > 0) else "#3b82f6" if family != "dummy" else "#94a3b8"

            bars.append(f"<rect x='{left}' y='{y}' width='{plot_w}' height='{bar_h}' fill='#f8fafc' rx='3'/>")
            bars.append(f"<rect x='{left}' y='{y}' width='{bar_w:.1f}' height='{bar_h}' fill='{color}' rx='3'/>")

            # Error bar for std
            if std is not None and math.isfinite(std) and std > 0:
                e1 = max(left, left + ((mean - std - min_val) / (max_val - min_val)) * plot_w)
                e2 = min(left + plot_w, left + ((mean + std - min_val) / (max_val - min_val)) * plot_w)
                bars.append(f"<line x1='{e1:.1f}' y1='{y + bar_h / 2}' x2='{e2:.1f}' y2='{y + bar_h / 2}' stroke='#0f172a' stroke-width='1.5'/>")
                bars.append(f"<line x1='{e1:.1f}' y1='{y + 4}' x2='{e1:.1f}' y2='{y + bar_h - 4}' stroke='#0f172a' stroke-width='1.5'/>")
                bars.append(f"<line x1='{e2:.1f}' y1='{y + 4}' x2='{e2:.1f}' y2='{y + bar_h - 4}' stroke='#0f172a' stroke-width='1.5'/>")

            score_text = f"{mean:.4f}"
            if std is not None and math.isfinite(std):
                score_text += f" ±{std:.4f}"
            bars.append(f"<text x='{left + bar_w + 8:.1f}' y='{y + 15}' font-size='11' fill='#334155' font-weight='500'>{score_text}</text>")
        else:
            bars.append(f"<text x='{left + 8}' y='{y + 15}' font-size='11' fill='#94a3b8'>No score</text>")

        bars.append(f"<text x='{left - 10}' y='{y + 15}' font-size='12' fill='#1e293b' text-anchor='end' font-weight='600'>{_h(family)}</text>")

    return f"""
    <svg viewBox="0 0 {width} {height}" class="chart-svg">
      <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" rx="4"/>
      {''.join(bars)}
      <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#cbd5e1" stroke-width="1"/>
    </svg>
    """


def _render_confusion_matrix_html(cm: list[list[int]] | tuple[tuple[int, ...], ...]) -> str:
    """Render an HTML 2x2 confusion matrix."""
    if not cm or len(cm) != 2 or len(cm[0]) != 2 or len(cm[1]) != 2:
        return ""
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    total = tn + fp + fn + tp
    tn_pct = f"{tn / total * 100:.1f}%" if total else "0%"
    fp_pct = f"{fp / total * 100:.1f}%" if total else "0%"
    fn_pct = f"{fn / total * 100:.1f}%" if total else "0%"
    tp_pct = f"{tp / total * 100:.1f}%" if total else "0%"

    return f"""
    <div class="matrix-card">
      <h4>Holdout Confusion Matrix (Total: {total})</h4>
      <table class="matrix-table">
        <thead>
          <tr>
            <th colspan="2" rowspan="2"></th>
            <th colspan="2">Predicted Class</th>
          </tr>
          <tr>
            <th>Negative (0)</th>
            <th>Positive (1)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th rowspan="2">Actual Class</th>
            <th>Negative (0)</th>
            <td class="matrix-cell cell-tn">
              <div class="m-val">{tn}</div>
              <div class="m-lbl">TN ({tn_pct})</div>
            </td>
            <td class="matrix-cell cell-fp">
              <div class="m-val">{fp}</div>
              <div class="m-lbl">FP ({fp_pct})</div>
            </td>
          </tr>
          <tr>
            <th>Positive (1)</th>
            <td class="matrix-cell cell-fn">
              <div class="m-val">{fn}</div>
              <div class="m-lbl">FN ({fn_pct})</div>
            </td>
            <td class="matrix-cell cell-tp">
              <div class="m-val">{tp}</div>
              <div class="m-lbl">TP ({tp_pct})</div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    """


def generate_report_html(
    experiment: dict[str, Any],
    cv_data: dict[str, Any],
    final_data: dict[str, Any] | None,
    audit_events: list[dict[str, Any]],
    curves_data: dict[str, Any],
) -> str:
    """Generate self-contained, escaped, offline HTML report without external assets."""
    exp_id = experiment["experiment_id"]
    task = experiment["task"]
    target = experiment["target"]
    primary_metric = experiment["primary_metric"]
    direction = experiment["direction"]
    created_at = experiment["created_at"]
    profile = experiment["profile"]
    limits = experiment["limits"]

    is_finalized = (final_data is not None)

    # Status Banner
    if is_finalized:
        status_banner = """
        <div class="banner banner-finalized">
          <span class="badge badge-final">Finalized</span>
          <div>
            <strong>Status: Final Evaluation (Holdout Partition — Evaluated Once)</strong>
            <p>The selected winning candidate was refit on all outer training rows and scored once on the locked holdout partition.</p>
          </div>
        </div>
        """
    else:
        status_banner = """
        <div class="banner banner-cv">
          <span class="badge badge-cv">CV Only</span>
          <div>
            <strong>Status: CV only — holdout not evaluated</strong>
            <p>Cross-validation completed across training folds. The holdout partition remains locked and untouched to prevent data leakage.</p>
          </div>
        </div>
        """

    # Feature profile rows
    profile_rows = []
    features = profile.get("features", [])
    for feat in features:
        fname = feat.get("name", "")
        ftype = feat.get("dtype", "")
        missing = feat.get("missing_count", 0)
        mean_val = _format_metric_val(feat.get("mean"))
        min_val = _format_metric_val(feat.get("min"))
        max_val = _format_metric_val(feat.get("max"))
        distinct = feat.get("distinct_count", "—")
        profile_rows.append(
            f"<tr>"
            f"<td><strong>{_h(fname)}</strong></td>"
            f"<td><code>{_h(ftype)}</code></td>"
            f"<td class='num'>{missing}</td>"
            f"<td class='num'>{distinct}</td>"
            f"<td class='num'>{min_val}</td>"
            f"<td class='num'>{max_val}</td>"
            f"<td class='num'>{mean_val}</td>"
            f"</tr>"
        )

    # Profile warnings
    warnings_html = ""
    warnings = profile.get("warnings", [])
    if warnings:
        warn_items = []
        for w in warnings:
            if isinstance(w, dict):
                code = w.get("code", "")
                msg = w.get("message", "")
            else:
                code = getattr(w, "code", "")
                msg = getattr(w, "message", "")
            warn_items.append(f"<li><strong>{_h(code)}:</strong> {_h(msg)}</li>")
        warnings_html = f"""
        <div class="callout callout-warning">
          <strong>Dataset Profile Warnings:</strong>
          <ul style="margin-left: 20px; margin-top: 6px;">
            {''.join(warn_items)}
          </ul>
        </div>
        """

    # CV comparison table
    ranked = cv_data.get("comparison", {}).get("ranked", [])
    unranked = cv_data.get("comparison", {}).get("unranked", [])
    candidates = cv_data.get("candidates", [])
    cand_by_family = {c["family"]: c for c in candidates}

    cv_table_rows = []
    for rank_idx, item in enumerate(ranked, start=1):
        fam = item["family"]
        cand = cand_by_family.get(fam, {})
        cid = cand.get("candidate_id", "—")
        mean = _format_metric_val(item.get("mean"))
        std = _format_metric_val(item.get("std"))
        delta = _format_delta(item.get("baseline_delta"))
        fit_sec = _format_seconds(item.get("mean_fit_seconds"))
        total_sec = _format_seconds(cand.get("total_seconds"))
        eligible = "Yes" if cand.get("eligible", True) else "No"
        tied_note = " (tied)" if item.get("tied") else ""

        cv_table_rows.append(
            f"<tr>"
            f"<td>{rank_idx}</td>"
            f"<td><strong>{_h(fam)}</strong>{tied_note}</td>"
            f"<td class='num'><strong>{mean}</strong> ± {std}</td>"
            f"<td class='num'>{delta}</td>"
            f"<td class='num'>{fit_sec}</td>"
            f"<td class='num'>{total_sec}</td>"
            f"<td><span class='badge badge-success'>Completed</span></td>"
            f"<td>{eligible}</td>"
            f"</tr>"
        )

    for item in unranked:
        fam = item if isinstance(item, str) else item.get("family", "")
        cand = cand_by_family.get(fam, {})
        cv_table_rows.append(
            f"<tr>"
            f"<td>—</td>"
            f"<td><strong>{_h(fam)}</strong></td>"
            f"<td class='num'>—</td>"
            f"<td class='num'>—</td>"
            f"<td class='num'>—</td>"
            f"<td class='num'>—</td>"
            f"<td><span class='badge badge-danger'>Failed</span></td>"
            f"<td>No</td>"
            f"</tr>"
        )

    leaderboard_svg = _render_leaderboard_svg(ranked, primary_metric)

    # MLP Curves
    mlp_curves_html = ""
    for cand in candidates:
        if cand.get("family") == "mlp":
            folds = cand.get("folds", [])
            curves_svg = _render_loss_curves_svg("mlp", folds)
            mlp_curves_html += f"""
            <div class="subcard">
              <h3>MLP Learning Trajectory (Per-Fold Training vs. Validation Loss)</h3>
              <p class="text-muted">Early stopping monitored inner-validation loss. Best epoch selected per fold before full refit.</p>
              {curves_svg}
            </div>
            """

    # Final holdout section
    if is_finalized:
        f_res = final_data.get("result", {}) or {}
        f_metrics_bundle = f_res.get("metrics", {}) or {}
        f_sample_count = f_metrics_bundle.get("sample_count", "—")
        f_eval_sec = _format_seconds(f_res.get("evaluation_seconds"))
        f_evaluated_at = _h(f_res.get("evaluated_at", "—"))
        f_cid = _h(final_data.get("candidate_id", "—"))
        f_digest = _h(final_data.get("candidate_digest", "—"))
        f_fin_id = _h(final_data.get("finalization_id", "—"))
        f_epoch_budget = final_data.get("epoch_budget", "—")

        f_metric_rows = []
        raw_metrics = f_metrics_bundle.get("metrics", [])
        for m in raw_metrics:
            m_name = m.get("name") if isinstance(m, dict) else getattr(m, "name", "")
            m_val = m.get("value") if isinstance(m, dict) else getattr(m, "value", None)
            m_reason = m.get("reason") if isinstance(m, dict) else getattr(m, "reason", None)
            val_str = _format_metric_val(m_val)
            reason_str = f" ({_h(m_reason)})" if m_reason else ""
            is_prim = " <strong>(Primary)</strong>" if m_name == primary_metric else ""
            f_metric_rows.append(
                f"<tr>"
                f"<td><strong>{_h(m_name)}</strong>{is_prim}</td>"
                f"<td class='num'>{val_str}{reason_str}</td>"
                f"</tr>"
            )

        cm = f_metrics_bundle.get("confusion_matrix")
        cm_html = _render_confusion_matrix_html(cm) if cm else ""

        final_section = f"""
        <div class="card">
          <h2>Final Holdout Evaluation Results</h2>
          <div class="callout callout-success">
            <strong>One-Time Locked Holdout Access:</strong>
            Evaluated exactly once using private held-out partition ({f_sample_count} observations).
            Finalization ID: <code>{f_fin_id}</code>. Evaluated at: {f_evaluated_at} (Duration: {f_eval_sec}).
          </div>
          <div class="grid-2">
            <div class="stat-box">
              <div class="stat-label">Selected Candidate</div>
              <div class="stat-value"><code>{f_cid}</code></div>
              <div style="font-size: 11px; color: #64748b; margin-top: 4px;">Digest: <code>{f_digest[:16]}...</code></div>
            </div>
            <div class="stat-box">
              <div class="stat-label">MLP Epoch Budget (if applicable)</div>
              <div class="stat-value">{f_epoch_budget}</div>
            </div>
          </div>
          <h3 style="margin-top: 20px;">Holdout Metric Summary</h3>
          <table>
            <thead>
              <tr><th>Metric</th><th class="num">Score</th></tr>
            </thead>
            <tbody>
              {''.join(f_metric_rows)}
            </tbody>
          </table>
          {cm_html}
        </div>
        """
    else:
        final_section = """
        <div class="card">
          <h2>Final Holdout Evaluation Status</h2>
          <div class="callout callout-info">
            <strong>Holdout Partition Untouched (Locked):</strong>
            No candidate has been finalized on the holdout partition for this experiment.
            In accordance with ModelArena's leakage-prevention protocol, the holdout partition is strictly isolated until a user explicitly approves and initiates final evaluation.
          </div>
        </div>
        """

    # Audit events table
    audit_rows = []
    for evt in audit_events:
        ts = _h(evt.get("occurred_at", ""))
        act = _h(evt.get("actor", ""))
        ev = _h(evt.get("event", ""))
        audit_rows.append(f"<tr><td>{ts}</td><td><code>{act}</code></td><td>{ev}</td></tr>")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ModelArena Experiment Report — {_h(exp_id)}</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --card: #ffffff;
      --text: #0f172a;
      --muted: #64748b;
      --border: #e2e8f0;
      --primary: #2563eb;
      --success: #16a34a;
      --warning: #d97706;
      --danger: #dc2626;
      --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: var(--font);
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.6;
      padding: 32px 16px;
    }}
    .container {{ max-width: 980px; margin: 0 auto; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 24px;
      margin-bottom: 24px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .subcard {{
      background: #fdfefe;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 16px;
      margin-top: 16px;
    }}
    .header {{ margin-bottom: 20px; }}
    .header h1 {{ font-size: 26px; font-weight: 700; color: #0f172a; margin-bottom: 6px; }}
    .header .meta {{ color: var(--muted); font-size: 13px; }}
    .banner {{
      padding: 14px 18px;
      border-radius: 6px;
      margin-bottom: 24px;
      display: flex;
      align-items: flex-start;
      gap: 14px;
    }}
    .banner-cv {{
      background-color: #eff6ff;
      border: 1px solid #bfdbfe;
      color: #1e40af;
    }}
    .banner-finalized {{
      background-color: #f0fdf4;
      border: 1px solid #bbf7d0;
      color: #166534;
    }}
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .badge-cv {{ background: #dbeafe; color: #1e40af; }}
    .badge-final {{ background: #dcfce7; color: #166534; }}
    .badge-success {{ background: #dcfce7; color: #166534; }}
    .badge-danger {{ background: #fee2e2; color: #991b1b; }}
    h2 {{ font-size: 18px; font-weight: 700; margin-bottom: 14px; border-bottom: 1px solid var(--border); padding-bottom: 8px; color: #1e293b; }}
    h3 {{ font-size: 15px; font-weight: 600; margin: 12px 0 8px 0; color: #334155; }}
    h4 {{ font-size: 13px; font-weight: 600; margin: 8px 0 6px 0; color: #475569; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
    th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
    th {{ background-color: #f8fafc; font-weight: 600; color: #475569; }}
    tr:hover {{ background-color: #fbfcfd; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .chart-svg {{ max-width: 100%; height: auto; display: block; margin: 14px auto; }}
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 12px; }}
    .stat-box {{ background: #f8fafc; border: 1px solid var(--border); border-radius: 6px; padding: 12px 14px; }}
    .stat-label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; font-weight: 600; }}
    .stat-value {{ font-size: 18px; font-weight: 700; margin-top: 2px; color: #0f172a; }}
    .callout {{
      padding: 12px 16px;
      margin: 14px 0;
      border-radius: 6px;
      font-size: 13px;
    }}
    .callout-warning {{ border-left: 4px solid var(--warning); background: #fffbeb; color: #92400e; }}
    .callout-info {{ border-left: 4px solid var(--primary); background: #eff6ff; color: #1e40af; }}
    .callout-success {{ border-left: 4px solid var(--success); background: #f0fdf4; color: #166534; }}
    .matrix-card {{ margin-top: 20px; }}
    .matrix-table {{ margin: 12px auto; width: auto; border: 2px solid #cbd5e1; }}
    .matrix-table th, .matrix-table td {{ text-align: center; padding: 10px 20px; }}
    .matrix-cell {{ font-size: 14px; }}
    .m-val {{ font-size: 18px; font-weight: 700; }}
    .m-lbl {{ font-size: 11px; color: var(--muted); }}
    .cell-tn, .cell-tp {{ background-color: #f0fdf4; }}
    .cell-fp, .cell-fn {{ background-color: #fef2f2; }}
    .footer {{ text-align: center; font-size: 12px; color: var(--muted); margin-top: 36px; padding-top: 16px; border-top: 1px solid var(--border); }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12px; background: #f1f5f9; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>ModelArena Experiment Report</h1>
      <div class="meta">
        Experiment ID: <code>{_h(exp_id)}</code> &bull; Generated: {_h(created_at)} &bull; Task: <strong>{_h(task)}</strong>
      </div>
    </div>

    {status_banner}

    <!-- Overview Card -->
    <div class="card">
      <h2>Experiment & Dataset Metadata</h2>
      <div class="grid-2">
        <div class="stat-box">
          <div class="stat-label">Target Column</div>
          <div class="stat-value">{_h(target)}</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Primary Metric</div>
          <div class="stat-value">{_h(primary_metric)} ({_h(direction)})</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Training Rows / Columns</div>
          <div class="stat-value">{profile.get('row_count', '—')} rows / {profile.get('feature_count', '—')} features</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Dataset Raw SHA-256</div>
          <div class="stat-value"><code style="font-size: 11px;">{_h(experiment.get('raw_sha256', '—')[:20])}...</code></div>
        </div>
      </div>
      <div style="margin-top: 14px; font-size: 12px; color: var(--muted);">
        <div>Split Manifest SHA-256: <code>{_h(experiment.get('split_sha256', '—'))}</code></div>
        <div>Config SHA-256: <code>{_h(experiment.get('config_sha256', '—'))}</code></div>
      </div>
    </div>

    <!-- Training Data Profile -->
    <div class="card">
      <h2>Training-Only Dataset Profile</h2>
      <p style="font-size: 13px; color: var(--muted);">
        All statistics and warnings are computed exclusively from the 80% training split. The holdout partition is excluded from all profiling steps.
      </p>
      {warnings_html}
      <table>
        <thead>
          <tr>
            <th>Feature Name</th>
            <th>Type</th>
            <th class="num">Missing</th>
            <th class="num">Distinct</th>
            <th class="num">Min</th>
            <th class="num">Max</th>
            <th class="num">Mean</th>
          </tr>
        </thead>
        <tbody>
          {''.join(profile_rows)}
        </tbody>
      </table>
    </div>

    <!-- Cross-Validation Leaderboard -->
    <div class="card">
      <h2>Cross-Validation Leaderboard (Training Partition Only)</h2>
      <p style="font-size: 13px; color: var(--muted); margin-bottom: 12px;">
        All candidates were evaluated across outer cross-validation folds. Preprocessing transformers and tuning were fit strictly fold-local.
      </p>
      {leaderboard_svg}
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Model Family</th>
            <th class="num">{_h(primary_metric)} (Mean ± Std)</th>
            <th class="num">Baseline &Delta;</th>
            <th class="num">Fit Time</th>
            <th class="num">Total Time</th>
            <th>Status</th>
            <th>Eligible</th>
          </tr>
        </thead>
        <tbody>
          {''.join(cv_table_rows)}
        </tbody>
      </table>

      {mlp_curves_html}
    </div>

    <!-- Final Holdout Evaluation -->
    {final_section}

    <!-- Resource Limits -->
    <div class="card">
      <h2>Resource Limits & Platform Configuration</h2>
      <div class="grid-2">
        <div class="stat-box">
          <div class="stat-label">Compute Threads (T)</div>
          <div class="stat-value">{limits.get('compute_threads', '—')}</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Child RSS Soft Cap</div>
          <div class="stat-value">{limits.get('child_rss_bytes', 4294967296) // (1024 * 1024)} MiB</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">MLP Max Epochs / Patience</div>
          <div class="stat-value">{limits.get('max_epochs', 100)} / {limits.get('patience', 10)}</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">Input Size Limits</div>
          <div class="stat-value">{limits.get('max_rows', 50000)} rows / {limits.get('upload_bytes', 20000000) // 1000000} MB</div>
        </div>
      </div>
    </div>

    <!-- Audit Timeline -->
    <div class="card">
      <h2>Audit Trail & Execution Timeline</h2>
      <table>
        <thead>
          <tr><th>Timestamp</th><th>Actor</th><th>Event</th></tr>
        </thead>
        <tbody>
          {''.join(audit_rows)}
        </tbody>
      </table>
    </div>

    <!-- Educational Notes & Limitations -->
    <div class="card">
      <h2>Scientific Methodology & Educational Limitations</h2>
      <div style="font-size: 13px; line-height: 1.7; color: #334155;">
        <p><strong>Leakage Prevention:</strong> Feature transformers, encoders, and scalers were fit fold-locally during cross-validation. The holdout partition was isolated at dataset configuration time and never exposed to candidate fitting, selection, or early stopping.</p>
        <p style="margin-top: 8px;"><strong>Row Independence Assumption:</strong> The split partitions assume observation independence. Acknowledging repeated feature rows does not guarantee absence of hidden correlation or clustered observations.</p>
        <p style="margin-top: 8px;"><strong>CV vs. Holdout Variance:</strong> Cross-validation estimates the expected performance of the learning algorithm across folds; the single holdout evaluation tests the specific final refitted model. Variance between CV and holdout metrics is expected and reflects finite test sample size.</p>
        <p style="margin-top: 8px;"><strong>Class Imbalance:</strong> For binary classification tasks with imbalanced distributions, accuracy alone can be misleading. Consider F1 score, precision, recall, and PR-AUC.</p>
        <p style="margin-top: 8px;"><strong>CPU Bounds:</strong> All models were trained within local CPU and single-child process bounds without cloud or external GPU dependencies.</p>
      </div>
    </div>

    <div class="footer">
      ModelArena &bull; Leakage-safe educational machine learning on local CPU &bull; Self-contained offline report
    </div>
  </div>
</body>
</html>
"""
    return html_content


def build_report(
    repository_or_experiment_id: JobRepository | str,
    experiment_id_or_path: str | Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Build the offline HTML/JSON/inline-SVG ZIP report for an experiment.

    Only reads persisted database records, verified CV bundles, and verified
    finalization bundles. Never triggers refit or holdout partition access.
    """
    if isinstance(repository_or_experiment_id, JobRepository):
        repository = repository_or_experiment_id
        if not isinstance(experiment_id_or_path, str):
            raise TypeError("experiment_id must be a string")
        experiment_id = experiment_id_or_path
        target_path = output_path
    elif isinstance(repository_or_experiment_id, str):
        experiment_id = repository_or_experiment_id
        if isinstance(experiment_id_or_path, JobRepository):
            repository = experiment_id_or_path
            target_path = output_path
        elif isinstance(experiment_id_or_path, (str, Path)):
            target_path = Path(experiment_id_or_path)
            from apps.api.settings import load_settings
            repository = JobRepository(load_settings().metadata_path)
        else:
            from apps.api.settings import load_settings
            repository = JobRepository(load_settings().metadata_path)
            target_path = output_path
    else:
        raise TypeError("repository or experiment_id required")

    # 1. Ensure dataset is available (not pending deletion)
    _ensure_available(repository, experiment_id)

    # 2. Query experiment metadata
    with repository._connection() as db:
        row = db.execute(
            "SELECT e.experiment_id, e.primary_metric, e.direction, e.run_json, e.created_at, "
            "c.configuration_id, c.config_json, c.profile_json, c.config_sha256, "
            "s.split_id, s.manifest_sha256, s.manifest_json, "
            "d.dataset_id, d.raw_sha256, d.canonical_sha256, d.byte_count, d.row_count, d.column_count, "
            "sc.scope_id, sc.target, sc.task "
            "FROM experiments e "
            "JOIN configurations c ON c.configuration_id=e.configuration_id "
            "JOIN splits s ON s.split_id=c.split_id "
            "JOIN datasets d ON d.dataset_id=c.dataset_id "
            "JOIN evaluation_scopes sc ON sc.scope_id=c.scope_id "
            "WHERE e.experiment_id=?",
            (experiment_id,),
        ).fetchone()

    if row is None:
        raise NotFoundError(f"Experiment '{experiment_id}' not found")

    config = json.loads(row["config_json"])
    profile = json.loads(row["profile_json"])
    run = json.loads(row["run_json"])
    limits = run.get("limits", {})
    scope_id = row["scope_id"]

    # 3. Read completed CV job and verify its bundle checksums
    with repository._connection() as db:
        cv_job = db.execute(
            "SELECT job_id FROM jobs "
            "WHERE experiment_id=? AND kind='cv' AND state='completed' "
            "ORDER BY sequence DESC LIMIT 1",
            (experiment_id,),
        ).fetchone()

    if cv_job is None:
        raise ValueError(f"No completed cross-validation job for experiment '{experiment_id}'")

    cv_payload = repository.read_published_results(cv_job["job_id"])

    # 4. Check finalization status
    fin = repository.get_finalization(scope_id)
    is_finalized = (fin is not None and fin["state"] == "evaluated")
    final_payload = None
    if is_finalized:
        final_payload = repository.read_published_final_result(scope_id)

    # 5. Read audit events
    entity_ids = [experiment_id, cv_job["job_id"], scope_id]
    if fin:
        entity_ids.append(fin["finalization_id"])
    placeholders = ",".join("?" for _ in entity_ids)

    with repository._connection() as db:
        audit_rows = db.execute(
            f"SELECT occurred_at, actor, event FROM audit_events "
            f"WHERE entity_id IN ({placeholders}) ORDER BY sequence ASC",
            tuple(entity_ids),
        ).fetchall()
    audit_events = [{"occurred_at": r["occurred_at"], "actor": r["actor"], "event": r["event"]} for r in audit_rows]

    # 6. Extract curves data for curves.json
    curves_data = {"mlp": {}}
    for cand in cv_payload.get("candidates", []):
        if cand.get("family") == "mlp":
            curves_data["mlp"][cand.get("candidate_id", "mlp")] = {
                "candidate_id": cand.get("candidate_id"),
                "folds": [
                    {
                        "fold_index": f.get("fold_index"),
                        "best_epoch": f.get("best_epoch"),
                        "epochs_trained": f.get("epochs_trained"),
                        "stop_reason": f.get("stop_reason"),
                        "loss_curve": f.get("loss_curve", []),
                    }
                    for f in cand.get("folds", [])
                ],
            }

    # 7. Generate report.html
    experiment_info = {
        "experiment_id": experiment_id,
        "task": row["task"],
        "target": row["target"],
        "primary_metric": row["primary_metric"],
        "direction": row["direction"],
        "created_at": row["created_at"],
        "profile": profile,
        "limits": limits,
        "raw_sha256": row["raw_sha256"],
        "canonical_sha256": row["canonical_sha256"],
        "split_sha256": row["manifest_sha256"],
        "config_sha256": row["config_sha256"],
    }

    final_info = None
    if is_finalized:
        final_info = {
            "finalization_id": fin["finalization_id"],
            "scope_id": scope_id,
            "candidate_id": fin["candidate_id"],
            "candidate_digest": fin["candidate_digest"],
            "epoch_budget": fin.get("epoch_budget"),
            "result": final_payload,
        }

    report_html = generate_report_html(experiment_info, cv_payload, final_info, audit_events, curves_data)
    html_bytes = report_html.encode("utf-8")
    html_sha256 = digest_bytes(html_bytes)

    # 8. Generate metrics.json
    metrics_obj = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "status": "evaluated" if is_finalized else "cv_only",
        "task": row["task"],
        "target": row["target"],
        "primary_metric": row["primary_metric"],
        "direction": row["direction"],
        "cv_results": {
            "ranked": cv_payload.get("comparison", {}).get("ranked", []),
            "unranked": cv_payload.get("comparison", {}).get("unranked", []),
            "candidates": [
                {
                    "candidate_id": c.get("candidate_id"),
                    "family": c.get("family"),
                    "eligible": c.get("eligible"),
                    "total_seconds": c.get("total_seconds"),
                    "summary": c.get("summary", []),
                }
                for c in cv_payload.get("candidates", [])
            ],
        },
        "final_evaluation": final_info,
        "resource_limits": limits,
    }
    metrics_bytes = canonical_bytes(metrics_obj)
    metrics_sha256 = digest_bytes(metrics_bytes)

    # 9. Generate curves.json
    curves_obj = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "curves": curves_data,
    }
    curves_bytes = canonical_bytes(curves_obj)
    curves_sha256 = digest_bytes(curves_bytes)

    # 10. Generate manifest.json
    now = _now()
    manifest_obj = {
        "schema_version": 1,
        "origin": "modelarena-report-v1",
        "experiment_id": experiment_id,
        "generated_at": now,
        "status": "evaluated" if is_finalized else "cv_only",
        "task": row["task"],
        "target": row["target"],
        "primary_metric": row["primary_metric"],
        "direction": row["direction"],
        "dataset": {
            "dataset_id": row["dataset_id"],
            "raw_sha256": row["raw_sha256"],
            "canonical_sha256": row["canonical_sha256"],
            "row_count": row["row_count"],
            "column_count": row["column_count"],
        },
        "split_sha256": row["manifest_sha256"],
        "config_sha256": row["config_sha256"],
        "finalization": (
            {
                "finalization_id": fin["finalization_id"],
                "candidate_id": fin["candidate_id"],
                "candidate_digest": fin["candidate_digest"],
            }
            if is_finalized
            else None
        ),
        "artifact_sha256": {
            "report.html": html_sha256,
            "metrics.json": metrics_sha256,
            "curves.json": curves_sha256,
        },
    }
    manifest_bytes = canonical_bytes(manifest_obj)

    # 11. Determine output zip path
    if target_path is None:
        reports_dir = (repository.path.parent / "reports").resolve()
        reports_dir.mkdir(parents=True, exist_ok=True)
        target_path = reports_dir / f"modelarena-report-{experiment_id}.zip"
    else:
        raw_path = Path(target_path)
        if raw_path.is_symlink():
            raise ValueError("Unsafe output path")
        target_path = raw_path.resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

    # 12. Write the ZIP file
    temp_zip = target_path.parent / f".tmp-{target_path.name}"
    try:
        with zipfile.ZipFile(temp_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("report.html", html_bytes)
            zf.writestr("metrics.json", metrics_bytes)
            zf.writestr("curves.json", curves_bytes)
            zf.writestr("manifest.json", manifest_bytes)
        if target_path.exists():
            target_path.unlink()
        temp_zip.rename(target_path)
    finally:
        if temp_zip.exists():
            temp_zip.unlink()

    # 13. Verify the generated ZIP before returning
    verify_report_zip(target_path)

    return target_path


def verify_report_zip(zip_path: Path) -> dict[str, Any]:
    """Verify that a generated report ZIP archive is valid, uncorrupted, and leakage-safe."""
    zip_path = Path(zip_path).resolve()
    if not zip_path.exists() or not zip_path.is_file():
        raise ValueError(f"Report ZIP '{zip_path}' does not exist")
    if zip_path.is_symlink():
        raise ValueError("Report ZIP cannot be a symlink")

    forbidden_exts = (".pkl", ".pt", ".pth", ".bin", ".joblib", ".onnx", ".exe", ".sh", ".bat", ".py")

    with zipfile.ZipFile(zip_path, mode="r") as zf:
        names = zf.namelist()
        for name in names:
            if any(name.casefold().endswith(ext) for ext in forbidden_exts):
                raise ValueError(f"Forbidden model binary or script found in report ZIP: '{name}'")

        required = {"report.html", "manifest.json", "metrics.json", "curves.json"}
        if not required.issubset(set(names)):
            raise ValueError(f"Missing required report files in ZIP: {required - set(names)}")

        # Verify manifest
        manifest_data = json.loads(zf.read("manifest.json").decode("utf-8"))
        if manifest_data.get("schema_version") != 1 or manifest_data.get("origin") != "modelarena-report-v1":
            raise ValueError("Invalid report manifest schema or origin")

        artifact_sha = manifest_data.get("artifact_sha256", {})
        for filename, expected_hash in artifact_sha.items():
            if filename not in names:
                raise ValueError(f"Manifest references missing file '{filename}'")
            actual_bytes = zf.read(filename)
            actual_hash = hashlib.sha256(actual_bytes).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f"Checksum mismatch for '{filename}' in report ZIP")

        # Verify HTML has no external assets or scripts
        html_text = zf.read("report.html").decode("utf-8")
        if re.search(r"<script\b", html_text, re.IGNORECASE):
            raise ValueError("Report HTML contains forbidden <script> tags")
        if re.search(r"https?://", html_text, re.IGNORECASE):
            raise ValueError("Report HTML contains forbidden external URL references")
        if re.search(r"//[a-zA-Z0-9]", html_text):
            raise ValueError("Report HTML contains forbidden protocol-relative URL references")

        metrics_data = json.loads(zf.read("metrics.json").decode("utf-8"))
        curves_data = json.loads(zf.read("curves.json").decode("utf-8"))

    return {
        "manifest": manifest_data,
        "metrics": metrics_data,
        "curves": curves_data,
        "html": html_text,
    }
