import numpy as np
from scipy import ndimage as ndi


def evaluate(prediction, target, spacing=(1, 1, 1), tolerance=1.0):
    pred, truth = np.asarray(prediction) > 0, np.asarray(target) > 0
    if pred.shape != truth.shape:
        raise ValueError("Evaluation requires matching native-space masks")
    if len(spacing) != pred.ndim or min(spacing) <= 0:
        raise ValueError("Spacing must match mask rank and be positive")
    tp = np.logical_and(pred, truth).sum()
    fp = np.logical_and(pred, ~truth).sum()
    fn = np.logical_and(~pred, truth).sum()
    dice = float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 1.0
    iou = float(tp/(tp+fp+fn)) if tp+fp+fn else 1.0
    precision = float(tp/(tp+fp)) if tp+fp else (1.0 if not truth.any() else 0.0)
    recall = float(tp/(tp+fn)) if tp+fn else 1.0
    hd95, nsd = None, 0.0
    if pred.any() and truth.any():
        a = pred ^ ndi.binary_erosion(pred)
        b = truth ^ ndi.binary_erosion(truth)
        ab = ndi.distance_transform_edt(~b, sampling=spacing)[a]
        ba = ndi.distance_transform_edt(~a, sampling=spacing)[b]
        hd95 = float(np.percentile(np.concatenate((ab, ba)), 95))
        nsd = float(((ab <= tolerance).sum() + (ba <= tolerance).sum()) / (len(ab)+len(ba)))
    elif not pred.any() and not truth.any():
        hd95, nsd = 0.0, 1.0
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall,
            "hd95": hd95, "nsd": nsd, "surface_tolerance": tolerance,
            "fp_voxels": int(fp), "fn_voxels": int(fn),
            "surface_status": "undefined_empty_mask" if hd95 is None else "measured"}


def regions(binary, *, kind, min_voxels=4, max_regions=20):
    labels, count = ndi.label(binary)
    sizes = np.bincount(labels.ravel())
    rows = []
    for label_id in sorted(range(1, count+1), key=lambda i: int(sizes[i]), reverse=True):
        if sizes[label_id] < min_voxels:
            continue
        indices = np.argwhere(labels == label_id)
        rows.append({"kind": kind, "voxels": int(sizes[label_id]),
                     "center_zyx": indices.mean(0).round(2).tolist(),
                     "bbox_zyx": [indices.min(0).tolist(), (indices.max(0)+1).tolist()]})
        if len(rows) >= max_regions:
            break
    return rows


def failure_mining(mask, uncertainty, target=None):
    rows = regions(uncertainty > 0.9, kind="low_confidence")
    if target is not None:
        rows += regions((mask > 0) & (target == 0), kind="FP_vs_reference")
        rows += regions((mask == 0) & (target > 0), kind="FN_vs_reference")
    return {"regions": rows, "reference_available": target is not None,
            "risk_score": round(float(np.mean(uncertainty > 0.9)), 6),
            "risk_definition": "fraction of voxels with binary entropy > 0.9; workflow heuristic"}


def regression(baseline, candidate, *, dice_drop=0.01, latency_increase=0.20, vram_increase=0.20):
    checks = []
    for metric, threshold, lower_good in [('p50_ms', latency_increase, True), ('peak_allocated_mib', vram_increase, True)]:
        old, new = baseline[metric], candidate[metric]
        checks.append({"metric": metric, "baseline": old, "candidate": new,
                       "passed": new <= old*(1+threshold) if old else new == 0, "max_relative_increase": threshold})
    if baseline.get('dice') is not None and candidate.get('dice') is not None:
        checks.append({"metric": "dice", "baseline": baseline['dice'], "candidate": candidate['dice'],
                       "passed": candidate['dice'] >= baseline['dice']-dice_drop, "max_drop": dice_drop})
    return {"passed": all(c['passed'] for c in checks), "checks": checks}
