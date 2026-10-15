import numpy as np
from scipy.integrate import trapezoid
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def detection_metrics(id_score, ood_score, validation_threshold, labels=None, predicted=None):
    id_score, ood_score = np.asarray(id_score), np.asarray(ood_score)
    y = np.r_[np.zeros(len(id_score)), np.ones(len(ood_score))]
    score = np.r_[id_score, ood_score]
    fpr, tpr, _ = roc_curve(y, score)
    point = np.flatnonzero(tpr >= 0.95)[0]
    threshold95 = np.quantile(id_score, 0.95, method="higher")
    metrics = {"auroc": float(roc_auc_score(y, score)),
               "aupr_out": float(average_precision_score(y, score)),
               "fpr95_ood_detection": float(fpr[point]),
               "oar_at_95id": float(np.mean(ood_score <= threshold95)),
               "curve_id_acceptance": float(np.mean(id_score <= threshold95)),
               "validation_locked_ood_acceptance": float(np.mean(ood_score <= validation_threshold)),
               "validation_locked_id_frr": float(np.mean(id_score > validation_threshold))}
    if labels is not None and predicted is not None:
        correct = np.asarray(labels) == np.asarray(predicted)
        order = np.argsort(score, kind="stable")
        unique_end = np.r_[np.flatnonzero(np.diff(score[order]) != 0), len(score) - 1]
        ccr = np.r_[0, np.cumsum(np.r_[correct, np.zeros(len(ood_score))][order])[unique_end] / len(id_score)]
        oar = np.r_[0, np.cumsum(y[order])[unique_end] / len(ood_score)]
        metrics["oscr"] = float(trapezoid(ccr, oar))
    return metrics
