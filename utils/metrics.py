import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, precision_score, recall_score, brier_score_loss


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01 * (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def BrierScore(pred, true):
    # 將 true 標籤轉換為二元格式以便計算 Brier Score
    true_binary = (true >= 0.5).astype(int)  # 假設 0.5 是分界線
    return brier_score_loss(true_binary, pred)


def LogLoss(pred, true):
    # 使用 scikit-learn 的 log_loss 函數
    true_binary = (true >= 0.5).astype(int)
    return log_loss(true_binary, pred)


def ROCAUC(pred, true):
    # 使用 scikit-learn 的 roc_auc_score 函數
    true_binary = (true >= 0.5).astype(int)
    return roc_auc_score(true_binary, pred)


def PrecisionRecall(pred, true):
    # 使用 scikit-learn 的 precision_score 和 recall_score
    true_binary = (true >= 0.5).astype(int)
    pred_binary = (pred >= 0.5).astype(int)
    precision = precision_score(true_binary, pred_binary)
    recall = recall_score(true_binary, pred_binary)
    return precision, recall


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    # rmse = RMSE(pred, true)
    # mape = MAPE(pred, true)
    # mspe = MSPE(pred, true)
    # rse = RSE(pred, true)
    # corr = CORR(pred, true)

    brier = BrierScore(pred, true)
    logloss = LogLoss(pred, true)
    roc_auc = ROCAUC(pred, true)
    precision, recall = PrecisionRecall(pred, true)

    return {
        "MAE": mae,
        "MSE": mse,
        # "RMSE": rmse,
        # "MAPE": mape,
        # "MSPE": mspe,
        # "RSE": rse,
        # "CORR": corr,
        "Brier Score": brier,
        "Log Loss": logloss,
        "ROC AUC": roc_auc,
        "Precision": precision,
        "Recall": recall
    }
