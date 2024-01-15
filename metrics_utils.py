from tqdm import tqdm, trange
from matplotlib import axis
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
import torch
import tqdm


class MultiLabelMetrics:
    def __init__(self):
        self.TP = 0.0
        self.TN = 0.0
        self.FP = 0.0
        self.FN = 0.0
        self.total_correct = 0.0
        self.total_samples = 0.0

    def update(self, y_pred, y_true, threshold=0.5):
        y_pred_binary = (y_pred > threshold).float()

        self.TP += (y_pred_binary * y_true).sum().item()
        self.TN += ((1 - y_pred_binary) * (1 - y_true)).sum().item()
        self.FP += (y_pred_binary * (1 - y_true)).sum().item()
        self.FN += ((1 - y_pred_binary) * y_true).sum().item()

        self.total_correct += ((y_pred_binary == y_true).sum(dim=1)
                               == y_true.size(1)).float().sum().item()
        self.total_samples += y_true.size(0)

    def results(self):
        accuracy = self.total_correct / self.total_samples

        precision = self.TP / (self.TP + self.FP + 1e-10)
        recall = self.TP / (self.TP + self.FN + 1e-10)

        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return accuracy, precision, recall, f1


class MultiLabelMetrics_macro_f1:
    def __init__(self, num_classes):
        self.TP = torch.zeros(num_classes).cuda()
        self.FP = torch.zeros(num_classes).cuda()
        self.FN = torch.zeros(num_classes).cuda()

    def update(self, y_pred, y_true, threshold=0.5):
        y_pred_binary = (y_pred > threshold).float()

        self.TP += (y_pred_binary * y_true).sum(dim=0)
        self.FP += (y_pred_binary * (1 - y_true)).sum(dim=0)
        self.FN += ((1 - y_pred_binary) * y_true).sum(dim=0)

    def macro_f1(self):
        precision = self.TP / (self.TP + self.FP + 1e-10)
        recall = self.TP / (self.TP + self.FN + 1e-10)

        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        return f1.mean().item()


class MultiLabelMetrics_class:
    def __init__(self, num_classes):
        self.TP = torch.zeros(num_classes).cuda()
        self.TN = torch.zeros(num_classes).cuda()
        self.FP = torch.zeros(num_classes).cuda()
        self.FN = torch.zeros(num_classes).cuda()

    def update(self, y_pred, y_true, threshold=0.5):
        y_pred_binary = (y_pred > threshold).float()
        self.TP += (y_pred_binary * y_true).sum(dim=0)
        self.TN += ((1 - y_pred_binary) * (1 - y_true)).sum(dim=0)
        self.FP += (y_pred_binary * (1 - y_true)).sum(dim=0)
        self.FN += ((1 - y_pred_binary) * y_true).sum(dim=0)

    def results(self):
        precision = self.TP / (self.TP + self.FP + 1e-10)
        recall = self.TP / (self.TP + self.FN + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        accuracy = (self.TP + self.TN) / (self.TP +
                                          self.TN + self.FP + self.FN + 1e-10)

        return accuracy, precision, recall, f1


class AllZeroClassMetrics:
    def __init__(self):
        self.true_all_zeros = 0.0
        self.predicted_all_zeros = 0.0
        self.correctly_predicted_all_zeros = 0.0
        self.false_negatives = 0.0
        self.false_positives = 0.0

    def update(self, y_pred, y_true, threshold=0.5):
        y_pred_binary = (y_pred > threshold).float()

        true_zeros_mask = (y_true.sum(dim=1) == 0)
        pred_zeros_mask = (y_pred_binary.sum(dim=1) == 0)

        self.true_all_zeros += true_zeros_mask.sum().item()
        self.predicted_all_zeros += pred_zeros_mask.sum().item()
        self.correctly_predicted_all_zeros += (
            true_zeros_mask & pred_zeros_mask).sum().item()
        self.false_negatives += (true_zeros_mask & ~
                                 pred_zeros_mask).sum().item()
        self.false_positives += (~true_zeros_mask &
                                 pred_zeros_mask).sum().item()

    def results(self):
        accuracy = self.correctly_predicted_all_zeros / \
            (self.true_all_zeros + 1e-10)
        precision = self.correctly_predicted_all_zeros / \
            (self.predicted_all_zeros + 1e-10)
        recall = self.correctly_predicted_all_zeros / \
            (self.true_all_zeros + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        return accuracy, precision, recall, f1


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def acc(y_true, y_hat):
    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot


def qua_acc(y_true, y_hat):
    # y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    # accuracy(predicted_labels == Y_test_tensor).float().mean()
    return hit.data.float() * 1.0 / tot


def qua_accuracy(predictions, labels, threshold=0.5):
    # 将概率转化为0或1
    preds = (predictions > threshold).float()
    correct = (preds == labels).float().sum(dim=1)  # 每个样本的正确标签数
    acc = (correct == labels.shape[1]).sum()  # 完全正确的样本数
    return acc / len(labels)


def multi_label_metrics(y_pred, y_true, threshold=0.5):
    """
    y_pred: Tensor of shape (batch_size, num_labels) 
    y_true: Tensor of shape (batch_size, num_labels)
    threshold: Threshold for converting probabilities to binary predictions
    """

    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred > threshold).float()
    # True positives
    TP = (y_pred_binary * y_true).sum()

    # True negatives
    TN = ((1 - y_pred_binary) * (1 - y_true)).sum()

    # False positives
    FP = (y_pred_binary * (1 - y_true)).sum()

    # False negatives
    FN = ((1 - y_pred_binary) * y_true).sum()

    # Accuracy
    accuracy = ((y_pred_binary == y_true).sum(
        dim=1) == y_true.size(1)).float().mean()

    # Precision
    precision = TP / (TP + FP + 1e-10)

    # Recall
    recall = TP / (TP + FN + 1e-10)

    # F1 score
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return accuracy, precision, recall, f1


def macro_f1(y_pred, y_true, threshold=0.5):
    y_pred_binary = (y_pred > threshold).float()

    TP = (y_pred_binary * y_true).sum(dim=0)
    FP = (y_pred_binary * (1 - y_true)).sum(dim=0)
    FN = ((1 - y_pred_binary) * y_true).sum(dim=0)

    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)

    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return f1.mean()


def evaluate(test_impressions, user_scoring, news_scoring):
    print("evaluate...")
    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 = []
    for i in range(len(test_impressions)):
        labels = test_impressions[i]['labels']
        nids = test_impressions[i]['docs']

        uv = user_scoring[i]

        nvs = news_scoring[nids]
        score = np.dot(nvs, uv)

        auc = roc_auc_score(labels, score)
        mrr = mrr_score(labels, score)
        ndcg5 = ndcg_score(labels, score, k=5)
        ndcg10 = ndcg_score(labels, score, k=10)

        AUC.append(auc)
        MRR.append(mrr)
        nDCG5.append(ndcg5)
        nDCG10.append(ndcg10)

    AUC = np.array(AUC).mean()
    MRR = np.array(MRR).mean()
    nDCG5 = np.array(nDCG5).mean()
    nDCG10 = np.array(nDCG10).mean()
    return AUC, MRR, nDCG5, nDCG10


def evaluate_model_qua_ours(test_impressions, user_scoring, news_scoring, news_qua_scorings, news_qua, alpha):
    print("evaluate model qua...")
    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 = []

    nNQS5 = []
    nNQS3 = []
    nNQS10 = []
    # for i in range(len(test_impressions)):
    # for i in tqdm(range(len(test_impressions))):
    for i in tqdm(range(len(test_impressions)), desc='Processing'):
        labels = test_impressions[i]['labels']
        nids = test_impressions[i]['docs']

        uv = user_scoring[i]

        nvs = news_scoring[nids]
        nvq = news_qua[nids]

        news_qua_scoring = news_qua_scorings[nids]
        news_qua_scoring = np.squeeze(news_qua_scoring)  # (x,1)->(x,)
        score = np.dot(nvs, uv)
        score = score + alpha * news_qua_scoring

        auc = roc_auc_score(labels, score)
        mrr = mrr_score(labels, score)
        ndcg5 = ndcg_score(labels, score, k=5)
        ndcg10 = ndcg_score(labels, score, k=10)

        order = np.argsort(score)[::-1]
        nvq = np.array(nvq)

        nvq1 = qua_score_overall(nvq, order, k=1)
        nvq3 = qua_score_overall(nvq, order, k=3)
        nvq5 = qua_score_overall(nvq, order, k=5)
        nvq10 = qua_score_overall(nvq, order, k=10)

        AUC.append(auc)
        MRR.append(mrr)
        nDCG5.append(ndcg5)
        nDCG10.append(ndcg10)

        nNQS1.append(nvq1)
        nNQS3.append(nvq3)
        nNQS5.append(nvq5)
        nNQS10.append(nvq10)

    AUC = np.array(AUC).mean()
    MRR = np.array(MRR).mean()
    nDCG5 = np.array(nDCG5).mean()
    nDCG10 = np.array(nDCG10).mean()

    nNQS1 = np.array(nNQS1)
    n_NQS1 = nNQS1.sum(axis=0)
    nNQS3 = np.array(nNQS3)
    n_NQS3 = nNQS3.sum(axis=0)
    nNQS5 = np.array(nNQS5)
    n_NQS5 = nNQS5.sum(axis=0)
    nNQS10 = np.array(nNQS10)
    n_NQS10 = nNQS10.sum(axis=0)
    return [AUC, MRR, nDCG5, nDCG10], (n_NQS1/nNQS1.shape[0]).tolist(), (n_NQS3/nNQS3.shape[0]).tolist(), (n_NQS5/nNQS5.shape[0]).tolist(), (n_NQS10/nNQS10.shape[0]).tolist()


def evaluate_model_qua_baseline(test_impressions, user_scoring, news_scoring, news_qua):
    print("evaluate model qua baseline...")
    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 = []

    nNQS1 = []
    nNQS3 = []
    nNQS5 = []
    nNQS10 = []
    for i in range(len(test_impressions)):
        labels = test_impressions[i]['labels']
        nids = test_impressions[i]['docs']

        uv = user_scoring[i]

        nvs = news_scoring[nids]
        nvq = news_qua[nids]

        score = np.dot(nvs, uv)

        auc = roc_auc_score(labels, score)
        mrr = mrr_score(labels, score)
        ndcg5 = ndcg_score(labels, score, k=5)
        ndcg10 = ndcg_score(labels, score, k=10)

        order = np.argsort(score)[::-1]
        nvq = np.array(nvq)

        nvq1 = qua_score_overall(nvq, order, k=1)  # 判断前1的质量
        nvq3 = qua_score_overall(nvq, order, k=3)  # 判断前3的质量
        nvq5 = qua_score_overall(nvq, order, k=5)  # 判断前5的质量
        nvq10 = qua_score_overall(nvq, order, k=10)  # 判断前10的质量

        AUC.append(auc)
        MRR.append(mrr)
        nDCG5.append(ndcg5)
        nDCG10.append(ndcg10)

        nNQS1.append(nvq1)
        nNQS3.append(nvq3)
        nNQS5.append(nvq5)
        nNQS10.append(nvq10)

    AUC = np.array(AUC).mean()
    MRR = np.array(MRR).mean()
    nDCG5 = np.array(nDCG5).mean()
    nDCG10 = np.array(nDCG10).mean()

    nNQS1 = np.array(nNQS1)
    n_NQS1 = nNQS1.sum(axis=0)
    nNQS3 = np.array(nNQS3)
    n_NQS3 = nNQS3.sum(axis=0)
    nNQS5 = np.array(nNQS5)
    n_NQS5 = nNQS5.sum(axis=0)
    nNQS10 = np.array(nNQS10)
    n_NQS10 = nNQS10.sum(axis=0)
    return [AUC, MRR, nDCG5, nDCG10], (n_NQS1/nNQS1.shape[0]).tolist(), (n_NQS3/nNQS3.shape[0]).tolist(), (n_NQS5/nNQS5.shape[0]).tolist(), (n_NQS10/nNQS10.shape[0]).tolist()


def evaluate_model_qua_baseline_alpha(test_impressions, user_scoring, news_scoring, news_qua, alpha):
    print("evaluate model qua baseline...")
    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 = []
    # 对比预测impressions里前k个的质量指标
    nNQS1 = []
    nNQS3 = []
    nNQS5 = []
    nNQS10 = []
    for i in range(len(test_impressions)):
        labels = test_impressions[i]['labels']
        nids = test_impressions[i]['docs']

        uv = user_scoring[i]

        nvs = news_scoring[nids]
        nvq = news_qua[nids]

        nvq = np.array(nvq)

        score = np.dot(nvs, uv)

        # 有两种方式加权
        # 1. 只要有1个指标差 他就需要 减去 1*alpha
        # nvq_socre = np.sum(nvq ,axis = 1) # 创建一个新的数组，将水平和大于0的元素标记为1，小于等于0的元素标记为0
        # nvq_socre = np.where(nvq_socre > 0, 1, 0)
        # nvq_socre = np.squeeze(nvq_socre)
        # 2. 指标和 *alpha (相当于细粒度一些)
        nvq_socre = np.sum(nvq, axis=1)
        score = score - alpha * nvq_socre

        auc = roc_auc_score(labels, score)
        mrr = mrr_score(labels, score)
        ndcg5 = ndcg_score(labels, score, k=5)
        ndcg10 = ndcg_score(labels, score, k=10)

        order = np.argsort(score)[::-1]  # 将a中的元素从小到大排列，提取其在排列前对应的index(索引)输出。

        nvq1 = qua_score_overall(nvq, order, k=1)  # 判断前1的质量
        nvq3 = qua_score_overall(nvq, order, k=3)  # 判断前3的质量
        nvq5 = qua_score_overall(nvq, order, k=5)  # 判断前5的质量
        nvq10 = qua_score_overall(nvq, order, k=10)  # 判断前10的质量

        AUC.append(auc)
        MRR.append(mrr)
        nDCG5.append(ndcg5)
        nDCG10.append(ndcg10)

        nNQS1.append(nvq1)
        nNQS3.append(nvq3)
        nNQS5.append(nvq5)
        nNQS10.append(nvq10)

    AUC = np.array(AUC).mean()
    MRR = np.array(MRR).mean()
    nDCG5 = np.array(nDCG5).mean()
    nDCG10 = np.array(nDCG10).mean()

    nNQS1 = np.array(nNQS1)
    n_NQS1 = nNQS1.sum(axis=0)
    nNQS3 = np.array(nNQS3)
    n_NQS3 = nNQS3.sum(axis=0)
    nNQS5 = np.array(nNQS5)
    n_NQS5 = nNQS5.sum(axis=0)
    nNQS10 = np.array(nNQS10)
    n_NQS10 = nNQS10.sum(axis=0)
    return [AUC, MRR, nDCG5, nDCG10], (n_NQS1/nNQS1.shape[0]).tolist(), (n_NQS3/nNQS3.shape[0]).tolist(), (n_NQS5/nNQS5.shape[0]).tolist(), (n_NQS10/nNQS10.shape[0]).tolist()


def evaluate_model_qua(test_impressions, user_scoring, news_scoring, user_qua_scorings, news_qua_scorings, news_qua, alpha):
    print("evaluate qua...")
    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 = []
    # 对比预测impressions里前k个的质量指标
    nNQS1 = []
    nNQS5 = []
    nNQS10 = []
    for i in range(len(test_impressions)):
        labels = test_impressions[i]['labels']
        nids = test_impressions[i]['docs']

        uv = user_scoring[i]
        uvq = user_qua_scorings[i]

        nvs = news_scoring[nids]
        nvsq = news_qua_scorings[nids]
        nvq = news_qua[nids]

        score = np.dot(nvs, uv)
        scorequa = np.dot(nvsq, uvq)
        score = score + alpha * scorequa

        auc = roc_auc_score(labels, score)
        mrr = mrr_score(labels, score)
        ndcg5 = ndcg_score(labels, score, k=5)
        ndcg10 = ndcg_score(labels, score, k=10)

        nvq1 = qua_score(nvq, score, k=1)  # 判断前1的质量
        nvq5 = qua_score(nvq, score, k=5)  # 判断前1的质量
        nvq10 = qua_score(nvq, score, k=10)  # 判断前1的质量

        AUC.append(auc)
        MRR.append(mrr)
        nDCG5.append(ndcg5)
        nDCG10.append(ndcg10)

        nNQS1.append(nvq1)
        nNQS5.append(nvq5)
        nNQS10.append(nvq10)

    AUC = np.array(AUC).mean()
    MRR = np.array(MRR).mean()
    nDCG5 = np.array(nDCG5).mean()
    nDCG10 = np.array(nDCG10).mean()

    nNQS1 = np.array(nNQS1)
    n_NQS1 = nNQS1.sum(axis=0)
    nNQS5 = np.array(nNQS5)
    n_NQS5 = nNQS5.sum(axis=0)
    nNQS10 = np.array(nNQS10)
    n_NQS10 = nNQS10.sum(axis=0)

    return AUC, MRR, nDCG5, nDCG10, (n_NQS1/nNQS1.shape[0]).tolist(), (n_NQS5/nNQS5.shape[0]).tolist(), (n_NQS10/nNQS10.shape[0]).tolist()


def qua_score(nvq, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    nvq = np.array(nvq)
    y_nvq = np.take(nvq, order[:k], axis=0)
    # print(y_nvq.sum(axis=0))
    nqs = y_nvq.sum(axis=0)

    return nqs/y_nvq.shape[0]


def qua_score_overall_ori(nvq, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    nvq = np.array(nvq)
    y_nvq = np.take(nvq, order[:k], axis=0)

    nqs = y_nvq.sum(axis=0)

    horizontal_sum_nqs_overall = np.sum(y_nvq, axis=1)

    nqs_overall = np.where(horizontal_sum_nqs_overall > 0, 1, 0)

    nqs_overall = np.sum(nqs_overall)
    final_nqs = np.append(nqs, nqs_overall)

    return final_nqs/y_nvq.shape[0]


def qua_score_overall(nvq, order, k=10):

    y_nvq = np.take(nvq, order[:k], axis=0)

    nqs = y_nvq.sum(axis=0)

    horizontal_sum_nqs_overall = np.sum(y_nvq, axis=1)

    nqs_overall = np.where(horizontal_sum_nqs_overall > 0, 1, 0)

    nqs_overall = np.sum(nqs_overall)
    final_nqs = np.append(nqs, nqs_overall)

    return final_nqs/y_nvq.shape[0]


def evaluate_top(test_impressions, user_scoring, news_scoring, news_qua):

    nNQS1 = []
    nNQS5 = []
    nNQS10 = []
    for i in range(len(test_impressions)):
        labels = test_impressions[i]['labels']
        nids = test_impressions[i]['docs']

        uv = user_scoring[i]
        nvs = news_scoring[nids]
        score = np.dot(nvs, uv)

        nvq = news_qua[nids]

        nvq1 = qua_score(nvq, score, k=1)
        nvq5 = qua_score(nvq, score, k=5)
        nvq10 = qua_score(nvq, score, k=10)

        nNQS1.append(nvq1)
        nNQS5.append(nvq5)
        nNQS10.append(nvq10)

    nNQS1 = np.array(nNQS1)
    n_NQS1 = nNQS1.sum(axis=0)

    nNQS5 = np.array(nNQS5)
    n_NQS5 = nNQS5.sum(axis=0)

    nNQS10 = np.array(nNQS10)
    n_NQS10 = nNQS10.sum(axis=0)
    # print(n_NQS5/nNQS5.shape[0])
    return (n_NQS1/nNQS1.shape[0]).tolist(), (n_NQS5/nNQS5.shape[0]).tolist(), (n_NQS10/nNQS10.shape[0]).tolist()


def evaluate_cold(test_impressions, clicks, user_scoring, news_scoring):
    colds = {}
    for i in range(6):
        colds[i] = [[], [], [], []]

    for i in range(len(test_impressions)):
        ucs = clicks[i]
        ucs = (ucs > 0).sum()
        ucs = int(ucs)
        if ucs > 5:
            continue

        labels = test_impressions[i]['labels']
        nids = test_impressions[i]['docs']

        uv = user_scoring[i]

        nvs = news_scoring[nids]
        score = np.dot(nvs, uv)

        auc = roc_auc_score(labels, score)
        mrr = mrr_score(labels, score)
        ndcg5 = ndcg_score(labels, score, k=5)
        ndcg10 = ndcg_score(labels, score, k=10)

        colds[ucs][0].append(auc)
        colds[ucs][1].append(mrr)
        colds[ucs][2].append(ndcg5)
        colds[ucs][3].append(ndcg10)
    for ucs in range(6):
        colds[ucs] = np.array(colds[ucs]).mean(axis=-1)

    return colds


def evaluate_cold(test_impressions, clicks, user_scoring, news_scoring):
    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 = []
    for i in range(len(test_impressions)):
        ucs = clicks[i]
        ucs = (ucs > 0).sum()
        ucs = int(ucs)
        if ucs >= 5:
            continue
        if ucs == 0:
            continue

        labels = test_impressions[i]['labels']
        nids = test_impressions[i]['docs']

        uv = user_scoring[i]

        nvs = news_scoring[nids]
        score = np.dot(nvs, uv)

        auc = roc_auc_score(labels, score)
        mrr = mrr_score(labels, score)
        ndcg5 = ndcg_score(labels, score, k=5)
        ndcg10 = ndcg_score(labels, score, k=10)

        AUC.append(auc)
        MRR.append(mrr)
        nDCG5.append(ndcg5)
        nDCG10.append(ndcg10)

    AUC = np.array(AUC).mean()
    MRR = np.array(MRR).mean()
    nDCG5 = np.array(nDCG5).mean()
    nDCG10 = np.array(nDCG10).mean()

    return AUC, MRR, nDCG5, nDCG10


def evaluate_recall_model_qua_baseline_alpha(test_impressions, user_scoring, news_scoring, news_qua, alpha):

    nNRQS10 = []
    nNRQS50 = []
    nNRQS100 = []
    nNRQS500 = []
    nNRQS1000 = []
    for i in range(len(test_impressions)):
        uv = user_scoring[i]
        nvs = news_scoring
        score = np.dot(nvs, uv)
        nvq = news_qua

        nvq_socre = np.sum(nvq, axis=1)
        score = score - alpha * nvq_socre

        order = np.argsort(score)[::-1]
        nvq = np.array(nvq)

        nrqs10 = qua_score_overall(nvq, order, k=10)
        nrqs50 = qua_score_overall(nvq, order, k=50)
        nrqs100 = qua_score_overall(nvq, order, k=100)
        nrqs500 = qua_score_overall(nvq, order, k=500)
        nrqs1000 = qua_score_overall(nvq, order, k=1000)

        nNRQS10.append(nrqs10)
        nNRQS50.append(nrqs50)
        nNRQS100.append(nrqs100)
        nNRQS500.append(nrqs500)
        nNRQS1000.append(nrqs1000)

    nNRQS10 = np.array(nNRQS10)
    n_NRQS10 = nNRQS10.sum(axis=0)

    nNRQS50 = np.array(nNRQS50)
    n_NRQS50 = nNRQS50.sum(axis=0)

    nNRQS100 = np.array(nNRQS100)
    n_NRQS100 = nNRQS100.sum(axis=0)
    nNRQS100 = np.array(nNRQS100)
    n_NRQS100 = nNRQS100.sum(axis=0)

    nNRQS500 = np.array(nNRQS500)
    n_NRQS500 = nNRQS500.sum(axis=0)

    nNRQS1000 = np.array(nNRQS1000)
    n_NRQS1000 = nNRQS1000.sum(axis=0)
    return [(n_NRQS10/nNRQS10.shape[0]).tolist(), (n_NRQS50/nNRQS50.shape[0]).tolist(), (n_NRQS100/nNRQS100.shape[0]).tolist(), (n_NRQS500/nNRQS500.shape[0]).tolist(), (n_NRQS1000/nNRQS1000.shape[0]).tolist()]


def evaluate_recall_model_qua_baseline(test_impressions, user_scoring, news_scoring, news_qua):

    nNRQS10 = []
    nNRQS50 = []
    nNRQS100 = []
    nNRQS500 = []
    nNRQS1000 = []
    for i in range(len(test_impressions)):
        uv = user_scoring[i]
        nvs = news_scoring
        score = np.dot(nvs, uv)
        nvq = news_qua

        order = np.argsort(score)[::-1]
        nvq = np.array(nvq)

        nrqs10 = qua_score_overall(nvq, order, k=10)
        nrqs50 = qua_score_overall(nvq, order, k=50)
        nrqs100 = qua_score_overall(nvq, order, k=100)
        nrqs500 = qua_score_overall(nvq, order, k=500)
        nrqs1000 = qua_score_overall(nvq, order, k=1000)

        nNRQS10.append(nrqs10)
        nNRQS50.append(nrqs50)
        nNRQS100.append(nrqs100)
        nNRQS500.append(nrqs500)
        nNRQS1000.append(nrqs1000)

    nNRQS10 = np.array(nNRQS10)
    n_NRQS10 = nNRQS10.sum(axis=0)

    nNRQS50 = np.array(nNRQS50)
    n_NRQS50 = nNRQS50.sum(axis=0)

    nNRQS100 = np.array(nNRQS100)
    n_NRQS100 = nNRQS100.sum(axis=0)
    nNRQS100 = np.array(nNRQS100)
    n_NRQS100 = nNRQS100.sum(axis=0)

    nNRQS500 = np.array(nNRQS500)
    n_NRQS500 = nNRQS500.sum(axis=0)

    nNRQS1000 = np.array(nNRQS1000)
    n_NRQS1000 = nNRQS1000.sum(axis=0)
    return [(n_NRQS10/nNRQS10.shape[0]).tolist(), (n_NRQS50/nNRQS50.shape[0]).tolist(), (n_NRQS100/nNRQS100.shape[0]).tolist(), (n_NRQS500/nNRQS500.shape[0]).tolist(), (n_NRQS1000/nNRQS1000.shape[0]).tolist()]


def evaluate_recall_model_qua_ours(test_impressions, user_scoring, news_scoring, news_qua_scorings, news_qua, alpha):
    print("evaluate_recall_model_qua_ours...")
    # 计算原始模型（不包含quality）
    # 用user emb和所有news emb做内积 然后把top k拿出来
    nNRQS10 = []
    nNRQS50 = []
    nNRQS100 = []
    nNRQS500 = []
    nNRQS1000 = []
    # for i in range(len(test_impressions)):
    for i in tqdm(range(len(test_impressions)), desc='Processing'):
        uv = user_scoring[i]
        nvs = news_scoring
        score = np.dot(nvs, uv)
        news_qua_scorings = np.squeeze(news_qua_scorings)  # (x,1)->(x,)
        score = score + alpha * news_qua_scorings

        nvq = news_qua

        order = np.argsort(score)[::-1]
        nvq = np.array(nvq)

        nrqs10 = qua_score_overall(nvq, order, k=10)
        nrqs50 = qua_score_overall(nvq, order, k=50)
        nrqs100 = qua_score_overall(nvq, order, k=100)
        nrqs500 = qua_score_overall(nvq, order, k=500)
        nrqs1000 = qua_score_overall(nvq, order, k=1000)
        nNRQS10.append(nrqs10)
        nNRQS50.append(nrqs50)
        nNRQS100.append(nrqs100)
        nNRQS500.append(nrqs500)
        nNRQS1000.append(nrqs1000)

    nNRQS10 = np.array(nNRQS10)
    n_NRQS10 = nNRQS10.sum(axis=0)

    nNRQS50 = np.array(nNRQS50)
    n_NRQS50 = nNRQS50.sum(axis=0)

    nNRQS100 = np.array(nNRQS100)
    n_NRQS100 = nNRQS100.sum(axis=0)
    nNRQS100 = np.array(nNRQS100)
    n_NRQS100 = nNRQS100.sum(axis=0)

    nNRQS500 = np.array(nNRQS500)
    n_NRQS500 = nNRQS500.sum(axis=0)

    nNRQS1000 = np.array(nNRQS1000)
    n_NRQS1000 = nNRQS1000.sum(axis=0)
    return [(n_NRQS10/nNRQS10.shape[0]).tolist(), (n_NRQS50/nNRQS50.shape[0]).tolist(), (n_NRQS100/nNRQS100.shape[0]).tolist(), (n_NRQS500/nNRQS500.shape[0]).tolist(), (n_NRQS1000/nNRQS1000.shape[0]).tolist()]


def evaluate_recall_model_qua(test_impressions, user_scoring, news_scoring, user_qua_scoring, news_qua_scoring, news_qua, alpha):
    # 计算原始模型（包含quality 然后计算score）
    # 用user emb和所有news emb做内积 然后把top k拿出来
    nNRQS10 = []
    nNRQS50 = []
    nNRQS100 = []
    nNRQS500 = []
    nNRQS1000 = []
    for i in range(len(test_impressions)):
        uv = user_scoring[i]
        uvq = user_qua_scoring[i]

        nvs = news_scoring
        nvsq = news_qua_scoring
        nvq = news_qua

        score = np.dot(nvs, uv)
        scorequa = np.dot(nvsq, uvq)
        score = score + alpha*scorequa

        order = np.argsort(score)[::-1]
        nvq = np.array(nvq)

        y_nvq_10 = np.take(nvq, order[:10], axis=0)
        nqs_10 = y_nvq_10.sum(axis=0)
        nrqs10 = nqs_10/y_nvq_10.shape[0]

        y_nvq_50 = np.take(nvq, order[:50], axis=0)
        nqs_50 = y_nvq_50.sum(axis=0)
        nrqs50 = nqs_50/y_nvq_50.shape[0]

        y_nvq_100 = np.take(nvq, order[:100], axis=0)
        nqs_100 = y_nvq_100.sum(axis=0)
        nrqs100 = nqs_100/y_nvq_100.shape[0]

        y_nvq_500 = np.take(nvq, order[:500], axis=0)
        nqs_500 = y_nvq_500.sum(axis=0)
        nrqs500 = nqs_500/y_nvq_500.shape[0]

        y_nvq_1000 = np.take(nvq, order[:1000], axis=0)
        nqs_1000 = y_nvq_1000.sum(axis=0)
        nrqs1000 = nqs_1000/y_nvq_1000.shape[0]

        nNRQS10.append(nrqs10)
        nNRQS50.append(nrqs50)
        nNRQS100.append(nrqs100)
        nNRQS500.append(nrqs500)
        nNRQS1000.append(nrqs1000)

    nNRQS10 = np.array(nNRQS10)
    n_NRQS10 = nNRQS10.sum(axis=0)

    nNRQS50 = np.array(nNRQS50)
    n_NRQS50 = nNRQS50.sum(axis=0)

    nNRQS100 = np.array(nNRQS100)
    n_NRQS100 = nNRQS100.sum(axis=0)
    nNRQS100 = np.array(nNRQS100)
    n_NRQS100 = nNRQS100.sum(axis=0)

    nNRQS500 = np.array(nNRQS500)
    n_NRQS500 = nNRQS500.sum(axis=0)

    nNRQS1000 = np.array(nNRQS1000)
    n_NRQS1000 = nNRQS1000.sum(axis=0)
    return [(n_NRQS10/nNRQS10.shape[0]).tolist(), (n_NRQS50/nNRQS50.shape[0]).tolist(), (n_NRQS100/nNRQS100.shape[0]).tolist(), (n_NRQS500/nNRQS500.shape[0]).tolist(), (n_NRQS1000/nNRQS1000.shape[0]).tolist()]
