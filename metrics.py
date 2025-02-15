import numpy as np


class Measurement:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def _make_confusion_matrix(self, pred: np.ndarray, target: np.ndarray):
        """make confusion matrix

        Args:
            pred (numpy.ndarray): Predicted class labels (N,)
            target (numpy.ndarray): Ground truth labels (N,)
            num_classes (int): the number of classes
        """
        assert pred.shape[0] == target.shape[0], "pred and target ndarray's batchsize must have same value"
        N = pred.shape[0]
        conf_mat = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)

        for i in range(N):
            conf_mat[target[i], pred[i]] += 1
        return conf_mat

    def accuracy(self, conf_mat: np.ndarray):
        correct = np.trace(conf_mat)    # TP
        total = np.sum(conf_mat)
        return correct / total

    def precision(self, conf_mat: np.ndarray):
        total_TP, total_FP = 0, 0
        for i in range(self.num_classes):
            TP = conf_mat[i, i]
            FP = np.sum(conf_mat[:, i]) - TP
            total_TP += TP
            total_FP += FP
        precision = total_TP / (total_TP + total_FP)
        return precision

    def recall(self, conf_mat: np.ndarray):
        total_TP, total_FN = 0, 0
        for i in range(self.num_classes):
            TP = conf_mat[i, i]
            FN = np.sum(conf_mat[i, :]) - TP
            total_TP += TP
            total_FN += FN
        recall = total_TP / (total_TP + total_FN)
        return recall

    def f1score(self, recall, precision):
        return 2 * recall * precision / (recall + precision)

    def measure(self, pred: np.ndarray, target: np.ndarray):
        conf_mat = self._make_confusion_matrix(pred, target)
        acc = self.accuracy(conf_mat)
        precision = self.precision(conf_mat)
        recall = self.recall(conf_mat)
        f1score = self.f1score(recall, precision)
        return acc, precision, recall, f1score

    __call__ = measure
