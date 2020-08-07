from sklearn.metrics import confusion_matrix, f1_score


class Evaluation(object):
    """
    precision : tp / (tp + np) [true positives out of positive predictions]
    recall : tp / (tp + fn) [true positives out of positive labels]
    accuracy : (tp + tn) / (all)
    F1-score : weighted average of the precision and recall
    """

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(
            self.y_true, self.y_pred
        ).ravel()

    @property
    def precision(self):
        return self.tp / (self.fp + self.tp)

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn)

    @property
    def acc(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    @property
    def F1_score(self):
        return f1_score(self.y_true, self.y_pred)
