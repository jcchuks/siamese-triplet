import numpy as np


class Metric:
    def __init__(self):
        self._metric_name = ""
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

    def set_metric_name(self, name):
        self._metric_name = name

    @property
    def metric_name(self):
        return self._metric_name


class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'


class AverageNonzeroTripletsMetric(Metric):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'


class SimilarityMetrics(Metric):
    '''
    Determines the True Accept Rate and the False Accept Rate
    '''

    def __init__(self):
        self.trueAccept = 0
        self.trueReject = 0
        self.falseAccept = 0
        self.falseReject = 0
        self.total = 0
        self.svmAcc = 0
        self.names = ["trueAccept", "trueReject", "falseAccept", "falseReject", "Accuracy", "svmAcc"]

    def __call__(self, output, target, data):
        '''
        self.trueAccepts - labels are same, model accepts/predicts as same
        self.trueReject - labels are same, model rejects as same
        self.falseAccept - labels not the same, model predicts as same
        self.falseReject - labels not same, model rejects as same
        :param positives:
        :param negatives:
        :return:
        '''
        positives, negatives, svm_perf = data[1]
        self.svmAcc = svm_perf
        #print(positives)
        #print(negatives)
        self.trueAccept += (positives == True).cpu().sum()
        self.trueReject += (positives == False).cpu().sum()
        self.falseAccept += (negatives == False).cpu().sum()
        self.falseReject += (negatives == True).cpu().sum()
        total = self.falseReject + self.trueReject + self.falseAccept + self.trueAccept
     
        self.total += ((len(positives))+ (len(negatives)))
        assert total == self.total, f"combined metric {total} is equal to length {self.total}"
        return self.value()

    def reset(self):
        self.trueAccept = 0
        self.trueReject = 0
        self.falseAccept = 0
        self.falseReject = 0
        self.total = 0

    def value(self):
        tac = 100 * float(self.trueAccept) / self.total
        tre = 100 * float(self.trueReject) / self.total
        fre = 100 * float(self.falseReject) / self.total
        fac = 100 * float(self.falseAccept) / self.total
        acc = 100 * float(self.trueAccept + self.falseReject) / self.total
        return [tac, tre, fac, fre, acc, self.svmAcc]

    def name(self):
        return self.names
