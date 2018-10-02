from dedupe.api import Dedupe
from dedupe.labeler import ActiveLearner, DisagreementLearner, BlockLearner, RLRLearner
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm.classes import SVC, LinearSVC

import numpy
import random


class SVMLearner(RLRLearner):

    def __init__(self, data_model):
        super().__init__(data_model)

        self.svm = SVC(kernel='linear', probability=True, tol=0.0001)

    def fit_transform(self, pairs, y):
        y = numpy.array(y)
        if not y.any() and self.y.any():
            random_pair = random.choice(self.candidates)
            exact_match = (random_pair[0], random_pair[0])
            pairs = pairs + [exact_match]
            y = numpy.concatenate([y, [1]])
        elif numpy.count_nonzero(y) == len(y) and numpy.count_nonzero(self.y) == len(self.y):
            random_pair = random.choice(self.candidates)
            pairs = pairs + [random_pair]
            y = numpy.concatenate([y, [0]])

        super().fit_transform(pairs, y)

    def fit(self, X, y):
        self.y = y
        self.X = X
        self.svm.fit(X, y)

    def predict_proba(self, examples):
        return self.svm.predict_proba(examples)[:, 1].reshape(-1, 1)


class SVMDisagreementLearner(DisagreementLearner):

    def __init__(self, data_model):
        self.data_model = data_model

        self.classifier = SVMLearner(data_model)
        self.blocker = BlockLearner(data_model)

        self.learners = (self.classifier, self.blocker)
        self.y = numpy.array([])
        self.pairs = []


class SVMDedupe(Dedupe):
    classifier = SVC(kernel='linear', probability=True, tol=0.0001)
    ActiveLearner = SVMDisagreementLearner
