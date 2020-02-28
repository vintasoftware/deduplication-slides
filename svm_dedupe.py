import random

import numpy
from dedupe.api import Dedupe
from dedupe.labeler import DedupeDisagreementLearner
from dedupe.labeler import RLRLearner
from sklearn.svm.classes import SVC


def _build_model():
    return SVC(kernel='rbf', probability=True)


class SVMLearner(RLRLearner):

    def __init__(self, data_model, *args, **kwargs):
        self.svm_classifier = _build_model()
        super().__init__(data_model, *args, **kwargs)

    def fit(self, X, y):
        y = numpy.array(y)

        # This replicates Dedupe's behavior, adapting it to sklearn:
        # if there are only non-matching examples on y,
        #     grab a random record and consider it as a match with itself
        # if there are only matching examples on y,
        #     grab a random pair and consider it as a non-match
        # Also, if both X and y are empty, do both things above.
        # This happens on active learning when there's no existing training_pairs.
        if not y.any():
            random_pair = random.choice(self.candidates)
            exact_match = (random_pair[0], random_pair[0])
            X = numpy.vstack([X, self.transform([exact_match])])
            y = numpy.concatenate([y, [1]])
        if numpy.count_nonzero(y) == len(y):
            random_pair = random.choice(self.candidates)
            X = numpy.vstack([X, self.transform([random_pair])])
            y = numpy.concatenate([y, [0]])

        self.y = y
        self.X = X
        self.svm_classifier.fit(X, y)

    def predict_proba(self, examples):
        return self.svm_classifier.predict_proba(examples)[:, 1].reshape(-1, 1)


class SVMDisagreementLearner(DedupeDisagreementLearner):

    def _common_init(self):
        self.classifier = SVMLearner(self.data_model,
                                     candidates=self.candidates)
        self.learners = (self.classifier, self.blocker)
        self.y = numpy.array([])
        self.pairs = []


class SVMDedupe(Dedupe):
    classifier = _build_model()
    ActiveLearner = SVMDisagreementLearner
