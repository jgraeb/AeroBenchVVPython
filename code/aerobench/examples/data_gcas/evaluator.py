# evaluator class comparing different flight computer results
from mahalanobis import mahalanobis_distance

class Evaluator:
    def __init__(data):
        self.data = data

    def evaluate(self,data1,data2):
        mahal1 = mahalanobis_distance(data1,data)
