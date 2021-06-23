from scipy.optimize import linear_sum_assignment
import numpy as np
from feature_extractor import *


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    score = interArea / float(boxAArea + boxBArea - interArea)

    return score

def match_object(list1, list2, extractor):
    cost = np.zeros((len(list1), len(list2)))
    for i, k1 in enumerate(list1):
        for j, k2 in enumerate(list2):
            if isinstance(extractor, SIFT) == True:
                cost[i, j] = extractor.score_by_matching_keypoint(k1.keypoints, k2.keypoints, k1.descriptions, k2.descriptions)
            else:
                cost[i, j] = 1.0 / extractor.score_by_correlation(k1.feature, k2.feature)
            cost[i, j] *= iou(k1.bbox, k2.bbox)
    row, col = linear_sum_assignment(cost_matrix=cost, maximize=True)
    matching = list(zip(row, col))
    return matching