from scipy.optimize import linear_sum_assignment
import numpy as np

def score_by_matching_keypoint(keypoints_1, keypoints_2, descriptions_1, descriptions_2):
    m = len(keypoints_1)
    n = len(keypoints_2)
    cost = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            cost[i, j] = keypoints_1[i].response * keypoints_2[j].response / (np.linalg.norm(descriptions_1[i] - descriptions_2[j]) + 0.0001)
    row, col = linear_sum_assignment(cost_matrix=cost, maximize=True)
    score = np.sum([cost[row[i], col[i]] for i in range(len(row))])
    return score

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

def match_object(list1, list2):
    cost = np.zeros((len(list1), len(list2)))
    for i, k1 in enumerate(list1):
        for j, k2 in enumerate(list2):
            cost[i, j] = score_by_matching_keypoint(k1.keypoints, k2.keypoints, k1.descriptions, k2.descriptions)
            cost[i, j] *= iou(k1.bbox, k2.bbox)
    row, col = linear_sum_assignment(cost_matrix=cost, maximize=True)
    matching = list(zip(row, col))
    return matching