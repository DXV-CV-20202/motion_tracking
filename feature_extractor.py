import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

class FeatureExtractor:
    def __init__(self, *args, **kwargs):
        pass

    def extract(self, image, *args, **kwargs):
        raise Exception("extract function must be implemented")

class SIFT(FeatureExtractor):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.extractor = cv2.SIFT_create()
        self.eps = 1e-7
        self.isRootSIFT = True
        self.size = 1024

    def extract(self, image, *args, **kwargs):
        kp, descriptor = self.extract_full(image, *args, **kwargs)
        kp_des = [(kp[i], descriptor[i]) for i in range(len(kp))]
        kp_des.sort(key=lambda x: x[0].response, reverse=True)
        if len(kp_des) > 0:
            features = np.concatenate([d[1] for d in kp_des])
            if features.shape[0] < 1024:
                features = np.concatenate([features, np.zeros(1024 - features.shape[0])])
        else:
            features = np.zeros(1024)
        return features[:1024]

    def extract_full(self, image, *args, **kwargs):
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, descriptor = self.extractor.detectAndCompute(image, None)
        if type(descriptor) == type(None):
            return kp, None
        if self.isRootSIFT == True:
            descriptor /= (descriptor.sum(axis=1, keepdims=True) + self.eps)
            descriptor = np.sqrt(descriptor)
        return kp, descriptor

    def score_by_matching_keypoint(self, keypoints_1, keypoints_2, descriptions_1, descriptions_2):
        m = len(keypoints_1)
        n = len(keypoints_2)
        cost = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                cost[i, j] = keypoints_1[i].response * keypoints_2[j].response / (np.linalg.norm(descriptions_1[i] - descriptions_2[j]) + 0.0001)
        row, col = linear_sum_assignment(cost_matrix=cost, maximize=True)
        score = np.sum([cost[row[i], col[i]] for i in range(len(row))])
        return score


class ColorHistogram(FeatureExtractor):
    def __init__(self, *args, nbins=8, type_histogram='RGB_2', **kwargs):
        super().__init__(*args, **kwargs)
        self.nbins = nbins
        self.type_histogram = type_histogram

    def extract(self, image, *args, **kwargs):
        if self.type_histogram == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            histograms = cv2.calcHist([image], [0, 1, 2], None, [8, 3, 3], [0, 180, 0, 256, 0, 256])
            cv2.normalize(histograms, histograms)
            return histograms.flatten()
        elif self.type_histogram == 'RGB_1':
            b, g, r = cv2.split(image)
            b_hist = cv2.calcHist([b], [0], None, [256], [0, 256])
            g_hist = cv2.calcHist([g], [0], None, [256], [0, 256])
            r_hist = cv2.calcHist([r], [0], None, [256], [0, 256])
            rgb_hist = np.array([r_hist, g_hist, b_hist])
            cv2.normalize(rgb_hist, rgb_hist)
            return rgb_hist.flatten()
        elif self.type_histogram == 'RGB_2':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            return hist.flatten()
        return np.zeros(self.nbins * 3)

    def score_by_distance(self, feature1, feature2):
        class EuclideanDistance:
            def __init__(self) -> None:
                pass

            def calculate_distance(self, x, y):
                return np.linalg.norm(x - y)

        metric = EuclideanDistance()
        return metric.calculate_distance(feature1, feature2)


    def score_by_correlation(self, feature1, feature2):
        score = cv2.compareHist(feature1, feature2, cv2.HISTCMP_CHISQR)
        return score
