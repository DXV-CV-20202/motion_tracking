import cv2 as cv
import numpy as np

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r], dtype=np.float32).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.], dtype=np.float32).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score], dtype=np.float32).reshape(
            (1, 5))

class KalmanTracking:
    def __init__(self, bbox):
        kalman = cv.KalmanFilter(7, 4, 0)
        dt = 0.2
        F = np.array([
            [1, 0, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)
        Q = 1e-1 * np.eye(7, dtype=np.float32)
        Q[-1, -1] *= 0.01
        Q[4:, 4:] *= 0.01
        R = 1e-1 * np.eye(4, dtype=np.float32)
        R[2:, 2:] *= 10
        P = 1e-1 * np.eye(7, dtype=np.float32)
        P[4:, 4:] *= 100
        kalman.transitionMatrix = F
        kalman.measurementMatrix = H
        kalman.processNoiseCov = Q
        kalman.measurementNoiseCov = R
        kalman.errorCovPost = P
        state = convert_bbox_to_z(bbox)
        kalman.statePost = np.array(list(state[:, 0]) + ([0] * 3), dtype=np.float32)
        # kalman.statePost = np.array([[state[0][0]], [state[1][0]], [state[2][0]], [state[3][0]], [0], [0], [0]], dtype=np.float32)
        self.kalman = kalman
        self.lastResult = bbox

    def predict(self):
        if (self.kalman.statePost[2] + self.kalman.statePost[6]) <= 0:
            self.kalman.statePost[6] *= 0.0
        prediction = self.kalman.predict()
        self.lastResult = convert_x_to_bbox(prediction)
        return self.lastResult

    def correct(self, bbox=None, flag=True):
        if not flag:
            measurement = self.lastResult
        else:
            measurement = bbox
        measurement = np.array(measurement, dtype=np.float32)
        measurement = convert_bbox_to_z(measurement.reshape((4, 1)))
        y = measurement - np.dot(self.kalman.measurementMatrix, self.kalman.statePre)
        C = np.dot(np.dot(self.kalman.measurementMatrix, self.kalman.errorCovPre), self.kalman.measurementMatrix.T) + self.kalman.measurementNoiseCov
        K = np.dot(np.dot(self.kalman.errorCovPre, self.kalman.measurementMatrix.T), np.linalg.inv(C))
        self.kalman.statePost = self.kalman.statePre + np.dot(K, y)
        self.kalman.errorCovPost = self.kalman.errorCovPre - np.dot(K, np.dot(C, K.T))
        estimate = self.kalman.statePost
        self.lastResult = convert_x_to_bbox(estimate)
        return self.lastResult