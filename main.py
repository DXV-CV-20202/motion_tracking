from __future__ import print_function

import argparse
from copy import deepcopy

import cv2 as cv
import numpy as np

from feature_extractor import *
from match_object import match_object
from moving_object import MovingObject

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='data/atrium.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
parser.add_argument('--disappear_threshold', type=int, help='', default=10)
parser.add_argument('--num_keypoints', type=int, help='', default=16)
parser.add_argument('--extractor_name', type=str, help='', default='SIFT')
parser.add_argument('--contour_area', type=int, help='', default=800)
args = parser.parse_args()

disappear_threshold = args.disappear_threshold
num_keypoints = args.num_keypoints
input = args.input
algo = args.algo
if args.extractor_name == 'SIFT':
    extractor = SIFT()
elif args.extractor_name == 'ColorHistogram':
    extractor = ColorHistogram()
else:
    extractor = SIFT()
contour_area = args.contour_area

class MotionTracking:

    def __init__(self, algo=None, disappear_threshold=None, extractor=None, contour_area=None):
        self._id = 0
        self.moving_objects = []
        self.all_objects = set()
        self.algo = algo

        if algo == 'MOG2':
            self.backSub = cv.createBackgroundSubtractorMOG2(varThreshold=24)
        else:
            self.backSub = cv.createBackgroundSubtractorKNN()

    def track_frame(self, frame, frame_count=0):
        blur = cv.GaussianBlur(frame, (5, 5), 0)
        fgMask = self.backSub.apply(blur)
        median = cv.medianBlur(fgMask, 5)
        _, thresh = cv.threshold(median, 220, 255, cv.THRESH_BINARY)
        canny = cv.Canny(thresh, 150, 200)
        contour, _ = cv.findContours(deepcopy(thresh), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        detected_objects = []

        for c in contour:
            if cv.contourArea(c) < contour_area:
                continue
            (x, y, w, h) = cv.boundingRect(c)
            patch = frame[y:y+h, x:x+w]

            if np.prod(patch.shape) <= 0:
                continue
            color1 = (list(np.random.choice(range(256), size=3)))
            color =[int(color1[0]), int(color1[1]), int(color1[2])]

            if isinstance(extractor, SIFT) == True:
                keypoints, descriptions = extractor.extract_full(patch)
                if type(descriptions) == type(None):
                    continue
                keypoint_description = list(zip(keypoints, descriptions))
                keypoint_description.sort(key=lambda x: x[0].response, reverse=True)
                keypoints = [kd[0] for kd in keypoint_description[:num_keypoints]]
                descriptions = np.array([kd[1] for kd in keypoint_description[:num_keypoints]])
                detected_objects.append(MovingObject(None, (x, y, x + w, y + h), None, keypoints, descriptions, color))

            elif isinstance(extractor, ColorHistogram) == True:
                feature = extractor.extract(patch)
                detected_objects.append(MovingObject(None, (x, y, x + w, y + h), feature, None, None, color))

            self.all_objects.add(detected_objects[-1])

        matching = []
        if len(self.moving_objects) > 0:
            matching = match_object(self.moving_objects, detected_objects, extractor)

        unseen_set = set(range(len(self.moving_objects)))
        firstseen_set = set(range(len(detected_objects)))

        for m in matching:
            mo = self.moving_objects[m[0]]
            do = detected_objects[m[1]]
            mo.kalman_tracking.predict()
            bbox = mo.kalman_tracking.correct(bbox=do.bbox, flag=True)[0]
            mo.bbox = bbox
            mo.keypoints = do.keypoints
            mo.descriptions = do.descriptions
            mo.feature = do.feature
            mo.tracking.append(bbox)
            mo.unseen_time = 0
            unseen_set.remove(m[0])
            firstseen_set.remove(m[1])

        for unseen in unseen_set:
            mo = self.moving_objects[unseen]
            mo.kalman_tracking.predict()
            bbox = mo.kalman_tracking.correct(bbox=None, flag=False)[0]
            mo.bbox = bbox
            mo.tracking.append(bbox)
            mo.unseen_time += 1

        self.moving_objects = list(filter(lambda x : x.unseen_time <= disappear_threshold, self.moving_objects))

        for firstseen in firstseen_set:
            detected_objects[firstseen]._id = self._id
            self._id += 1
            self.moving_objects.append(detected_objects[firstseen])

        for obj in self.moving_objects:
            if obj.unseen_time > 5:
                continue
            xLeft, yTop, xRight, yBottom = [int(c) for c in obj.bbox]
            cv.rectangle(frame, (xLeft, yTop), (xRight, yBottom), obj.color, 2)
            cv.rectangle(frame, (xLeft, yTop), (xRight, yTop + 20), obj.color, -1)
            cv.putText(frame, str(obj._id), (xLeft, yTop + 20), cv.FONT_HERSHEY_SIMPLEX, 0.8 , (0,0,0), thickness=2)

        cv.rectangle(frame, (10, 2), (200,20), (255,255,255), -1)
        cv.putText(frame, str(frame_count) + '   No.Obj = ' + str(self._id), (15, 15),
                cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
                
        return frame, canny, thresh

    def track(self, input=None):
        capture = cv.VideoCapture(cv.samples.findFileOrKeep(input))
        if not capture.isOpened():
            print('Unable to open: ' + input)
            exit(0)

        while True:
            _, frame = capture.read()
            if frame is None:
                break

            frame, canny, thresh = self.track_frame(frame, capture.get(cv.CAP_PROP_POS_FRAMES))
            yield (frame, canny, thresh)

    def test(self, input=None):
        for res in self.track(input=input):
            frame, canny, thresh = res
            cv.imshow('Frame', frame)
            cv.imshow('FG Mask', canny)
            cv.imshow('thresh', thresh)

            keyboard = cv.waitKey(30)
            if keyboard == 'q' or keyboard == 27:
                break

        print(len(self.all_objects))
        print(max([len(obj.tracking) for obj in self.all_objects]))
        print(sum([len(obj.tracking) for obj in self.all_objects]) / len(self.all_objects))

if __name__ == '__main__':
    motion_tracking = MotionTracking(algo=algo, disappear_threshold=disappear_threshold, extractor=extractor, contour_area=contour_area)
    motion_tracking.test(input=input)
