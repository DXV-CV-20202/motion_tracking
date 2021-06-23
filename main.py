from __future__ import print_function

import argparse
import time

import cv2 as cv

import numpy as np

from feature_extractor import *
from match_object import match_object
from moving_object import MovingObject

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

disappear_threshold = 10
num_keypoints = 16
extractor = SIFT()
input = 'data/visiontraffic.avi'
algo = 'MOG2'
extractor = SIFT()
#extractor = ColorHistogram()
contour_area = 800

def motion_tracking(input=None, algo=None, disappear_threshold=None, extractor=None, contour_area=None):
    _id = 0
    moving_objects = []
    all_objects = set()

    if algo == 'MOG2':
        backSub = cv.createBackgroundSubtractorMOG2(varThreshold=24)
    else:
        backSub = cv.createBackgroundSubtractorKNN()
    capture = cv.VideoCapture(cv.samples.findFileOrKeep(input))
    if not capture.isOpened():
        print('Unable to open: ' + args.input)
        exit(0)

    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        blur = cv.GaussianBlur(frame, (5, 5), 0)
        fgMask = backSub.apply(blur)
        median = cv.medianBlur(fgMask, 5)
        _, thresh = cv.threshold(median, 220, 255, cv.THRESH_BINARY)
        canny = cv.Canny(thresh, 150, 200)
        contour, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

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

            all_objects.add(detected_objects[-1])

        matching = []
        if len(moving_objects) > 0:
            matching = match_object(moving_objects, detected_objects, extractor)

        unseen_set = set(range(len(moving_objects)))
        firstseen_set = set(range(len(detected_objects)))

        for m in matching:
            mo = moving_objects[m[0]]
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
            mo = moving_objects[unseen]
            mo.kalman_tracking.predict()
            bbox = mo.kalman_tracking.correct(bbox=None, flag=False)[0]
            mo.bbox = bbox
            mo.tracking.append(bbox)
            mo.unseen_time += 1

        moving_objects = list(filter(lambda x : x.unseen_time <= disappear_threshold, moving_objects))

        for firstseen in firstseen_set:
            detected_objects[firstseen]._id = _id
            _id += 1
            moving_objects.append(detected_objects[firstseen])

        for obj in moving_objects:
            if obj.unseen_time > 5:
                continue
            xLeft, yTop, xRight, yBottom = [int(c) for c in obj.bbox]
            cv.rectangle(frame, (xLeft, yTop), (xRight, yBottom), obj.color, 2)
            cv.rectangle(frame, (xLeft, yTop), (xRight, yTop + 20), obj.color, -1)
            cv.putText(frame, str(obj._id), (xLeft, yTop + 20), cv.FONT_HERSHEY_SIMPLEX, 0.8 , (0,0,0), thickness=2)

        cv.rectangle(frame, (10, 2), (200,20), (255,255,255), -1)
        cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)) + '   No.Obj = ' + str(_id), (15, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

        cv.imshow('Frame', frame)
        cv.imshow('FG Mask', canny)
        # cv.imshow('thresh', thresh)

        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

    print(len(all_objects))
    print(max([len(obj.tracking) for obj in all_objects]))
    print(sum([len(obj.tracking) for obj in all_objects]) / len(all_objects))

if __name__ == '__main__':
    motion_tracking(input=input, algo=algo, disappear_threshold=disappear_threshold, extractor=extractor, contour_area=contour_area)