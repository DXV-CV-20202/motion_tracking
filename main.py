from __future__ import print_function

import argparse
import time

import cv2 as cv
import imutils
import numpy as np

from feature_extractor import SIFT
from match_object import match_object
from moving_object import MovingObject

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2(varThreshold=24)
else:
    backSub = cv.createBackgroundSubtractorKNN()
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)

disappear_threshold = 10
num_keypoints = 16

extractor = SIFT()
moving_objects = []
all_objects = set()
_id = 0

start_time = time.time()
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    fgMask = backSub.apply(frame)

    thresh = cv.dilate(fgMask, None, iterations=0)
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    detected_objects = []

    for c in cnts:
        if cv.contourArea(c) < 400:
            continue
        (x, y, w, h) = cv.boundingRect(c)
        patch = frame[y:y+h, x:x+w]
        if np.prod(patch.shape) <= 0:
            continue
        keypoints, descriptions = extractor.extract_full(patch)
        if type(descriptions) == type(None):
            continue
        keypoint_description = list(zip(keypoints, descriptions))
        keypoint_description.sort(key=lambda x:x[0].response, reverse=True)
        keypoints = [kd[0] for kd in keypoint_description[:num_keypoints]]
        descriptions = np.array([kd[1] for kd in keypoint_description[:num_keypoints]])
        color1 = (list(np.random.choice(range(256), size=3)))
        color =[int(color1[0]), int(color1[1]), int(color1[2])]
        detected_objects.append(MovingObject(_id, (x, y, x + w, y + h), keypoints, descriptions, color))
        _id += 1
        all_objects.add(detected_objects[-1])

    matching = []
    if len(moving_objects) > 0:
        matching = match_object(moving_objects, detected_objects)

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
        moving_objects.append(detected_objects[firstseen])

    for obj in moving_objects:
        if obj.unseen_time > 0:
            continue
        xLeft, yTop, xRight, yBottom = [int(c) for c in obj.bbox]
        cv.rectangle(frame, (xLeft, yTop), (xRight, yBottom), obj.color, 2)
        cv.rectangle(frame, (xLeft, yTop), (xRight, yTop + 20), obj.color, -1)
        cv.putText(frame, str(obj._id), (xLeft, yTop + 20), cv.FONT_HERSHEY_SIMPLEX, 0.8 , (0,0,0), thickness=2)

    # print(moving_objects[-2:]) 
    
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    
    cv.imshow('Frame', frame)
    # cv.imshow('FG Mask', fgMask)
    # cv.imshow('thresh', thresh)
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

print(len(all_objects))
print(max([len(obj.tracking) for obj in all_objects]))
print(sum([len(obj.tracking) for obj in all_objects]) / len(all_objects))
print(time.time() - start_time)