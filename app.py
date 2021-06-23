# import the necessary packages
import argparse
import threading
import time
from copy import deepcopy

import cv2
from flask import Flask, Response, render_template

from feature_extractor import SIFT, ColorHistogram
from main import MotionTracking

original_frame = None
frame = None
canny = None
thresh = None

lock = threading.Lock()
app = Flask(__name__)

time.sleep(2.0)

@app.route("/")
def index():
    return render_template("index.html")


def track_object(algo, disappear_threshold, extractor, contour_area):
    global vs, original_frame, frame, canny, thresh

    # Basically, you just need to assign the processed value to the variables
    # original_frame, keyed_frame and tracked_frame for this to work

    md = MotionTracking(algo, disappear_threshold, extractor, contour_area)
    while True:
        _, frame = vs.read()
        original_frame = deepcopy(frame)
        frame, canny, thresh = md.track_frame(frame, vs.get(cv2.CAP_PROP_POS_FRAMES))

def generate_frame(frame_type):
    global vs, original_frame, frame, canny, thresh
    output_frame = None
    while True:
        if frame_type == 'original':
            output_frame = original_frame
        elif frame_type == 'frame':
            output_frame = frame
        elif frame_type == 'canny':
            output_frame = canny
        elif frame_type == 'thresh':
            output_frame == thresh
        else:
            raise Exception('Unknown frame type')

        if output_frame is None:
            continue

        (flag, encoded_img) = cv2.imencode(".jpg", output_frame)
        if not flag:
            continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encoded_img) + b'\r\n')


@app.route("/video_feed/<frame_type>")
def video_feed(frame_type):
    return Response(generate_frame(frame_type),
            mimetype = "multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
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

    vs = cv2.VideoCapture(cv2.samples.findFileOrKeep(input))
    
    params = {
        "algo": algo,
        "disappear_threshold": disappear_threshold,
        "extractor": extractor,
        "contour_area": contour_area
    }
    t = threading.Thread(target=track_object, kwargs=params)
    t.daemon = True
    t.start()
    if not vs.isOpened():
        print('Unable to open: ' + input)
        exit(0)

    # start the flask app
    app.run(host="localhost", port=8989, debug=True,
        threaded=True, use_reloader=False)
