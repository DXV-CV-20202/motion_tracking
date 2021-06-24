# Requirements
1. Installed all of Python modules in file `requirements.txt`. You may use below command to install these modules:
```
    pip install -r requirements.txt
```
# Script
- Running `python main.py` to detect and tracking objects from input video. Paramets;
    + --input: input video
    + --algo: background subtraction algorithm ("MOG2" or "KNN")
    + --disappear_threshold: number of frames for reversing object before permanently delete
    + --num_keypoints: number of SIFT keypoints for object matching
    + --extractor_name: matching algorithm ("SIFT" or "ColorHistogram")
    + --contour_area: minimize area of contour that correspond to an object
- Running `python app.py` with similar parameters to open Web UI (unstable).