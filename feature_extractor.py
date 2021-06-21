import numpy as np
import cv2

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
        self.isRootSIFT = False
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
        kp, descriptor = self.extractor.detectAndCompute(image, None)
        if self.isRootSIFT == True:
            descriptor /= (descriptor.sum(axis=1, keepdims=True) + self.eps)
            descriptor = np.sqrt(descriptor)
        return kp, descriptor