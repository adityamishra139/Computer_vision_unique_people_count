import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import cv2
import numpy as np
from torchreid.utils import FeatureExtractor
extractor = FeatureExtractor(model_name='osnet_x1_0', device='cpu')
img = np.zeros((256, 128, 3), dtype=np.uint8)
feat = extractor([img])[0].numpy()
print(feat.shape)
