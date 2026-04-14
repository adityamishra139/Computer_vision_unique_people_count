import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from torchreid.reid.utils import FeatureExtractor
import numpy as np

extractor = FeatureExtractor(model_name='osnet_x1_0', device='cpu')
img = np.zeros((256, 128, 3), dtype=np.uint8)
feat = extractor(img)
print(feat.shape)
