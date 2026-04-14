import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torchreid
from torchreid.utils import FeatureExtractor

try:
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='',
        device='cpu' # fallback to default
    )
    print("TORCHREID OSNET LOADED SUCCESSFULLY!")
except Exception as e:
    print(f"FAILED: {e}")
