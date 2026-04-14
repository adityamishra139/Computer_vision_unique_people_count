import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
try:
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    print("DINO LOADED")
except Exception as e:
    print(f"FAILED: {e}")
