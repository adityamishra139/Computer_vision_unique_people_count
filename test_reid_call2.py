import torchreid
print(dir(torchreid))
try:
    from torchreid.utils import FeatureExtractor
    print("Found in utils")
except:
    try:
        from torchreid.reid.utils import FeatureExtractor
        print("Found in reid.utils")
    except Exception as e:
        print(f"Error: {e}")
