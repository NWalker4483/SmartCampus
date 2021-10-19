from torchreid.utils import FeatureExtractor

extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='/Users/nilez/Jobs/Morgan/SmartCampus/raw_data/models/frozen/osnet_x1_0_imagenet.pth',
    device='cpu'
)

image_list = [
    'test.jpeg',
]

features = extractor(image_list)
print(features.shape) # output (5, 512)