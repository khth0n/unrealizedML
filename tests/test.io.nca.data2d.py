from unrealizedML.io.nca.data2d import TrainingData2d, pickle_dir

from pathlib import Path

'''
pickle_dir(
    Path('test_data/voronoi_ed/raw/input'),
    32,
    Path('test_data/voronoi_ed/processed/input')
)

pickle_dir(
    Path('test_data/voronoi_ed/raw/target'),
    32,
    Path('test_data/voronoi_ed/processed/target')
)
'''

dataset = TrainingData2d(
    Path('test_data/voronoi_ed/processed/input'),
    Path('test_data/voronoi_ed/processed/target')
)

print(dataset[0][0].shape)

