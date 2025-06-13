import sys
# Add path to root utils/python/ directory to find config scripts
sys.path.insert(0, "../../../utils/python/")

from config import Empty, ObjectInSpace
from box_channel_config import BoxChannelConfig

if __name__ == "__main__":
    comp_list = {'empty': Empty(),
                 'circle': ObjectInSpace('square-circle'),
                 'square': ObjectInSpace('square-square'),
                 'triangle': ObjectInSpace('square-triangle'),
                 'star': ObjectInSpace('square-star'),}

    example = BoxChannelConfig(2,2)
    for name, comp in comp_list.items():
        example.addComponent(comp)

    example.GenerateAllConfigs(0)

    test = BoxChannelConfig(8,8)
    for name, comp in comp_list.items():
        test.addComponent(comp)
    test.CreateRandomConfig('test.box-channel.8x8.h5')
