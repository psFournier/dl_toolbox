from torch_datasets import MiniworldCity
from torch_datasets import BaseDatasetLabeled, BaseDatasetUnlabeled


class Tyrol(MiniworldCity):

    labeled_image_paths = ['test/{}_x.png'.format(i) for i in range(15)] + \
                          ['train/{}_x.png'.format(i) for i in range(20)]

    label_paths = ['test/{}_y.png'.format(i) for i in range(15)] + \
                  ['train/{}_y.png'.format(i) for i in range(20)]

    unlabeled_image_paths = []

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    @property
    def __image_size__(cls):

        return 3000*3000

class TyrolLabeled(Tyrol, BaseDatasetLabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

class TyrolUnlabeled(Tyrol, BaseDatasetUnlabeled):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)