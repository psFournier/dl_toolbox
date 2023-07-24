from PIL import Image
from torchvision.datasets import Cityscapes
from torchvision.transforms.functional import pil_to_tensor


class Cityscapes(Cityscapes):
    # https://pytorch.org/vision/0.15/_modules/torchvision/datasets/cityscapes.html#Cityscapes

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise, target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert("RGB")
        image = pil_to_tensor(image)

        targets = []
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
                target = pil_to_tensor(target).squeeze()

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return {
            "image": image,
            "label": target,
            "path": self.images[index],
        }
