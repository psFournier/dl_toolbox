import torchvision.transforms.v2 as v2
from torchvision import tv_tensors


class BatchMultiCrop(v2.Transform):
    def forward(self, sample):
        images_or_videos, labels = sample
        batch_size = len(images_or_videos)
        image_or_video = images_or_videos[0]
        images_or_videos = tv_tensors.wrap(torch.stack(images_or_videos), like=image_or_video)
        labels = torch.full((batch_size,), label, device=images_or_videos.device)
        return images_or_videos, labels
