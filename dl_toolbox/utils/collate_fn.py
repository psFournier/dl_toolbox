import torch
from torch.utils.data._utils.collate import default_collate


class CustomCollate:
    def __call__(self, batch, *args, **kwargs):
        keys_to_collate = ["image", "label"]
        to_collate = [
            {
                k: v
                for k, v in elem.items()
                if (k in keys_to_collate) and (v is not None)
            }
            for elem in batch
        ]
        collated_batch = default_collate(to_collate)

        if "label" not in collated_batch.keys():
            collated_batch["label"] = None

        for aux in ["image_path", "window", "label_path"]:
            collated_batch[aux] = [elem[aux] for elem in batch if aux in elem.keys()]

        return collated_batch
