from torch.utils.data._utils.collate import default_collate
import torch


class CustomCollate():

    def __call__(self, batch, *args, **kwargs):

        windows = [elem['window'] for elem in batch if 'window' in elem.keys()]
        paths = [elem['path'] for elem in batch if 'path' in elem.keys()]
        keys_to_collate = ['image', 'orig_image', 'mask', 'orig_mask']
        to_collate = [{k: v for k, v in elem.items() if (k in keys_to_collate) and (v is not None)} for elem in batch]
        batch = default_collate(to_collate)
        if 'mask' not in batch.keys():
            batch['mask'] = None
        
        batch['window'] = windows
        batch['path'] = paths

        return batch
