from torch.utils.data._utils.collate import default_collate
import torch


class CustomCollate():

    def __call__(self, batch, *args, **kwargs):
        
        crops = [elem['crop'] for elem in batch if 'crop' in elem.keys()]
        #crops_tf = [elem['crop_tf'] for elem in batch if 'crop_tf' in elem.keys()]
        paths = [elem['path'] for elem in batch if 'path' in elem.keys()]
        
        keys_to_collate = ['image', 'label']
        to_collate = [{k: v for k, v in elem.items() if (k in keys_to_collate) and (v is not None)} for elem in batch]
        batch = default_collate(to_collate)
        
        if 'label' not in batch.keys():
            batch['label'] = None
        batch['crop'] = crops
        #batch['crop_tf'] = crops_tf
        batch['path'] = paths

        return batch
