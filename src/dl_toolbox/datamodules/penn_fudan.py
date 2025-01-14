from dl_toolbox.datamodules import Base
from dl_toolbox.utils import label
from dl_toolbox.utils import list_of_dicts_to_dict_of_lists

class PennFudan(Base):
    def __init__(
        self,
        *args,
        **kwargs
    ):    
        super().__init__(*args, **kwargs)

    def setup(self, stage):
        dataset = PennFudanDataset('/data/PennFudanPed', self.train_tf)
        dataset_test = PennFudanDataset('/data/PennFudanPed', self.val_tf)
        # split the dataset in train and test set
        indices = torch.randperm(len(dataset)).tolist()
        self.train_set = torch.utils.data.Subset(dataset, indices[:-50])
        self.val_set = torch.utils.data.Subset(dataset_test, indices[-50:])
        
    def collate(self, batch, train):
        batch = list_of_dicts_to_dict_of_lists(batch)
        batch['image'] = torch.stack(batch['image'])
        #batch['target'] = torch.stack(batch['target'])
        return batch