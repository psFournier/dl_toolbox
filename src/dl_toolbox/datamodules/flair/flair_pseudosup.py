import pandas as pd
from .flair import Flair
import dl_toolbox.datasets as datasets
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader
from dl_toolbox.utils import CustomCollate
from pathlib import Path

class FlairPseudosup(Flair):

    def __init__(
        self,
        pl_dir,
        thresh,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pl_dir = Path(pl_dir)
        self.stats = pd.read_csv(self.pl_dir/'stats.csv', index_col=0)
        self.stats.sort_values('avg_cert', ascending=False, inplace=True)
        self.thresh = thresh

    def prepare_data(self):
        super().prepare_data()  
        top_pl_img = list(self.stats.index[:self.thresh])
        self.dict_pl = {
            "IMG": [self.data_path/img for img in top_pl_img],
            "MSK": [self.pl_dir/img for img in top_pl_img]
        }

    def setup(self, stage):
        super().setup(stage)
        if stage in ("fit"):
            self.pl_set = datasets.Flair(
                self.dict_pl["IMG"],
                self.dict_pl["MSK"],
                self.bands,
                self.merge,
                transforms=self.dataset_tf,
            )
        
    def train_dataloader(self):
        train_dataloaders = {}
        train_dataloaders["sup"] = DataLoader(
            dataset=self.train_set,
            collate_fn=CustomCollate(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True
        )
        train_dataloaders["pseudosup"] = DataLoader(
            dataset=self.pl_set,
            collate_fn=CustomCollate(),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True
        )
        return CombinedLoader(train_dataloaders, mode="max_size_cycle")