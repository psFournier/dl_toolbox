from .flair import DatamoduleFlair2
from .utils import flair_gather_data
import pandas as pd
from dl_toolbox.datasets import DatasetFlair2
from pytorch_lightning.utilities import CombinedLoader
from pathlib import Path

class Flair2Pseudosup(DatamoduleFlair2):

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
            self.pl_set = DatasetFlair2(
                self.dict_pl["IMG"],
                self.dict_pl["MSK"],
                self.bands,
                self.merge,
                transforms=self.train_tf,
            )
        
    def train_dataloader(self):
        train_dataloaders = {}
        train_dataloaders["sup"] = self.get_loader(self.train_set)(
            shuffle=True,
            drop_last=True,
        )
        train_dataloaders["pseudosup"] = self.get_loader(self.pl_set)(
            shuffle=True,
            drop_last=True
        )
        return CombinedLoader(train_dataloaders, mode="max_size_cycle")