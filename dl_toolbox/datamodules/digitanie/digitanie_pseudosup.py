import pandas as pd
from .digitanie import Digitanie
import dl_toolbox.datasets as datasets
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader, RandomSampler
from dl_toolbox.utils import CustomCollate
from pathlib import Path

class DigitaniePseudosup(Digitanie):

    def __init__(
        self,
        pl_dir,
        thresh,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pl_dir = Path(pl_dir)
        stats = pd.read_csv(self.pl_dir/'stats.csv', index_col=0)
        stats.sort_values('avg_cert', ascending=False, inplace=True)
        self.stats = stats[:thresh]

    def prepare_data(self):
        super().prepare_data()  
        self.dict_pl = {
            "IMG": [self.data_path/img for img in list(self.stats['img'])],
            "MSK": [self.pl_dir/img for img in list(self.stats['msk'])],
            "WIN": [tuple(int(i) for i in win.split('_')) for win in list(self.stats['win'])]
        }

    def setup(self, stage):
        super().setup(stage)
        self.pl_set = datasets.DigitaniePseudosup(
            self.dict_pl["IMG"],
            self.dict_pl["MSK"],
            self.dict_pl["WIN"],
            self.bands,
            self.merge,
            transforms=self.get_tf(self.train_tf, self.city)
        )
        
    def train_dataloader(self):
        train_dataloaders = {}
        train_dataloaders["sup"] = self.dataloader(self.train_s_set)(
            sampler=RandomSampler(
                self.train_s_set,
                replacement=True,
                num_samples=self.steps_per_epoch*self.batch_size_s
            ),
            drop_last=True,
            batch_size=self.batch_size_s
        )
        train_dataloaders["pseudosup"] = self.dataloader(self.pl_set)(
            sampler=RandomSampler(
                self.pl_set,
                replacement=True,
                num_samples=self.steps_per_epoch*self.batch_size_u
            ),
            drop_last=True,
            batch_size=self.batch_size_u
        )
        return CombinedLoader(train_dataloaders, mode="max_size_cycle")