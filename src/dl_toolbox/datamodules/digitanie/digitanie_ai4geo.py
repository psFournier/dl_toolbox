
from pathlib import Path
from functools import partial

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset

import dl_toolbox.datasets as datasets
from dl_toolbox.utils import CustomCollate, get_tiles
from dl_toolbox.transforms import Compose


class DigitanieAi4geo(LightningDataModule):
    
    cities = {
        'ABU-DHABI': '4_2',
        'CAN-THO': '6_2',
        'HELSINKI': '8_1',
        'MAROS': '0_2',
        'ARCACHON': '0_1',
        #'PARIS': '5_3',
        'SAN-FRANCISCO': '7_9',
        'SHANGHAI': '8_7',
        'MONTPELLIER': '2_0',
        'TOULOUSE': '5_2',
        'PARIS': '15_10',
        'NEW-YORK': '7_3',
        'NANTES': '2_9',
        'TIANJIN': '4_8',
        'STRASBOURG': '1_9',
        'BIARRITZ': '5_7',
        'BRISBANE': '9_8',
        'BUENOS-AIRES': '0_5',
        'LAGOS': '9_4',
        'LE-CAIRE': '2_6',
        'MUNICH': '5_1',
        'PORT-ELISABETH': '6_9',
        'RIO-JANEIRO': '8_9'
    }

    def __init__(
        self,
        data_path,
        merge,
        sup,
        unsup,
        bands,
        to_0_1,
        train_tf,
        test_tf,
        batch_size_s,
        batch_size_u,
        steps_per_epoch,
        num_workers,
        pin_memory,
        *args,
        **kwargs
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.merge = merge
        self.sup = sup #unused
        self.unsup = unsup #unused
        self.bands = bands
        self.to_0_1 = to_0_1
        self.train_tf = train_tf
        self.test_tf = test_tf
        self.batch_size_s = batch_size_s
        self.batch_size_u = batch_size_u
        self.steps_per_epoch = steps_per_epoch
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.in_channels = len(self.bands)
        self.classes = datasets.Digitanie.classes[merge].value
        self.num_classes = len(self.classes)
        self.class_names = [l.name for l in self.classes]
        self.class_colors = [(i, l.color) for i, l in enumerate(self.classes)]
        self.npy_stats = self.data_path/'DIGITANIE_v4/normalisation_stats.npy'

    def get_tf(self, tf, city):
        if isinstance(self.to_0_1, partial):
            return Compose([self.to_0_1(npy=self.npy_stats, city=city), tf])
        else:
            return Compose([self.to_0_1, tf])    
        
    def prepare_data(self):
        self.dicts = {}
        for city, val_test in self.cities.items():
            dict_train = {'IMG':[], 'MSK':[], "WIN":[]}
            dict_val = {'IMG':[], 'MSK':[], "WIN": []}
            dict_test = {'IMG':[], 'MSK':[], "WIN": []}
            dict_predict = {'IMG':[], 'MSK':[], "WIN": []}
            citypath = self.data_path/f'DIGITANIE_v4/{city}'
            val_idx, test_idx = map(int, val_test.split('_'))
            imgs = list(citypath.glob('*16bits_COG_*.tif'))
            imgs = sorted(imgs, key=lambda x: int(x.stem.split('_')[-1]))
            msks = list(citypath.glob('COS43/*[0-9].tif'))
            msks = sorted(msks, key=lambda x: int(x.stem.split('_')[-1]))
            for i, (img, msk) in enumerate(zip(imgs,msks)):
                if i==val_idx:
                    for win in get_tiles(2048, 2048, 512):
                        dict_val['IMG'].append(img)
                        dict_val['MSK'].append(msk)
                        dict_val['WIN'].append(win)  
                elif i==test_idx:
                    for win in get_tiles(2048, 2048, 512):
                        dict_test['IMG'].append(img)
                        dict_test['MSK'].append(msk)
                        dict_test['WIN'].append(win)  
                        dict_predict['IMG'].append(img)
                        dict_predict['MSK'].append(msk)
                        dict_predict['WIN'].append(win)  
                else:
                    for win in get_tiles(2048, 2048, 512):
                        dict_train['IMG'].append(img)
                        dict_train['MSK'].append(msk)
                        dict_train['WIN'].append(win)
            self.dicts[city] = {'train': dict_train, 'val': dict_val, 'test': dict_test, 'predict': dict_predict}
        
    def setup(self, stage):
        self.train_set = ConcatDataset([
            datasets.Digitanie(
                self.dicts[city]['train']["IMG"],
                self.dicts[city]['train']["MSK"],
                self.dicts[city]['train']["WIN"],
                self.bands,
                self.merge,
                self.get_tf(self.train_tf, city)
            ) for city in self.cities.keys()
        ])
        self.val_set = ConcatDataset([
            datasets.Digitanie(
                self.dicts[city]['val']["IMG"],
                self.dicts[city]['val']["MSK"],
                self.dicts[city]['val']["WIN"],
                self.bands,
                self.merge,
                self.get_tf(self.test_tf, city)
            ) for city in self.cities.keys()
        ])
        self.test_set = ConcatDataset([
            datasets.Digitanie(
                self.dicts[city]['test']["IMG"],
                self.dicts[city]['test']["MSK"],
                self.dicts[city]['test']["WIN"],
                self.bands,
                self.merge,
                self.get_tf(self.test_tf, city)
            ) for city in self.cities.keys()
        ])
        self.predict_set = ConcatDataset([
            datasets.Digitanie(
                self.dicts[city]['predict']["IMG"],
                self.dicts[city]['predict']["MSK"],
                self.dicts[city]['predict']["WIN"],
                self.bands,
                self.merge,
                self.get_tf(self.test_tf, city)
            ) for city in self.cities.keys()
        ])


    def dataloader(self, dataset):
        return partial(
            DataLoader,
            dataset=dataset,
            collate_fn=CustomCollate(),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
                       
    def train_dataloader(self):
        train_dataloaders = {}
        train_dataloaders["sup"] = self.dataloader(self.train_set)(
            sampler=RandomSampler(
                self.train_set,
                replacement=True,
                num_samples=self.steps_per_epoch*self.batch_size_s
            ),
            drop_last=True,
            batch_size=self.batch_size_s
        )
        return CombinedLoader(train_dataloaders, mode="max_size_cycle")
    
    def val_dataloader(self):
        return self.dataloader(self.val_set)(
            shuffle=False,
            drop_last=False,
            batch_size=self.batch_size_s
        )
    
    def test_dataloader(self):
        return self.dataloader(self.test_set)(
            shuffle=False,
            drop_last=False,
            batch_size=self.batch_size_s
        )
    
    def predict_dataloader(self):
        return self.dataloader(self.predict_set)(
            shuffle=False,
            drop_last=False,
            batch_size=self.batch_size_s
        )