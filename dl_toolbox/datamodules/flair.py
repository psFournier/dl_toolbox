from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from pytorch_lightning import LightningDataModule
import dl_toolbox.datasets as datasets
from dl_toolbox.utils import CustomCollate
from pytorch_lightning.utilities import CombinedLoader

import os
import numpy as np
from os.path import join
from pathlib import Path
import json
import random
import re


def _gather_data(path_folders, path_metadata: str, use_metadata: bool, test_set: bool) -> dict:

    #### return data paths
    def get_data_paths (path, filter):
        for path in Path(path).rglob(filter):
             yield path.resolve().as_posix()        

    #### encode metadata
    def coordenc_opt(coords, enc_size=32) -> np.array:
        d = int(enc_size/2)
        d_i = np.arange(0, d / 2)
        freq = 1 / (10e7 ** (2 * d_i / d))

        x,y = coords[0]/10e7, coords[1]/10e7
        enc = np.zeros(d * 2)
        enc[0:d:2]    = np.sin(x * freq)
        enc[1:d:2]    = np.cos(x * freq)
        enc[d::2]     = np.sin(y * freq)
        enc[d + 1::2] = np.cos(y * freq)
        return list(enc)           

    def norm_alti(alti: int) -> float:
        min_alti = 0
        max_alti = 3164.9099121094
        return [(alti-min_alti) / (max_alti-min_alti)]        

    def format_cam(cam: str) -> np.array:
        return [[1,0] if 'UCE' in cam else [0,1]][0]

    def cyclical_enc_datetime(date: str, time: str) -> list:
        def norm(num: float) -> float:
            return (num-(-1))/(1-(-1))
        year, month, day = date.split('-')
        if year == '2018':   enc_y = [1,0,0,0]
        elif year == '2019': enc_y = [0,1,0,0]
        elif year == '2020': enc_y = [0,0,1,0]
        elif year == '2021': enc_y = [0,0,0,1]    
        sin_month = np.sin(2*np.pi*(int(month)-1/12)) ## months of year
        cos_month = np.cos(2*np.pi*(int(month)-1/12))    
        sin_day = np.sin(2*np.pi*(int(day)/31)) ## max days
        cos_day = np.cos(2*np.pi*(int(day)/31))     
        h,m=time.split('h')
        sec_day = int(h) * 3600 + int(m) * 60
        sin_time = np.sin(2*np.pi*(sec_day/86400)) ## total sec in day
        cos_time = np.cos(2*np.pi*(sec_day/86400))
        return enc_y+[norm(sin_month),norm(cos_month),norm(sin_day),norm(cos_day),norm(sin_time),norm(cos_time)]        


    data = {'IMG':[],'MSK':[],'MTD':[]}
    if path_folders:
        for domain in path_folders:
            list_img = sorted(list(get_data_paths(domain, 'IMG*.tif')), key=lambda x: int(x.split('_')[-2][1:]))
            #data['IMG'] += sorted(list(get_data_paths(domain, 'IMG*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))
            data['IMG'] += list_img
            if test_set == False:
                list_msk = sorted(list(get_data_paths(domain, 'MSK*.tif')), key=lambda x: int(x.split('_')[-2][1:]))
                #data['MSK'] += sorted(list(get_data_paths(domain, 'MSK*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))
                data['MSK'] += list_msk
            #print(f'domain {domain}: {[(img, msk) for img, msk in zip(list_img, list_msk)]}')
            #break

        if use_metadata == True:

            with open(path_metadata, 'r') as f:
                metadata_dict = json.load(f)              
            for img in data['IMG']:
                curr_img = img.split('/')[-1][:-4]
                enc_coords   = coordenc_opt([metadata_dict[curr_img]["patch_centroid_x"], metadata_dict[curr_img]["patch_centroid_y"]])
                enc_alti     = norm_alti(metadata_dict[curr_img]["patch_centroid_z"])
                enc_camera   = format_cam(metadata_dict[curr_img]['camera'])
                enc_temporal = cyclical_enc_datetime(metadata_dict[curr_img]['date'], metadata_dict[curr_img]['time'])
                mtd_enc = enc_coords+enc_alti+enc_camera+enc_temporal 
                data['MTD'].append(mtd_enc)

        if test_set == False:
            if len(data['IMG']) != len(data['MSK']): 
                print('[WARNING !!] UNMATCHING NUMBER OF IMAGES AND MASKS ! Please check load_data function for debugging.')
            if data['IMG'][0][-10:-4] != data['MSK'][0][-10:-4] or data['IMG'][-1][-10:-4] != data['MSK'][-1][-10:-4]: 
                print('[WARNING !!] UNSORTED IMAGES AND MASKS FOUND ! Please check load_data function for debugging.')                

    return data

class Flair(LightningDataModule):
    
    def __init__(
        self,
        #data_path,
        batch_size,
        crop_size,
        epoch_len,
        labels,
        workers,
        use_metadata,
        train_domains,
        val_domains,
        test_domains,
        unsup_train_idxs=None,
        img_aug=None,
        unsup_img_aug=None,
        *args,
        **kwargs
    ):

        super().__init__()
        self.batch_size = batch_size
        self.epoch_len = epoch_len
        self.crop_size = crop_size
        self.num_workers = workers
        #self.data_path = data_path
        
        dict_train = _gather_data(train_domains, path_metadata=None, use_metadata=use_metadata, test_set=False)
        dict_val = _gather_data(val_domains, path_metadata=None, use_metadata=use_metadata, test_set=False)
        dict_test = _gather_data(test_domains, path_metadata=None, use_metadata=use_metadata, test_set=True)
                
        self.train_set = datasets.Flair(
            dict_files=dict_train,
            labels=labels,
            crop_size=crop_size,
            use_metadata=False,
            img_aug=img_aug
        )
        
        self.unsup_train_set = None
        
        self.val_set = datasets.Flair(
            dict_files=dict_val,
            labels=labels,
            crop_size=512
        )
        
        self.test_set = datasets.Flair(
            dict_files=dict_test,
            labels=labels,
            crop_size=512
        )

        self.class_names = list(self.val_set.labels.keys())
        
    def train_dataloader(self):
        
        train_dataloaders = {}
        train_dataloaders['sup'] = DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            collate_fn=CustomCollate(),
            sampler=RandomSampler(
                data_source=self.train_set,
                replacement=True,
                num_samples=self.epoch_len
            ),
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        if self.unsup_train_set:
    
            train_dataloaders['unsup'] = DataLoader(
                dataset=self.unsup_train_set,
                batch_size=self.batch_size,
                sampler=RandomSampler(
                    data_source=self.train_set,
                    replacement=True,
                    num_samples=self.epoch_len
                ),
                collate_fn=CustomCollate(),
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True
            )

        return CombinedLoader(
            train_dataloaders,
            mode='min_size'
        )

    def val_dataloader(self):

        val_dataloader = DataLoader(
            dataset=self.val_set,
            shuffle=False,
            collate_fn=CustomCollate(),
            batch_size=8,
            num_workers=self.num_workers,
            pin_memory=True
        )
        self.nb_val_batch = len(self.val_set) // self.batch_size

        return val_dataloader
    
    def predict_dataloader(self):
        
        predict_dataloader = DataLoader(
            dataset=self.test_set,
            shuffle=False,
            batch_size=8,
            collate_fn=CustomCollate(),
            num_workers=self.num_workers
        )
        
        return predict_dataloader
    
class OCS_DataModule(LightningDataModule):

    def __init__(
        self,
        dict_train=None,
        dict_val=None,
        dict_test=None,
        num_workers=1,
        batch_size=2,
        drop_last=True,
        num_classes=13,
        num_channels=5,
        use_metadata=True,
        use_augmentations=True
    ):
        super().__init__()
        self.dict_train = dict_train
        self.dict_val = dict_val
        self.dict_test = dict_test
        self.batch_size = batch_size
        self.num_classes, self.num_channels = num_classes, num_channels
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.pred_dataset = None
        self.drop_last = drop_last
        self.use_metadata = use_metadata
        self.use_augmentations = use_augmentations

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage == "validate":
            self.train_dataset = Fit_Dataset(
                dict_files=self.dict_train,
                num_classes=self.num_classes,
                use_metadata=self.use_metadata,
                use_augmentations=self.use_augmentations
            )

            self.val_dataset = Fit_Dataset(
                dict_files=self.dict_val,
                num_classes=self.num_classes,
                use_metadata=self.use_metadata
            )

        elif stage == "predict":
            self.pred_dataset = Predict_Dataset(
                dict_files=self.dict_test,
                num_classes=self.num_classes,
                use_metadata=self.use_metadata
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )
    
    def predict_dataloader(self):
        return DataLoader(
            dataset=self.pred_dataset,
            batch_size=1, 
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )