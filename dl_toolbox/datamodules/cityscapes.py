class Cityscapes(LightningDataModule):
    
    def __init__(
        self,
        merge,
        bands,
        crop_size,
        data_path,
        csv_path,
        csv_name,
        train_tf,
        val_tf,
        epoch_len,
        batch_size,
        num_workers,
        pin_memory
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_path = data_path
        #self.dims = (3, 1024, 2048)
        self.quality_mode = 'fine'
        self.target_type = 'semantic'
        
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.target_transforms = target_transforms
        
        self.train_tf = train_tf
        self.val_tf = val_tf            
        
    def setup(self, stage):
        
        self.train_set = Cityscapes(
            self.data_dir,
            split="train",
            target_type=self.target_type,
            mode=self.quality_mode,
            transforms=transforms,
            target_transform=target_transforms,
        )
        
        self.val_set = Cityscapes(
            self.data_dir,
            split="val",
            target_type=self.target_type,
            mode=self.quality_mode,
            transform=transforms,
            target_transform=target_transforms,
        )
        
    def train_dataloader(self) -> DataLoader:

        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:

        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )