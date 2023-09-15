import hydra
import torch
from dl_toolbox.inference import test 

torch.set_float32_matmul_precision('high')

@hydra.main(version_base="1.3", config_path="../../../configs", config_name="test.yaml")
def main(cfg: DictConfig) -> None:
    datamodule = datamodules.Resisc(
        data_path=Path('/data'),
        merge='all45',
        sup=10,
        unsup=0,
        dataset_tf=tf.StretchToMinmax([0]*3, [255.]*3),
        batch_size=4,
        num_workers=4,
        pin_memory=True,
        class_weights=None
    )
    dataloader = datamodule.val_dataloaders()
    test(cfg, dataloader)

if __name__ == "__main__":
    main()