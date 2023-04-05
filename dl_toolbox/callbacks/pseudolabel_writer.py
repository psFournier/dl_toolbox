import os
from pathlib import Path


class PseudolabelWriter(BasePredictionWriter):

    def __init__(
        self,
        out_path,
        write_interval,
        cls_names
    ):
        super().__init__(write_interval)
        
        self.out_path = Path(out_path)
        self.out_path.mkdir(exist_ok=True, parents=True)
        self.cls_names = cls_names
        
        self.counts = [0] * len(cls_names)

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        
        logits = outputs.cpu()  # Move predictions to CPU    
        probas = pl_module.logits2probas(logits)
        confs, preds = pl_module.probas2confpreds(probas)

        for i, pred in enumerate(preds):
            if confs[i] > 0.9:
                pred = int(pred)
                cls_name = self.cls_names[pred]
                num = self.counts[pred]
                class_dir = self.out_path / cls_name
                class_dir.mkdir(parents=True, exist_ok=True)
                dst = class_dir / f'{cls_name}_{num:04}.jpg'
                os.symlink(
                    batch['path'][i],
                    dst
                )
                self.counts[pred] += 1