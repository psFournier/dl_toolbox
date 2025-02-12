# Third-party libraries
import pytorch_lightning as pl
import torch
import torchvision

from torchvision.utils import draw_bounding_boxes as draw_bb
from torchvision.ops import box_convert

def process_detections(detections, color_table, names, with_scores):
    batch_boxes = [box_convert(det['boxes'].detach().cpu(), 'xywh', 'xyxy') for det in detections]
    batch_labels = [det['labels'].detach().cpu()-1 for det in detections]
    batch_colors = [[color_table[l][1] for l in label] for label in batch_labels]
    if with_scores:
        batch_scores = [det['scores'].detach().cpu() for det in detections]
        batch_legends = [[f"{names[l]}: {s:.2f}" for l,s in p] for p in zip(batch_labels, batch_scores)]
    else:
        batch_legends = [[f"{names[l]}: 1." for l in p] for p in batch_labels]
    return batch_boxes, batch_legends, batch_colors
    
def display_det_batch(trainer, module, batch, prefix):
    x, tgt, p = batch
    nb = min(4, x.shape[0])
    imgs = list(x.detach().cpu()[:nb])
    mins = [i.view(3, -1).min(1)[0].view(3, 1, 1) for i in imgs]
    maxs = [i.view(3, -1).max(1)[0].view(3, 1, 1) for i in imgs]
    unnorm_imgs = [(i-m)/(M-m) for i,m,M in zip(imgs, mins, maxs)]
    int_imgs = [img.mul(255.).to(torch.uint8) for img in unnorm_imgs]
    targets = tgt[:nb]
    outputs = module.forward(x)
    outputs.update({k: v.detach().cpu() for k, v in outputs.items()})
    preds = module.post_process(outputs, x)
    #preds = [{**t, **{'scores': 1.}} for t in targets] #to change
    colors = trainer.datamodule.class_colors
    names = trainer.datamodule.class_names
    processed_preds = zip(int_imgs, *process_detections(preds, colors, names, False))
    processed_targets = zip(int_imgs, *process_detections(targets, colors, names, False))
    params = {'font': '/usr/share/fonts/truetype/ubuntu/Ubuntu-L.ttf', 'font_size': 10}
    drawn_preds = [draw_bb(img, bb, l, c, **params) for img, bb, l, c in processed_preds]
    drawn_targets = [draw_bb(img, bb, l, c, **params) for img, bb, l, c in processed_targets]
    grid_preds = torchvision.utils.make_grid(torch.stack(drawn_preds))
    grid_targets = torchvision.utils.make_grid(torch.stack(drawn_targets))
    grid = torch.cat([grid_targets, grid_preds], dim=1)
    step = trainer.global_step
    trainer.logger.experiment.add_image(f"{prefix} images", grid, step)

class DetectionImagesVisualisation(pl.Callback):

    def __init__(self, freq, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.freq>0 and trainer.current_epoch % self.freq == 0 and batch_idx <= 1:
            display_det_batch(trainer, pl_module, batch["sup"], "Train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.freq>0 and trainer.current_epoch % self.freq == 0 and batch_idx <= 1:
            display_det_batch(trainer, pl_module, batch, "Val")
            
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.freq>0 and batch_idx <= 1:
            display_det_batch(trainer, pl_module, batch, "Test")
