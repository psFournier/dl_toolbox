import torchvision.utils as tv_utils
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
import torch
import matplotlib.patches as mpatches 

def show_classifications(imgs, classifications, class_list, targets=None):
    n = len(imgs)
    if targets is None:
        targets = classifications
    fig, axs = plt.subplots(nrows=1, ncols=n, squeeze=False)
    for i, (img, classif, tgt) in enumerate(zip(imgs, classifications, targets)):
        label_classif = class_list[classif].name
        label_target = class_list[tgt].name
        axs[0,i].imshow(img.permute(1, 2, 0).numpy())
        axs[0,i].set_title(f'classif: {label_classif}, label: {label_target}')
        axs[0,i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.tight_layout()
    return fig
    

def show_detections(imgs, detections, class_list):
    n = len(imgs)
    fig, axs = plt.subplots(nrows=1, ncols=n, squeeze=False)
    for i, (img, detection) in enumerate(zip(imgs, detections)):
        tags = [class_list[label-1].name for label in detection['labels']]
        tags_colors = [class_list[label-1].color for label in detection['labels']]
        bboxes = v2.functional.convert_bounding_box_format(detection['boxes'],  new_format='XYXY')
        img = tv_utils.draw_bounding_boxes(
            img, 
            bboxes,
            tags,
            colors=tags_colors,
        )
        axs[0,i].imshow(img.permute(1, 2, 0).numpy())
        axs[0,i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.tight_layout()
    return fig
        
def show_segmentations(imgs, segmentations, class_list, alpha):
    n = len(imgs)
    fig, axs = plt.subplots(ncols=n, squeeze=False)
    colors = [l.color for l in class_list]
    for i, (img, segmentation) in enumerate(zip(imgs, segmentations)):
        one_hot = torch.nn.functional.one_hot(segmentation, len(class_list))
        one_hot = torch.movedim(one_hot,-1,0)
        img = tv_utils.draw_segmentation_masks(
            img,
            one_hot.to(torch.bool),
            colors=colors,
            alpha=alpha
        )
        axs[0,i].imshow(img.permute(1, 2, 0).numpy())
        axs[0,i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    # Creating legend with color box 
    color_legends = [mpatches.Patch(color=tuple(c/255. for c in l.color), label=l.name) for l in class_list]
    fig.legend(handles=color_legends, loc='center right') 
    fig.tight_layout()
    fig.subplots_adjust(right=0.75)
    return fig
    
    
    