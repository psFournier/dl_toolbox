import torchvision.utils as tv_utils
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt


def show_detections(imgs, detections, classes):
    n = len(imgs)
    _, axs = plt.subplots(nrows=1, ncols=n, squeeze=False)
    for i, (img, detection) in enumerate(zip(imgs, detections)):
        labels = [classes[label-1].name for label in detection['labels']]
        bboxes = v2.functional.convert_bounding_box_format(detection['boxes'],  new_format='XYXY')
        img = tv_utils.draw_bounding_boxes(
            img, 
            bboxes,
            labels,
            colors=(0,255,255),
        )
        axs[0,i].imshow(img.permute(1, 2, 0).numpy())
        axs[0,i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.tight_layout()
    
    
    