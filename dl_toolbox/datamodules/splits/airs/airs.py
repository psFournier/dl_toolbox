import csv
import rasterio
from pathlib import Path

                
with open(Path.home() / f'ai4geo/splits/airs/airs.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    path = Path('/data/AIRS/trainval')
    writer.writerow(['dataset_cls',
                     'tile_id',
                     'img_path',
                     'label_path',
                     'x0',
                     'y0',
                     'patch_width',
                     'patch_height',
                     'fold_id'
                     ])
    
    i = 0
    for full_image_path in (path / 'val/image').iterdir():
        full_label_path = path / 'val/label' / full_image_path.name
        with rasterio.open(full_image_path) as f:
            height, width = f.shape
            if width == 10000 and height == 10000:
                m = f.dataset_mask().mean()
                if m < 250:
                    print(full_image_path)
                else:
                    writer.writerow(
                        [
                            'Airs',
                            i,
                            f'val/image/{full_image_path.name}',
                            f'val/label/{full_image_path.name}',
                            0,
                            0,
                            width,
                            height,
                            i % 5
                        ]
                    )
                    i += 1
        
    for full_image_path in (path / 'train/image').iterdir():
        full_label_path = path / 'train/label' / full_image_path.name
        with rasterio.open(full_image_path) as f:
            height, width = f.shape
            if width == 10000 and height == 10000:
                m = f.dataset_mask().mean()
                if m < 250:
                    print(full_image_path)
                else:
                    writer.writerow(
                        [
                            'Airs',
                            i,
                            f'train/image/{full_image_path.name}',
                            f'train/label/{full_image_path.name}',
                            0,
                            0,
                            width,
                            height,
                            i % 10 + 5
                        ]
                    )
                    i += 1
