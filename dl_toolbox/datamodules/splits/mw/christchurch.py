import csv
import imagesize
from pathlib import Path

data = Path.home() / Path('ai4geo/data/miniworld_tif')
csv_dst = Path.home() / 'ai4geo/splits/mw/christchurch.csv'
csv_src = Path.home() / 'ai4geo/splits/split_christchurch_filtered.csv'

with open(csv_src, 'r') as src:
    
    reader = csv.reader(src)
    next(reader)
    
    with open(csv_dst, 'w', newline='') as dst:
        
        writer = csv.writer(dst)
        writer.writerow([
            'dataset_cls',
            'tile_id',
            'img_path',
            'label_path',
            'x0',
            'y0',
            'patch_width',
            'patch_height',
            'fold_id'
        ])
        
        for i, row in enumerate(reader):
            
            _, _, image_path, label_path, x0, y0, w, h, _ = row[:9]
            p_img = Path('christchurch') / image_path
            p_lbl = Path('christchurch') / label_path
            full_img_path = data / p_img          
            width, height = imagesize.get(full_img_path)
            if width == 1500 and height == 1500:
                writer.writerow(
                    [
                        'Miniworld',
                        i,
                        p_img,
                        p_lbl,
                        0,
                        0,
                        width,
                        height,
                        i % 20
                    ]
                )