import csv
import imagesize
from pathlib import Path, PurePath


with open(Path.home() / f'ai4geo/splits/digitanieV2/sc1b.csv', 'w+', newline='') as csvfile:
    
    writer = csv.writer(csvfile)
    writer.writerow(
        [
            'city',
            'tile_id',
            'img_path',
            'label_path',
            'x0',
            'y0',
            'patch_width',
            'patch_height',
            'fold_id',
            'origin_tile'
        ]
    )
    
    digitanie = Path('/work/OT/ai4geo/DATA/DATASETS/DIGITANIE')
    
    for city, reproj, origin_tile in [
        ('Arcachon', 32630, 'ARCACHON_20180821_T_TOA_reproj-EPSG:32630.tif'),
        ('Biarritz', 32630, 'BIARRITZ_20140902_T_TOA_reproj-EPSG:32630.tif'),
        ('Brisbane', 32756, 'BRISBANE_20180316_T_TOA_reproj-EPSG:32756.tif'),
        ('Can-Tho', 32648, 'Can-Tho_EPSG32648_'),
        ('Helsinki', 32635, 'HELSINKI_20180222_T_TOA_reproj-EPSG:32635.tif'),
        ('Maros', 32750, 'MAROS_20181005_T_TOA_reproj-EPSG:32750.tif'),
        ('Montpellier', 2154, 'MONTPELLIER_20190912_T_TOA_reproj-EPSG:2154.tif'),
        ('Munich', 32632, 'MUNICH_20181012_T_TOA_reproj-EPSG:32632.tif'),
        ('Nantes', 32630, 'NANTES_20131220_T_TOA_reproj-EPSG:32630.tif'),
        ('Paris', 2154, 'PARIS_20180326_T_TOA_reproj-EPSG:2154.tif'),
        ('Shanghai', 32651, 'SHANGHAI_20170913_T_TOA_reproj-EPSG:32651.tif'),
        ('Strasbourg', 32632, 'STRASBOURG_20180620_T_TOA_reproj-EPSG:32632.tif'),
        ('Tianjin', 32650, 'TIANJIN_20170929_T_TOA_reproj-EPSG:32650.tif')        
    ]:
        
        city_path = PurePath(city)
        
        for i in range(0, 10):
            
            if city=='Tianjin':
                img_path = city_path / Path(city+f'_{reproj}_{i}.tif')
            else:
                img_path = city_path / Path(city+f'_EPSG{reproj}_{i}.tif')

            label_path = city_path / Path(f'COS9/{city}_{i}-v4.tif')
            width, height = imagesize.get(digitanie / label_path)
            writer.writerow(
                [
                    f'DigitanieV2',
                    i,
                    img_path,
                    label_path,
                    0,
                    0,
                    width,
                    height,
                    i,
                    city_path / origin_tile
                ]
            )
