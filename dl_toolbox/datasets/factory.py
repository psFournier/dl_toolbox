import rasterio
import csv
import numpy as np
import ast 

from dl_toolbox.datasets import *
import rasterio.windows as windows


datasets = {
    #'Resisc': Resisc,
    'Digitanie': Digitanie,
    #'DigitanieToulouse': DigitanieToulouseDs,
    #'DigitanieBiarritz': DigitanieBiarritzDs,
    #'DigitanieMontpellier': DigitanieMontpellierDs,
    #'DigitanieParis': DigitanieParisDs,
    #'DigitanieStrasbourg': DigitanieStrasbourgDs,
    'Semcity': Semcity,
    'Airs': Airs,
    #'Miniworld': Miniworld,
    #'Inria': Inria
}

class DatasetFactory:

    @staticmethod
    def create(name):
        return datasets[name]

def datasets_from_csv(data_path, split_path, folds):
    
    dataset_factory = DatasetFactory()
    
    with open(split_path, newline='') as splitfile:
        reader = csv.reader(splitfile)
        next(reader)
        for row in reader:
            name, _, image_path, label_path, x0, y0, w, h, fold, mins, maxs = row[:11]
            if int(fold) in folds:
                co, ro, w, h = [int(e) for e in [x0, y0, w, h]]
                if w==0 or h==0:
                    data_src = dataset_factory.create(name)(
                        image_path=data_path/image_path,
                        label_path=data_path/label_path if label_path != 'none' else None,
                        mins=np.array(ast.literal_eval(mins)).reshape(-1, 1, 1),
                        maxs=np.array(ast.literal_eval(maxs)).reshape(-1, 1, 1)
                    )
                else:
                    data_src = dataset_factory.create(name)(
                        image_path=data_path/image_path,
                        label_path=data_path/label_path if label_path != 'none' else None,
                        mins=np.array(ast.literal_eval(mins)).reshape(-1, 1, 1),
                        maxs=np.array(ast.literal_eval(maxs)).reshape(-1, 1, 1),
                        zone=windows.Window(co, ro, w, h)
                    ) 
                yield data_src