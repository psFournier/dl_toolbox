import rasterio
import csv
import numpy as np
import ast
import rasterio.windows as windows


def data_src_from_csv(src_class, data_path, csv_path, folds):
    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            _, _, image_path, label_path, x0, y0, w, h, fold, mins, maxs = row[:11]
            if int(fold) in folds:
                co, ro, w, h = [int(e) for e in [x0, y0, w, h]]
                yield src_class(
                    image_path=data_path / image_path,
                    label_path=data_path / label_path if label_path != "none" else None,
                    mins=np.array(ast.literal_eval(mins)).reshape(-1, 1, 1),
                    maxs=np.array(ast.literal_eval(maxs)).reshape(-1, 1, 1),
                    zone=windows.Window(co, ro, w, h),
                )
