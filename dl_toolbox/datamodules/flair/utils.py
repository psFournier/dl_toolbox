import numpy as np
from pathlib import Path
import json

#### encode metadata
def coordenc_opt(coords, enc_size=32) -> np.array:
    d = int(enc_size / 2)
    d_i = np.arange(0, d / 2)
    freq = 1 / (10e7 ** (2 * d_i / d))

    x, y = coords[0] / 10e7, coords[1] / 10e7
    enc = np.zeros(d * 2)
    enc[0:d:2] = np.sin(x * freq)
    enc[1:d:2] = np.cos(x * freq)
    enc[d::2] = np.sin(y * freq)
    enc[d + 1 :: 2] = np.cos(y * freq)
    return list(enc)

def norm_alti(alti: int) -> float:
    min_alti = 0
    max_alti = 3164.9099121094
    return [(alti - min_alti) / (max_alti - min_alti)]

def format_cam(cam: str) -> np.array:
    return [[1, 0] if "UCE" in cam else [0, 1]][0]

def cyclical_enc_datetime(date: str, time: str) -> list:
    def norm(num: float) -> float:
        return (num - (-1)) / (1 - (-1))

    year, month, day = date.split("-")
    if year == "2018":
        enc_y = [1, 0, 0, 0]
    elif year == "2019":
        enc_y = [0, 1, 0, 0]
    elif year == "2020":
        enc_y = [0, 0, 1, 0]
    elif year == "2021":
        enc_y = [0, 0, 0, 1]
    sin_month = np.sin(2 * np.pi * (int(month) - 1 / 12))  ## months of year
    cos_month = np.cos(2 * np.pi * (int(month) - 1 / 12))
    sin_day = np.sin(2 * np.pi * (int(day) / 31))  ## max days
    cos_day = np.cos(2 * np.pi * (int(day) / 31))
    h, m = time.split("h")
    sec_day = int(h) * 3600 + int(m) * 60
    sin_time = np.sin(2 * np.pi * (sec_day / 86400))  ## total sec in day
    cos_time = np.cos(2 * np.pi * (sec_day / 86400))
    return enc_y + [
        norm(sin_month),
        norm(cos_month),
        norm(sin_day),
        norm(cos_day),
        norm(sin_time),
        norm(cos_time),
    ]

def flair_gather_data(
    path_folders, path_metadata: str, use_metadata: bool, test_set: bool
) -> dict:
    #### return data paths
    def get_data_paths(path, filter):
        for path in Path(path).rglob(filter):
            yield path.resolve().as_posix()

    img, msk, mtd = [], [], []
    if path_folders:
        for domain in path_folders:
            print(f"Processing domain {domain}")
            # list_img = sorted(list(get_data_paths(domain, 'IMG*.tif')), key=lambda x: int(x.split('_')[-2][1:]))
            list_img = sorted(
                list(get_data_paths(domain, "IMG*.tif")),
                key=lambda x: int(x.split("_")[-1][:-4]),
            )
            img += list_img
            if test_set == False:
                # list_msk = sorted(list(get_data_paths(domain, 'MSK*.tif')), key=lambda x: int(x.split('_')[-2][1:]))
                list_msk = sorted(
                    list(get_data_paths(domain, "MSK*.tif")),
                    key=lambda x: int(x.split("_")[-1][:-4]),
                )
                msk += list_msk
            # print(f'domain {domain}: {[(img, msk) for img, msk in zip(list_img, list_msk)]}')
            # break

        if use_metadata == True:
            with open(path_metadata, "r") as f:
                metadata_dict = json.load(f)
            for img in data["IMG"]:
                curr_img = img.split("/")[-1][:-4]
                enc_coords = coordenc_opt(
                    [
                        metadata_dict[curr_img]["patch_centroid_x"],
                        metadata_dict[curr_img]["patch_centroid_y"],
                    ]
                )
                enc_alti = norm_alti(metadata_dict[curr_img]["patch_centroid_z"])
                enc_camera = format_cam(metadata_dict[curr_img]["camera"])
                enc_temporal = cyclical_enc_datetime(
                    metadata_dict[curr_img]["date"], metadata_dict[curr_img]["time"]
                )
                mtd_enc = enc_coords + enc_alti + enc_camera + enc_temporal
                mtd.append(mtd_enc)

    return img, msk, mtd