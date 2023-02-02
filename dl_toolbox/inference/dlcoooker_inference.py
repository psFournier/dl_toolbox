from dataclasses import dataclass
from dlcooker import dlcooker_infer
import glob
import numpy as np
import os
from dlcooker.configs.inference import (
    GeoTIFFOutputConf,
    ImageConf,
    ModelConf,
    SemanticSegmentationInferenceConfig,
    LabelMergeStrategy,
    TTAMergeStrategy,
    TilingConf,
    TTASequentialPipelineConf,
    TTAHorizontalFlipConf,
    TTAVerticalFlipConf,
    TTARotationsConf,
    TTACombinationPipelineConf
)

@dataclass
class MyInferenceConfig(SemanticSegmentationInferenceConfig):

    image: ImageConf = ImageConf(path="/work/OT/ai4geo/DATA/DATASETS/DIGITANIE/Montpellier/Montpellier_EPSG2154_0.tif", bands=[1, 2, 3])
  
    model: ModelConf = ModelConf(
        folder= "/work/OT/ai4usr/hummerc/dlcooker_wrapper/",
        metadata_filename='metadata.yaml',
        pth_filename='model.pth',
        tta_pipeline = None,
        #tta_pipeline=TTACombinationPipelineConf(transforms=[
        #    TTAHorizontalFlipConf(),
        #    TTAVerticalFlipConf(),
        #    TTARotationsConf(angles=[90, 180, 270])
        #])
        #,
        tta_merge_strategy=TTAMergeStrategy.MEAN,
    )
    
    
    tiling: TilingConf = TilingConf(
        strategy=LabelMergeStrategy.BLENDING,
        batch_size= 8,
        num_workers= 1,
        tile_size = 2048,
        size_kept = 2048,
        patch_size=2048,
        patch_overlap=0
)
    
    output: GeoTIFFOutputConf = GeoTIFFOutputConf(folder = "/work/OT/ai4usr/hummerc/dlcooker_wrapper/", filename="Montpellier_EPSG2154_0", save_labels=True, confidence_threshold=0.5)
    

if __name__ == '__main__':

    dlcooker_infer(config=MyInferenceConfig)