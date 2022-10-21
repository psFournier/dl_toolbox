from dl_toolbox.lightning_datamodules import *


datamodules = {
    'ResiscSup': ResiscSup,
    'ResiscSemisup': ResiscSemisup,
    'SplitfileSup': SplitfileSup,
    'SplitfileSemisup': SplitfileSemisup
}

class DatamoduleFactory:

    @staticmethod
    def create(name):
        return datamodules[name]
