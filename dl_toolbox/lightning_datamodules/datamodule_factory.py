from dl_toolbox.lightning_datamodules import *


datamodules = {
    'Splitfile': Splitfile,
    'FromFolderDataset': FromFolderDataset
    #'SplitIdxSup': SplitIdxSup,
    #'SplitIdxSemisup': SplitIdxSemisup,
    #'SplitfileSup': SplitfileSup,
    #'SplitfileSemisup': SplitfileSemisup
}

class DatamoduleFactory:

    @staticmethod
    def create(name):
        return datamodules[name]
