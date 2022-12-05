from dl_toolbox.torch_datasets import *


datasets = {
    'Resisc': ResiscDs,
    'DigitanieV2': DigitanieV2,
    #'DigitanieToulouse': DigitanieToulouseDs,
    #'DigitanieBiarritz': DigitanieBiarritzDs,
    #'DigitanieMontpellier': DigitanieMontpellierDs,
    #'DigitanieParis': DigitanieParisDs,
    #'DigitanieStrasbourg': DigitanieStrasbourgDs,
    'SemcityToulouse': SemcityBdsdDs,
    'Airs': Airs,
    'Miniworl': Miniworld,
    #'Inria': Inria
}

class DatasetFactory:

    @staticmethod
    def create(name):
        return datasets[name]
