from dl_toolbox.torch_datasets import *


datasets = {
    'Resisc': ResiscDs,
    'DigitanieToulouse': DigitanieToulouseDs,
    'DigitanieBiarritz': DigitanieBiarritzDs,
    'DigitanieMontpellier': DigitanieMontpellierDs,
    'DigitanieParis': DigitanieParisDs,
    'DigitanieStrasbourg': DigitanieStrasbourgDs,
    'SemcityToulouse': SemcityBdsdDs
}

class DatasetFactory:

    @staticmethod
    def create(name):
        return datasets[name]
