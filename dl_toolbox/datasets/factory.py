from dl_toolbox.torch_datasets import *


datasets = {
    #'Resisc': Resisc,
    'DigitanieV2': Digitanie,
    #'DigitanieToulouse': DigitanieToulouseDs,
    #'DigitanieBiarritz': DigitanieBiarritzDs,
    #'DigitanieMontpellier': DigitanieMontpellierDs,
    #'DigitanieParis': DigitanieParisDs,
    #'DigitanieStrasbourg': DigitanieStrasbourgDs,
    'Semcity': Semcity,
    #'Airs': Airs,
    #'Miniworld': Miniworld,
    #'Inria': Inria
}

class DatasetFactory:

    @staticmethod
    def create(name):
        return datasets[name]
