from dl_toolbox.datasets import *


datasets = {
    #'Resisc': Resisc,
    'Digitanie': Digitanie,
    #'DigitanieToulouse': DigitanieToulouseDs,
    #'DigitanieBiarritz': DigitanieBiarritzDs,
    #'DigitanieMontpellier': DigitanieMontpellierDs,
    #'DigitanieParis': DigitanieParisDs,
    #'DigitanieStrasbourg': DigitanieStrasbourgDs,
    #'Semcity': Semcity,
    #'Airs': Airs,
    #'Miniworld': Miniworld,
    #'Inria': Inria
}

class DatasetFactory:

    @staticmethod
    def create(name):
        return datasets[name]
