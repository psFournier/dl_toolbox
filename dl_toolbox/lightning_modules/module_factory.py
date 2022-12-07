from dl_toolbox.lightning_modules import *


modules = {
    'CE': CE,
    'CE_PL': CE_PL,
    'CE_MT': CE_MT,
    'BCE': BCE,
    'CPS': CPS
}

class ModuleFactory:

    @staticmethod
    def create(name):
        return modules[name]
