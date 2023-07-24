from dl_toolbox.lightning_modules import *


modules = {
    "CE": CE,
    "CE_PL": CE_PL,
    "CE_MT": CE_MT,
    "BCE": BCE,
    "BCE_PL": BCE_PL,
    "CPS": CPS,
}


class ModuleFactory:
    @staticmethod
    def create(name):
        return modules[name]
