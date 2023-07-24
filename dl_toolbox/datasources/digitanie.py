import enum
from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import rasterio
import shapely

import torch

from .tif_datasource import TifDatasource


class DigiPolygons(enum.Enum):
    toulouse = shapely.Polygon(
        [
            [359326, 4833160],
            [376735, 4842547],
            [385238, 4826271],
            [367914, 4816946],
            [359326, 4833160],
        ]
    )

    arcachon = shapely.Polygon(
        [
            [629473, 4959797],
            [649299, 4960143],
            [649315, 4940864],
            [630504, 4940913],
            [629473, 4959797],
        ]
    )

    biarritz = shapely.Polygon(
        [
            [620357, 4825277],
            [628680, 4825402],
            [629181, 4802997],
            [610093, 4802622],
            [610155, 4808442],
            [620357, 4825277],
        ]
    )

    brisbane = shapely.Polygon(
        [
            [494125, 6974741],
            [494125, 6974741],
            [518910, 6973355],
            [518910, 6973355],
            [518712, 6953715],
            [518712, 6953715],
            [493976, 6955348],
            [493976, 6955348],
            [494125, 6974741],
        ]
    )

    buenosaires = shapely.Polygon(
        [
            [356354, 6181219],
            [375609, 6181386],
            [376276, 6145542],
            [356937, 6145459],
            [356354, 6181219],
        ]
    )

    cantho = shapely.Polygon(
        [
            [574413, 1117316],
            [593936, 1117304],
            [593959, 1103770],
            [574344, 1103701],
            [574413, 1117316],
        ]
    )

    helsinki = shapely.Polygon(
        [
            [378038, 6677645],
            [397619, 6676929],
            [397165, 6664249],
            [377680, 6664846],
            [378038, 6677645],
        ]
    )

    lagos = shapely.Polygon(
        [
            [528578, 733313],
            [547696, 733545],
            [547542, 705913],
            [528578, 706068],
            [528578, 733313],
        ]
    )

    cairo = shapely.Polygon(
        [
            [321171, 3336141],
            [340490, 3335740],
            [340162, 3314598],
            [320588, 3314926],
            [321171, 3336141],
        ]
    )

    maros = shapely.Polygon(
        [
            [775303, 9473671],
            [795021, 9473362],
            [794849, 9461765],
            [777230, 9461971],
            [775303, 9473671],
        ]
    )

    montpellier = shapely.Polygon(
        [
            [569263, 4842348],
            [579275, 4847307],
            [585825, 4841974],
            [585170, 4832523],
            [576562, 4831868],
            [576936, 4825880],
            [579181, 4824102],
            [579369, 4821576],
            [568982, 4816804],
            [569263, 4842348],
        ]
    )

    munich = shapely.Polygon(
        [
            [681767, 5348957],
            [701333, 5349752],
            [701955, 5330290],
            [682666, 5329512],
            [681767, 5348957],
        ]
    )

    strasbourg = shapely.Polygon(
        [
            [393875, 5382644],
            [398185, 5383652],
            [411526, 5383377],
            [411434, 5373887],
            [409371, 5372282],
            [408179, 5365681],
            [403503, 5367698],
            [404236, 5369761],
            [402998, 5371549],
            [396901, 5369944],
            [395755, 5374345],
            [397864, 5375446],
            [393875, 5382644],
        ]
    )

    nantes = shapely.Polygon(
        [
            [597911, 5240048],
            [619604, 5239798],
            [620103, 5220682],
            [598410, 5220848],
            [597911, 5240048],
        ]
    )
    newyork = shapely.Polygon(
        [
            [582100, 4516163],
            [590073, 4516192],
            [590013, 4503800],
            [582190, 4503741],
            [582100, 4516163],
        ]
    )

    paris = shapely.Polygon(
        [
            [443272, 5423320],
            [462678, 5423046],
            [462472, 5404120],
            [443203, 5404051],
            [443272, 5423320],
        ]
    )

    portelisabeth = shapely.Polygon(
        [
            [358931, 6255259],
            [376445, 6255433],
            [372154, 6242906],
            [378823, 6236701],
            [359569, 6236469],
            [358931, 6255259],
        ]
    )

    rio = shapely.Polygon(
        [
            [670012, 7479859],
            [676732, 7479713],
            [689295, 7460867],
            [669939, 7460429],
            [670012, 7479859],
        ]
    )

    sanfrancisco = shapely.Polygon(
        [
            [541991, 4182570],
            [553709, 4185286],
            [559529, 4158202],
            [543155, 4157426],
            [541991, 4182570],
        ]
    )

    shanghai = shapely.Polygon(
        [
            [344856, 3486292],
            [363657, 3486033],
            [363139, 3442206],
            [344337, 3442206],
            [344856, 3486292],
        ]
    )

    tianjin = shapely.Polygon(
        [
            [552371, 4330867],
            [571582, 4330867],
            [571674, 4295401],
            [552648, 4295401],
            [552371, 4330867],
        ]
    )


label = namedtuple("label", ["name", "color", "values"])

initial_nomenclature = [
    label("nodata", (250, 250, 250), {0}),
    label("bare_ground", (100, 50, 0), {1}),
    label("low_vegetation", (0, 250, 50), {2}),
    label("water", (0, 50, 250), {3}),
    label("building", (250, 50, 50), {4}),
    label("high_vegetation", (0, 100, 50), {5}),
    label("parking", (200, 200, 200), {6}),
    label("road", (100, 100, 100), {7}),
    label("railways", (200, 100, 200), {8}),
    label("swimmingpool", (50, 150, 250), {9}),
]

main_nomenclature = [
    label("other", (0, 0, 0), {0, 1, 6, 9}),
    label("low vegetation", (0, 250, 50), {2}),
    label("high vegetation", (0, 100, 50), {5}),
    label("water", (0, 50, 250), {3}),
    label("building", (250, 50, 50), {4}),
    label("road", (100, 100, 100), {7}),
    label("railways", (200, 100, 200), {8}),
]

nomenc24 = [
    label("nodata", (250, 250, 250), {0}),
    label("bare_ground", (100, 50, 0), {1}),
    label("low_vegetation", (0, 250, 50), {2}),
    label("water", (0, 50, 250), {3}),
    label("building", (250, 50, 50), {4}),
    label("high_vegetation", (0, 100, 50), {5}),
    label("parking", (200, 200, 200), {6}),
    label("road", (100, 100, 100), {7}),
    label("railways", (200, 100, 200), {8}),
    label("swimmingpool", (50, 150, 250), {9}),
    label("arboriculture", (50, 150, 100), {10}),
    label("snow", (250, 250, 250), {11}),
    label("sportsground", (250, 200, 50), {12}),
    label("storage_tank", (180, 180, 180), {13}),
    label("pond", (150, 200, 200), {14}),
    label("pedestrian", (150, 50, 50), {15}),
    label("roundabout", (50, 50, 50), {16}),
    label("container", (250, 250, 0), {17}),
    label("aquaculture", (50, 150, 200), {18}),
    label("port", (80, 80, 80), {19}),
    label("boat", (0, 250, 250), {20}),
    label("high building", (200, 0, 0), {21}),
    label("winter_vegetation", (200, 250, 200), {22}),
    label("industry", (200, 0, 250), {23}),
    label("beach", (250, 250, 100), {24}),
]


@dataclass
class Digitanie9(TifDatasource):
    classes = enum.Enum(
        "Digitanie9classes",
        {
            "all": initial_nomenclature,
            "main": main_nomenclature,
        },
    )


@dataclass
class Digitanie24(TifDatasource):
    classes = enum.Enum(
        "Digitanie24classes",
        {
            "all": nomenc24,
        },
    )
