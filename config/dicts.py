from __future__ import annotations
from torch import nn
from collections import namedtuple, defaultdict
from typing import DefaultDict, NamedTuple
from nodes import layer, dataset
from app.trash import CustomDataset

import torch
import torchvision.datasets as ds # type: ignore
from torchvision.transforms import v2 # type: ignore


Module: NamedTuple = namedtuple('Module', ['generator', 'func', 'params', 'default_params','popup','details','image', 'tooltip'], 
                                defaults=(None, None, None, None, None, None, None, None))
modules: DefaultDict[str, NamedTuple] = defaultdict(lambda: Module("Not present"))

_dtypes = {'float32': v2.ToDtype(torch.float32, scale=True), 
                  'int32': v2.ToDtype(torch.int32), 
                  'int64': v2.ToDtype(torch.int64)} 
transforms_img = [
    {"label":"Resize", "type":'text/tuple', "default_value":"[224, 224]", "user_data": v2.Resize},
    {"label":"ToImage", "type":'blank', "user_data": v2.ToImage},
    {"label":"ToDtype", "type":'combo', "default_value":"float32",
        "items":tuple(_dtypes.keys()), "user_data": _dtypes},
    {"label":"AutoAugment", "type":'blank', "user_data": v2.AutoAugment },
    {"label":"RandomIoUCrop", "type":'blank', "user_data": v2.RandomIoUCrop},
    {"label":"ElasticTransform", "type":'blank', "user_data": v2.ElasticTransform},
    {"label":"Grayscale", "type":'blank', "user_data": v2.Grayscale},
    # {"label":"RandomCrop", "type":'blank', "user_data": v2.RandomCrop},
    {"label":"RandomVerticalFlip", "type":'blank', "user_data": v2.RandomVerticalFlip},
    {"label":"RandomHorizontalFlip", "type":'blank', "user_data": v2.RandomHorizontalFlip}
                    ] 
transforms_setting_img = {"label":"Transform", "type":'collaps', "items":transforms_img}


params = {
    "img_transforms": transforms_setting_img,
    "batch_size" :   {"label":"batch_size", "type":'int', "default_value":64, "tooltip": "__Батчи/Пакеты/сеты/партии__ - это набор объектов тренировочного датасета, который пропускается итеративно через сеть во время обучения\n___\n1 < batch_size < full_size"},
    "val_size":     {"label":"val_size", "type":'float', 
                     "max_value": 0.9999999, "max_clamped":True, "default_value":0.2},
    "button_load":  {"label":"Load Dataset", "type":"path", "default_value":"/home/luzinsan/Environments/petrol/data/"},
    "default_train":{'Loss':'L1 Loss','Optimizer':'SGD'},
    "out_features": {"label":"out_features", "type":'int', "default_value":1, "tooltip":"Количество признаков на выходе линейной трансформации.\nКоличество признаков на входе определяется автоматически"},
    "out_channels": {"label":"out_channels", "type":'int', "default_value":6, 
                     "tooltip":"Количество выходных каналов/признаковых карт, которые являются репрезентациями для последующих слоёв (рецептивное поле)"
                     },
    "num_features": {"label":"num_features", "type":'int', "default_value":6},
    "output_size":{"label":"output_size", "type":'text/tuple', "default_value":'[1, 2]', 'tooltip':"Целевой выходной размер изображения формы HxW. Может быть списком [H, W] или одним H (для квадратного изображения HxH).\nH и W могут быть либо int , либо None. None означает, что размер будет таким же, как и у входных данных."},
    "kernel_size":{"label":"kernel_size", "type":'int', "default_value":5, 
                   'tooltip':"Размер тензорного ядра"
                   },
    "stride":{"label":"stride", "type":'int', "default_value":1, 
              'tooltip': "Шаг прохождения тензорного ядра во время свёртки (взаимной корреляции)"
              },
    "padding":{"label":"padding", "type":'int', "default_value":0, 
               'tooltip':"Размер заполнения краёв входной матрицы"
               },
    "eps":          {"label":"eps", "type":'float', "default_value":1e-5},
    "momentum":{"label":"momentum", "type":'float', "default_value":0.1},
    "affine":{"label":"affine", "type":'bool', "default_value": True},
    "p":{"label":"p", "type":'float', "default_value":0.5, 'tooltip':"Вероятность обнуления элемента"},
    "dim":{"label":"dim", "type":'int', "default_value":1, 'tooltip':"Рассматриваемое измерение"},
}

default_params = {
    'Loss':'Cross Entropy',
    'Optimizer':'SGD',
}
fromkeys = lambda d, keys: {x:d.get(x) for x in keys}
modules.update({
####################################################################~~~ DATASETS ~~~####################################################################
    "FashionMNIST":       Module(dataset.DataNode.factory, ds.FashionMNIST, 
                                 (params['img_transforms'], params['batch_size']), 
                                 fromkeys(default_params, ['Loss', 'Optimizer']), image='./static/images/fashion-mnist.png', popup="""
### Fashion-MNIST
___
+ это набор изображений, предоставленный [Zalando](https://arxiv.org/pdf/1708.07747)
+ состоит из обучающего набора из 60 000 примеров и тестового набора из 10 000 примеров. 
+ Каждый пример представляет собой изображение в оттенках серого размером 28x28
+ Представлено 10 классов
_На изображении каждый класс представлен в трёх строках_
                            """),
    "Caltech101":       Module(dataset.DataNode.factory, ds.Caltech101, 
                               (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "Caltech256":       Module(dataset.DataNode.factory, ds.Caltech256, 
                               (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "CarlaStereo":       Module(dataset.DataNode.factory, ds.CarlaStereo, 
                                (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "CelebA":       Module(dataset.DataNode.factory, ds.CelebA, 
                           (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "CIFAR10":       Module(dataset.DataNode.factory, ds.CIFAR10, 
                            (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "Cityscapes":       Module(dataset.DataNode.factory, ds.Cityscapes, 
                               (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "CLEVRClassification":       Module(dataset.DataNode.factory, ds.CLEVRClassification, 
                                        (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "EMNIST":       Module(dataset.DataNode.factory, ds.EMNIST, 
                           (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "CocoCaptions":       Module(dataset.DataNode.factory, ds.CocoCaptions, 
                                 (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "EuroSAT":       Module(dataset.DataNode.factory, ds.EuroSAT, 
                            (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "Flowers102":       Module(dataset.DataNode.factory, ds.Flowers102, 
                               (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "Food101":       Module(dataset.DataNode.factory, ds.Food101, 
                            (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "ImageNet":       Module(dataset.DataNode.factory, ds.ImageNet, 
                             (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "SUN397":       Module(dataset.DataNode.factory, ds.SUN397, 
                           (params['img_transforms'], params['batch_size']), fromkeys(default_params, ['Loss', 'Optimizer'])),
    "Dataset from File":       Module(dataset.DataNode.factory, CustomDataset, 
                                      (params['val_size'], params['button_load']), fromkeys(default_params, ['Loss', 'Optimizer'])),

######################################################################~~~ LINEARS ~~~#######################################################################
    "LazyLinear":       Module(layer.LayerNode.factory, nn.LazyLinear, (params['out_features'],), image="./static/images/linear_layer.png", popup=
                               """
Линейный слой 
___
_Другие названия: полносвязный или плотный (Dense) слой_
+ это линейное преобразование над входящими данными (его обучаемые параметры - это матрица _W_ и вектор _b_). Такой слой преобразует _d_-размерные векторы в _k_-размерные
                               """),
    "LazyConv1d":       Module(layer.LayerNode.factory, nn.LazyConv1d, (params['out_channels'], params['kernel_size'], params['stride'], params['padding']), tooltip="Применяет одномерную свертку к входному сигналу, состоящему из нескольких входных плоскостей"),
    "LazyConv2d":       Module(layer.LayerNode.factory, nn.LazyConv2d, (params['out_channels'], params['kernel_size'], params['stride'], params['padding']), tooltip="Применяет 2D-свертку к входному сигналу, состоящему из нескольких входных плоскостей"),
    "LazyConv3d":       Module(layer.LayerNode.factory, nn.LazyConv3d, (params['out_channels'], params['kernel_size'], params['stride'], params['padding']), tooltip="Применяет 3D-свертку к входному сигналу, состоящему из нескольких входных плоскостей"),
    "LazyBatchNorm1d":  Module(layer.LayerNode.factory, nn.LazyBatchNorm1d, (params['eps'], params['momentum'], params['affine']), tooltip='_Рекомендуемый размер пакета (в гиперпараметрах обучения) = 50-100_'),
    "LazyBatchNorm2d":  Module(layer.LayerNode.factory, nn.LazyBatchNorm2d, (params['eps'], params['momentum'], params['affine'])),
    "LazyBatchNorm3d":  Module(layer.LayerNode.factory, nn.LazyBatchNorm3d, (params['eps'], params['momentum'], params['affine'])),
    "Flatten":          Module(layer.LayerNode.factory, nn.Flatten),
    "Concatenate":      Module(layer.LayerNode.factory, torch.cat, (params['dim'], )),
    "AvgPool2d":        Module(layer.LayerNode.factory, nn.AvgPool2d, (params['kernel_size'], params['stride'], params['padding'])),
    "MaxPool2d":        Module(layer.LayerNode.factory, nn.MaxPool2d, (params['kernel_size'], params['stride'], params['padding'])),
    "AdaptiveAvgPool2d":Module(layer.LayerNode.factory, nn.AdaptiveAvgPool2d, (params['output_size'], ), tooltip="Применяет двумерное адаптивное усреднение к входному сигналу, состоящему из нескольких входных плоскостей"),
    "Dropout":          Module(layer.LayerNode.factory, nn.Dropout, (params['p'], ), tooltip="__Метод регуляризации и предотвращения совместной адаптации нейронов__\nВо время обучения случайным образом обнуляет некоторые элементы входного тензора.\nЭлементы выбираются независимо во время каждого прямого прохода (feed-forward) из распределения Бернулли. "),

#####################################################################~~~ ACTIVATIONS ~~~#####################################################################
    "ReLU":             Module(layer.LayerNode.factory, nn.ReLU, image='./static/images/ReLU.png', details="__Функция активации__ (activation function) - нелинейное преобразование, поэлементно применяющееся к пришедшим на вход данным. Благодаря функциям активации нейронные сети способны порождать более информативные признаковые описания, преобразуя данные нелинейным образом.", popup=
                               """
### Rectified Linear Unit
___
+ Наиболее популярная функция активации из-за простоты реализации и хорошей производительности
+ Сохраняет только положительные значения, обнуляя все отрицательные
+ Кусочно-линейная функция
+ Решает проблему затухающего градиента
                               """),
    "Sigmoid":          Module(layer.LayerNode.factory, nn.Sigmoid, image='./static/images/sigmoid.png', details="__Функция активации__ (activation function) - нелинейное преобразование, поэлементно применяющееся к пришедшим на вход данным. Благодаря функциям активации нейронные сети способны порождать более информативные признаковые описания, преобразуя данные нелинейным образом.", popup=
                               """
### Сигмоидная функция активации
___
+ Сжимает входные данные, преобразуя их в значения на интервале (0, 1) 
+ По этой причине сигмоиду часто называют сжимающей функцией: она сжимает любой вход в диапазоне (-inf, inf) до некоторого значения в диапазоне (0, 1)
+ Градиент функции обращается в нуль при больших положительных и отрицательных значениях аргументов, что является проблемой затухающего градиента
+ Полезна в рекуррентных сетях
                               """),
    "Tanh":             Module(layer.LayerNode.factory, nn.Tanh, image='./static/images/tanh.png', details="__Функция активации__ (activation function) - нелинейное преобразование, поэлементно применяющееся к пришедшим на вход данным. Благодаря функциям активации нейронные сети способны порождать более информативные признаковые описания, преобразуя данные нелинейным образом.", popup=
                               """
### Гиперболический тангенс
___
+ Сжимает входные данные, преобразуя их в значения на интервале (-1, 1)
+ Является симметричной относительно начала координат
+ Производная функции в нуле принимает значени 1
+ Градиент функции обращается в нуль при больших положительных и отрицательных значениях аргументов, что является проблемой затухающего градиента
                               """),
    
####################################################################~~~ ARCHITECTURES ~~~####################################################################
    "LeNet5":           Module(layer.ModuleNode.factory, image='./static/images/lenet.png', details="Базовыми единицами в каждом сверточном блоке являются сверточный слой, сигмоидальная функция активации и последующая операция объединения усреднений. Обратите внимание, что, хотя ReLU и max-pooling работают лучше, они еще не были обнаружены. Каждый сверточный слой использует ядро 5x5 и сигмоидальную функцию активации. Эти слои сопоставляют пространственно упорядоченные входные данные с несколькими двумерными картами объектов, обычно увеличивая количество каналов. Первый сверточный слой имеет 6 выходных каналов, а второй - 16. Каждая 2x2 операция объединения в пул (stride/шаг=2) уменьшает размерность в 4 раза, понижая пространственную дискретизацию. Сверточный блок выдает объект, размерностью (размер пакета, номер канала, высота, ширина).", popup=
                               """
### Свёрточная нейросеть
___
Одна из первых свёрточных сетей, заложившая основы глубокого обучения
+ Открыл: Ян Лекун [LeCun et al., 1989](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf), [LeCun et al., 1998b](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
+ В оригинальной версии используется average pooling и сигмоидные функции активации
+ Данный модуль модифицирован: max-pooling и ReLU функции активации
+ Применяется для классификации изображений (по-умолчанию настроен на 10 классов)
### Архитектура 
   Состоит из 2-х частей: 
+ Сверточные слои (kernel_size=5) с пуллинг слоями (kernel_size=2, stride=2)
+ 3 полносвязных слоя (out_features: 120 | 84 | 10 _(кол-во классов)_)
+ Без модификаций принимает изображения размером [28, 28]
                               """), 
    "AlexNet":          Module(layer.ModuleNode.factory, image='./static/images/alexnet.png', details="Вплоть до 2012 года самой важной частью конвейера было репрезентативность, которая рассчитывалась в основном механически. К тому же, разработка нового набора признаков (feature engineering), улучшение результатов и описание метода - все это занимало видное место в статьях: SIFT [Lowe, 2004](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf), SURF [Bay et al., 2006](https://people.ee.ethz.ch/~surf/eccv06.pdf), HOG (гистограммы ориентированного градиента) [Dalal and Triggs, 2005](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf), наборы визуальных слов [Sivic and Zisserman, 2003](https://www.robots.ox.ac.uk/~vgg/publications/2003/Sivic03/sivic03.pdf) и аналогичные экстракторы признаков. На самых нижних уровнях сети модель изучает элементарные признаки, которые напоминали некоторые традиционные фильтры. Более высокие слои сети могут опираться на эти представления для представления более крупных структур, таких как глаза, носы, травинки и так далее. Еще более высокие слои могут представлять целые объекты, такие как люди, самолеты, собаки или фрисби. В итоге, конечное скрытое состояние изучает компактное представление изображения, которое суммирует его содержимое таким образом, что данные, принадлежащие к различным категориям, могут быть легко разделены.", popup=
                               """
### Глубокая сверточная нейросеть
___
Углубленная и расширенная версия LeNet, разработанная для конкурса/набора_данных ImageNet 
+ Открыл: Алекс Крижевский [Krizhevsky et al., 2012](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf#page=5&zoom=310,88,777)
+ Используются функции активации ReLU в качестве нелинейностей.
+ Используется методика отбрасывания (Dropout) для выборочного игнорирования отдельных нейронов в ходе обучения, что позволяет избежать переобучения модели.
+ Перекрытие max pooling, что позволяет избежать эффектов усреднения (в случае average pooling)
+ Почти 10 раз больше карт признаков, чем у LeNet
### Архитектура 
   Состоит из 2-х частей: 
* Сверточные слои (kernel_size: 11 | 5 | 3 * 3) с max-pooling слоями (kernel_size=3, stride=2)
* 3 полносвязных слоя (out_features: 4096 | 84 | 10 _(кол-во классов)_) _(Осторожно: требуется почти 1 ГБ параметров модели)_
+ Без модификаций принимает изображения размером [224, 224]
                        """),
    "VGG-11":              Module(layer.ModuleNode.factory, image='./static/images/vgg.png', details="VGG противоречит заложенным в LeNet принципам, согласно которым большие свёртки использовались для извлечения одинаковых свойств изображения. Вместо применяемых в AlexNet фильтров 9х9 и 11х11 стали применять гораздо более мелкие фильтры, опасно близкие к свёрткам 1х1, которых старались избежать авторы LeNet, по крайней мере в первых слоях сети. Но большим преимуществом VGG стала находка, что несколько свёрток 3х3, объединённых в последовательность, могут эмулировать более крупные рецептивные поля, например, 5х5 или 7х7. \nСети VGG для представления сложных признаков используют многочисленные свёрточные слои 3x3. \n__Примечательно__: в [VGG-E](https://arxiv.org/pdf/1409.1556#page=3&zoom=160,-97,717) в блоках 3, 4 и 5 для извлечения более сложных свойств и их комбинирования применяются последовательности 256×256 и 512×512 фильтров 3×3. Это равносильно большим свёрточным классификаторам 512х512 с тремя слоями! Это даёт нам огромное количество параметров и прекрасные способности к обучению. Но учить такие сети было сложно, приходилось разбивать их на более мелкие, добавляя слои один за другим. Причина заключалась в отсутствии эффективных способов регуляризации моделей или каких-то методов ограничения большого пространства поиска, которому способствует множество параметров.", popup=
                                  """
### Блочная свёрточная нейросеть
___
Противоречит принципам LeNet и AlexNet, однако заложила основы для архитектур Inception и ResNet
+ Открыла: группа исследователей из VGG, Оксфордский университет [Simonyan and Zisserman, 2014](https://arxiv.org/pdf/1409.1556#page=3&zoom=160,-97,717)
+ Состоит из повторяющихся структур - VGG блоков
### Архитектура:
Данная версия рассчитана на 10 классов и намного меньше VGG-11 по кол-ву параметров.
+ __5__ VGG блоков:
  - __(1 | 1 | 2 | 2 | 3)__ свёрточных слоя на каждый блок соответственно (out_channels: 16 | 32 | 64 | 128 | 128, kernel_size=3, padding=1) с ReLU функциями активаций
  - Max-pooling слой (kernel_size=2, stride=2)
+ 3 полносвязных слоя (out_features: 120 | 84 | 10 _(кол-во классов)_) со слоями отбрасывания (Dropout, p=0.5)
+ Без модификаций принимает изображения размером [224, 224]
                               """),
    "Conv-MLP":              Module(layer.ModuleNode.factory, popup=
                                    """
### Многослойный перцептрон со свёрточными слоями
___
Является составляющим архитектуры NiN
### Архитектура:
+ 3 свёрточных слоя (out_channels: 96 | 96 | 96, kernel_size: 11 | 1 | 1, stride: 4 | 1 | 1, padding=0)
+ ReLU функции активации после каждого свёрточного слоя
                               """),
    "NiN":          Module(layer.ModuleNode.factory, image='./static/images/NiN.png', details="NiN были предложены на основе очень простого понимания: \n1. использовать 1x1 свертки для добавления локальных нелинейностей по активациям каналов и\n2. использовать глобальный средний пул для интеграции по всем местоположениям в последнем слое представления. _Причём глобальное среднее объединение не было бы эффективным, если бы не дополнительные нелинейности_", popup=
                               """                    
### Архитектура NiN
___
Призвана решить проблемы VGG по части большого кол-ва параметров
+ Открыл: Мин Линь [Lin et al., 2013](https://arxiv.org/pdf/1312.4400)
+ Содержит модули Conv-MLP
+ MLP позволяют сильно повысить эффективность отдельных свёрточных слоёв посредством их комбинирования в более сложные группы.
+ Совершенно не использует полносвязные слои, что кратно уменьшает кол-во параметров, однако потенциально увеличивает время обучения
                                """),
     "Inception":          Module(layer.ModuleNode.factory, image='./static/images/inception.png', popup='_Блок сети GoogLeNet_'),
                                 
     "GoogLeNet":          Module(layer.ModuleNode.factory, image='./static/images/googlenet.png',
                                details="""
### Задания
GoogLeNet оказался настолько успешным, что прошел ряд итераций, постепенно улучшая скорость и точность. Попробуйте реализовать и запустить некоторые из них. Они включают в себя следующее:
+ Добавьте слой пакетной нормализации [Иоффе и Сегеди, 2015](https://arxiv.org/pdf/1502.03167)
+ Внесите изменения в блок Inception (ширина, выбор и порядок сверток), как описано у [Szegedy et al. (2016)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)
+ Используйте сглаживание меток для регуляризации модели, как описано в [Szegedy et al. (2016)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)
+ Внесите дальнейшие корректировки в блок Inception, добавив остаточное (residual) соединение [Szegedy et al., 2017](http://www.cs.cmu.edu/~jeanoh/16-785/papers/szegedy-aaai2017-inception-v4.pdf)
                                """,
                                  popup="""
### Многофилиальная сеть [GoogLeNet](https://arxiv.org/pdf/1409.4842)
___
Сочетает в себе сильные стороны NiN, вдохновлен LeNet и AlexNet
+ Довольно время- и ресурсозатратна
+ Здесь представлен упрощенный вариант. Оригинал включает ряд приёмов стабилизации обучения. В них больше нет необходимости из-за улучшенных алгоритмов обучения.
+ Базовый сверточный блок - Inception. Настраеваемые гиперпараметры - кол-во выходных каналов (output channels)
+ GoogLeNet использует 9 Inception блоков, сгруппированных на 3 группы (по 2, 5, 2 блока) 
+ Одна из первых сверточных сетей, в которой различаются основная часть (приём данных), тело (обработка данных) и голова (прогнозирование)
  - Основу задают 2-3 свертки, которые работают с изображением и извлекают низкоуровневые признаки
  - Телом является набор сверточных блоков
  - В голове сопоставляются полученные признаки с целевым признаком по задаче (классификации, сегментации, детекции или отслеживания)
                                """),
    "BN LeNet":         Module(layer.ModuleNode)

})

