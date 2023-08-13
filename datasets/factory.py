import os
from .build import CIFAR10LT, CIFAR100LT
from .augmentation import train_augmentation, test_augmentation

def load_cifar10lt(
    datadir: str, img_size: int, mean: tuple, std: tuple, aug_info: list = None,
    imbalance_type: str = None, imbalance_factor: int = 1):

    trainset = CIFAR10LT(
        root             = os.path.join(datadir,'CIFAR10'), 
        train            = True, 
        download         = True,
        transform        = train_augmentation(img_size=img_size, mean=mean, std=std, aug_info=aug_info),
        imbalance_type   = imbalance_type,
        imbalance_factor = imbalance_factor
    )

    testset = CIFAR10LT(
        root             = os.path.join(datadir,'CIFAR10'), 
        train            = False, 
        download         = True,
        transform        = test_augmentation(img_size=img_size, mean=mean, std=std)
    )
        
    return trainset, testset


def load_cifar100lt(
    datadir: str, img_size: int, mean: tuple, std: tuple, aug_info: list = None, 
    imbalance_type: str = None, imbalance_factor: int = 1):

    trainset = CIFAR100LT(
        root             = os.path.join(datadir,'CIFAR100'), 
        train            = True, 
        download         = True,
        transform        = train_augmentation(img_size=img_size, mean=mean, std=std, aug_info=aug_info),
        imbalance_type   = imbalance_type,
        imbalance_factor = imbalance_factor
    )

    testset = CIFAR100LT(
        root             = os.path.join(datadir,'CIFAR100'), 
        train            = False, 
        download         = True,
        transform        = test_augmentation(img_size=img_size, mean=mean, std=std)
    )

    return trainset, testset



