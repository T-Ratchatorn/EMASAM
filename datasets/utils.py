from datasets.cifar100 import cifar100
from datasets.cifar10 import cifar10
from datasets.oxfordiiitpet import oxfordiiitpet
from datasets.flowers102 import flowers102
from datasets.imagenet_1k import imagenet_1k
from datasets.emnist import emnist
from datasets.fashionmnist import fashionmnist

def get_data(dataset, num_classes, batch_size, num_worker, tsubame_id, is_DDP=False):
    dataset = dataset.lower()
    if( dataset == 'cifar100' and num_classes == 100 ):
        data = cifar100(batch_size=batch_size, threads=num_worker, tsubame_id=tsubame_id)
    elif( dataset == 'cifar10' and num_classes == 10 ):
        data = cifar10(batch_size=batch_size, threads=num_worker, tsubame_id=tsubame_id)
    elif( dataset == 'fashionmnist' and num_classes == 10 ):
        data = fashionmnist(batch_size=batch_size, threads=num_worker, tsubame_id=tsubame_id)
    elif( dataset == 'emnist' and num_classes == 47 ):
        data = emnist(batch_size=batch_size, threads=num_worker, tsubame_id=tsubame_id)
    elif( dataset == 'flowers102' and num_classes == 102 ):
        data = flowers102(batch_size=batch_size, threads=num_worker, tsubame_id=tsubame_id)
    elif( dataset == 'oxfordiiitpet' and num_classes == 37 ):
        data = oxfordiiitpet(batch_size=batch_size, threads=num_worker, tsubame_id=tsubame_id)
    elif( dataset == 'imagenet_1k' and num_classes == 1000 ):
        data = imagenet_1k(batch_size=batch_size, threads=num_worker, tsubame_id=tsubame_id, is_DDP=is_DDP)
    else:
        msg = f"dataset: {name}, num_classes: {num_classes} -- NotCompatible"
        raise ValueError(msg)
    return data