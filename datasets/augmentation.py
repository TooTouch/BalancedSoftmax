from torchvision import transforms

def add_augmentation(transform: transforms.Compose, img_size: int, aug_info: list = None):
    augments_dict = {
        'RandomCrop': transforms.RandomCrop((img_size, img_size), padding=4),
        'RandomHorizontalFlip': transforms.RandomHorizontalFlip(),
        'RandomVerticalFlip': transforms.RandomVerticalFlip(),
        'Resize': transforms.Resize((img_size, img_size)),
    }
    # insert augmentations
    if aug_info != None:    
        for aug in aug_info:
            transform.transforms.insert(-1, augments_dict[aug])   
    else:
        transform.transforms.insert(-1, augments_dict['Resize'])
    
    return transform


def train_augmentation(img_size: int, mean: tuple, std: tuple, aug_info: list = None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform = add_augmentation(transform=transform, img_size=img_size, aug_info=aug_info)

    return transform

def test_augmentation(img_size: int, mean: tuple, std: tuple):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean, std),
    ])

    return transform
