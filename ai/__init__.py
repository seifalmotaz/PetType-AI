import random
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def shuffle_files(img_files: list[str]) -> tuple[list[str], list[str]]:
    """
    Args:
        img_files (list[str]): 

    Returns:
        tuple[list[str], list[str]]: train_files, valid_files
    """
    random.shuffle(img_files)
    img_files_len = len(img_files)
    train_files = img_files[: int(img_files_len * 0.8)]
    valid_files = img_files[int(img_files_len * 0.8) :]
    return train_files, valid_files