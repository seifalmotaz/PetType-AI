import os
import time
import torch
import torch.nn as nn
from ai import shuffle_files, transform
from torch.utils.data import DataLoader
from ai.data_classes import AnimalDataset, AnimalNet

EPOCHS = 5
BATCH_SIZE = 36
FILTRED_CATS_PATH = "./dataset/filtred/cat"
FILTRED_DOGS_PATH = "./dataset/filtred/dog"
# FILTRED_BIRDS_PATH = "./dataset/filtred/bird"


cats_img_files = os.listdir(FILTRED_CATS_PATH)
cats_img_files = list(map(lambda p: f"{FILTRED_CATS_PATH}/{p}", cats_img_files))

dogs_img_files = os.listdir(FILTRED_DOGS_PATH)
dogs_img_files = list(map(lambda p: f"{FILTRED_DOGS_PATH}/{p}", dogs_img_files))

# birds_img_files = os.listdir(FILTRED_BIRDS_PATH)
# birds_img_files = list(map(lambda p: f"{FILTRED_BIRDS_PATH}/{p}", birds_img_files))

img_files = cats_img_files + dogs_img_files# + birds_img_files
train_files, valid_files = shuffle_files(img_files)

train_ds = AnimalDataset(train_files, transform)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE)

valid_ds = AnimalDataset(valid_files, transform)
valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE)

model = AnimalNet().cuda()
epochs = EPOCHS
losses = []
accuracies = []
start = time.time()
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    for X, y in train_dl:
        X = X.cuda()
        y = y.cuda()
        preds = model(X)
        loss = loss_fn(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = (preds.argmax(dim=1) == y).float().mean()
        epoch_accuracy += accuracy
        epoch_loss += loss
        print(".", end="", flush=True)

    epoch_accuracy = epoch_accuracy / len(train_dl)
    accuracies.append(epoch_accuracy)
    epoch_loss = epoch_loss / len(train_dl)
    losses.append(epoch_loss)
    print(
        "\nEpoch: {}, train loss: {:.4f}, train accracy: {:.4f}, time: {}".format(
            epoch, epoch_loss, epoch_accuracy, time.time() - start
        )
    )

    with torch.no_grad():
        val_epoch_loss = 0
        val_epoch_accuracy = 0
        for val_X, val_y in valid_dl:
            val_X = val_X.cuda()
            val_y = val_y.cuda()
            val_preds = model(val_X)
            val_loss = loss_fn(val_preds, val_y)

            val_epoch_loss += val_loss
            val_accuracy = (val_preds.argmax(dim=1) == val_y).float().mean()
            val_epoch_accuracy += val_accuracy
        val_epoch_accuracy = val_epoch_accuracy / len(valid_dl)
        val_epoch_loss = val_epoch_loss / len(valid_dl)
        print(
            "Epoch: {}, valid loss: {:.4f}, valid accracy: {:.4f}, time: {}\n".format(
                epoch, val_epoch_loss, val_epoch_accuracy, time.time() - start
            )
        )

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": losses,
        "accuracies": accuracies,
        "loss_fn": loss_fn,
    },
    "./animal_type_model.pth",
)
