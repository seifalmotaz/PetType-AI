# import os
import torch
from PIL import Image
from ai import transform
from ai.data_classes import AnimalNet, labels_list
from io import BytesIO
import requests
# import torch.nn.functional as F


load_model = torch.load("./animal_type_model.pth")
model = AnimalNet().cuda()
model.load_state_dict(load_model["model_state_dict"])

response = requests.get("https://instagram.fcai19-8.fna.fbcdn.net/v/t51.2885-15/116708183_303983080913507_1866743988291278391_n.jpg?stp=dst-jpg_e35&_nc_ht=instagram.fcai19-8.fna.fbcdn.net&_nc_cat=111&_nc_ohc=Tk4fIZrAIaoAX8I-I7Z&edm=ABmJApABAAAA&ccb=7-5&ig_cache_key=MjM2OTAyNzA3ODExMjAzMjg2MQ%3D%3D.2-ccb7-5&oh=00_AfAQTQjOg_64WaeXy__rYr7BXRNCJOWFuDAq06AliOgJ8Q&oe=647BBE1E&_nc_sid=a1ad6c")
img = Image.open(BytesIO(response.content))

img = transform(img)
img = img.unsqueeze(0).cuda()

model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    preds = model(img)
    pred = preds.argmax(dim=1)
    label =  list(labels_list.keys())[list(labels_list.values())[pred]]
    title = " Classified as: " + label
    print(title)

"""
TEST_CATS_PATH = "./test_set/cats"
TEST_DOGS_PATH = "./test_set/dogs"

cats_img_files = os.listdir(TEST_CATS_PATH)
cats_img_files = list(map(lambda p: f"{TEST_CATS_PATH}/{p}", cats_img_files))

dogs_img_files = os.listdir(TEST_DOGS_PATH)
dogs_img_files = list(map(lambda p: f"{TEST_DOGS_PATH}/{p}", dogs_img_files))

correct_preds = 0
total_preds = 0

correct_cat_preds = 0
total_cat_preds = 0

correct_dog_preds = 0
total_dog_preds = 0

load_model = torch.load("./animal_type_model.pth")
model = AnimalNet().cuda()
model.load_state_dict(load_model["model_state_dict"])

model.eval()  
with torch.no_grad():
    for img_file in cats_img_files:
        img = Image.open(img_file)
        img = transform(img)
        img = img.unsqueeze(0).cuda()
        pred = model(img)
        pred = pred.argmax(dim=1)
        total_preds += 1
        total_cat_preds += 1
        if pred == labels_list["cat"]:
            correct_preds += 1
            correct_cat_preds += 1

    for img_file in dogs_img_files:
        img = Image.open(img_file)
        img = transform(img)
        img = img.unsqueeze(0).cuda()
        pred = model(img)
        pred = pred.argmax(dim=1)
        total_preds += 1
        total_dog_preds += 1
        if pred == labels_list["dog"]:
            correct_preds += 1
            correct_dog_preds += 1

print(f"Accuracy: {correct_preds / total_preds}")
print(f"Cat Accuracy: {correct_cat_preds / total_cat_preds}")
print(f"Dog Accuracy: {correct_dog_preds / total_dog_preds}")
"""