###데이터 다운로드 및 데이터 학습 환경 형성

import os, shutil
import matplotlib.pyplot as plt
#import koreanize_matplotlib
from bing_image_downloader.bing_image_downloader import downloader

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import time


###다운로드 데이터 디렉토리 생성
directory_list = [
    './dataset/train/',
    './dataset/test/'
]

for dir in directory_list:
    if not os.path.isdir(dir):  #dir에 해당하는 경로가 없다면
        os.makedirs(dir)  #경로에 폴더 생성

# query=검색어, limit=다운받을개수, output_dir=이미지 저장할 경로, adult_filter_off=성인콘텐츠필터, force_replace=덮어쓰기, timeout=최대요청시간)
downloader.download(query='cat', limit=50, output_dir='./', adult_filter_off=True, force_replace=False, timeout=60)
downloader.download(query='dog', limit=50, output_dir='./', adult_filter_off=True, force_replace=False, timeout=60)
downloader.download(query='fox', limit=50, output_dir='./', adult_filter_off=True, force_replace=False, timeout=60)


###이미지데이터 train, test 폴더에 할당
def dataset_split(query,train_cnt):

    #학습 데이터용 디렉토리 생성
    for dir in directory_list:
        if not os.path.isdir(dir+query):
            os.makedirs(dir+query)

    cnt = 0
    for file_name in os.listdir(query):
        if cnt < train_cnt:
            print(f'trainset : {file_name}')
            shutil.move(query+'/'+file_name,'./dataset/train/'+query+'/'+file_name)
        else:
            print(f'testset : {file_name}')
            shutil.move(query+'/'+file_name,'./dataset/test/'+query+'/'+file_name)

        cnt += 1

    shutil.rmtree(query)

dataset_split('cat',40)
dataset_split('dog',40)
dataset_split('fox',40)

###모델 전이학습


device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


train_datasets = datasets.ImageFolder('./dataset/train',transform_train)
test_datasets = datasets.ImageFolder('./dataset/test',transform_test)


train_dataloader = torch.utils.data.DataLoader(train_datasets,shuffle=True,batch_size=4)
test_dataloader = torch.utils.data.DataLoader(test_datasets,shuffle=True,batch_size=4)


class_names = train_datasets.classes
model = models.resnet34(weights=True)


for param in model.parameters():
    param.requires_grad = False


fc_input_features = model.fc.in_features
fc_input_features

model.fc = nn.Linear(fc_input_features, 3)

for name,module in model.named_parameters():
    print(name , module.requires_grad)



optimizer = optim.Adam(model.parameters(), lr=1e-4)
model.to(device)
start_time = time.time()
for epoch in range(50):
    for data,labels in train_dataloader:
        optimizer.zero_grad()
        preds = model(data.to(device))
        loss = nn.CrossEntropyLoss()(preds,labels.to(device))
        loss.backward()
        optimizer.step()

    print(f'epoch : {epoch+1} loss : {loss.item()} time : {time.time() - start_time}')


model.eval()
with torch.no_grad():
    corrects = 0
    for data,labels in test_dataloader:
        preds = model(data.to(device))
        pred = torch.max(preds,1)[1]

        corrects += torch.sum(pred == labels.to(device).data)

        print(f'예측결과 : {class_names[pred[0]]}  실제정답 : {class_names[labels.data[0]]}')

    acc = corrects / len(test_datasets) * 100
    print(f'정확도 : {acc}')

torch.save(model, 'model.pt')  # 전체 모델 저장
torch.save(model.state_dict(), 'model_state_dict.pt')


model1 = models.resnet34(weights=True)
model1.fc = nn.Linear(512, 3)

model_state_dict = torch.load('model_state_dict.pt',map_location='cuda')
model1.load_state_dict(model_state_dict)