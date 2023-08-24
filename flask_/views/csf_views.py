from flask import Blueprint, request

# 추론
import torch
from torchvision import transforms #전처리

from PIL import Image # 이미지 불러오기


bp = Blueprint('csf', __name__,url_prefix='/csf' )

# 모델 입력 받기
model = torch.load('model.pt', map_location = torch.device('cpu'))

@bp.route('/', methods = ['POST'])
def csf_main():

    print(request.files)

    # 키값으로 이미지 파일 받기
    f = request.files['image']
    print(f)
    f.save(f.filename)

    # 이미지 불러오기
    image = Image.open(f.filename)
    # 이미지 전처리
    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    image = transform_test(image).unsqueeze(0).to('cpu')

    model.to('cpu')
    with torch.no_grad():
        outputs = model(image) # 추론
        _, preds = torch.max(outputs, 1)

        # 결과 확인
        classname = ['cat','dog','fox']
        print(classname[preds[0]])

    return classname[preds[0]] + ' 입니다.'