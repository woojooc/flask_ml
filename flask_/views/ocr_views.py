from flask import Blueprint, request
from PIL import Image
from PaddleOCR import PaddleOCR, draw_ocr

bp = Blueprint('ocr',__name__,url_prefix='/ocr')

@bp.route('/', methods=['POST'])
def ocr_main():
    ocr = PaddleOCR(lang='korean')
    f = request.files['image']
    f.save(f.filename)

    result = ocr.ocr(f.filename, cls=False)
    aa = []
    for i in result:
        for j in i:
            for ii in j:
                for jj in ii:
                    if type(jj) == str:
                        aa.append(jj)
    return ' '.join(aa)