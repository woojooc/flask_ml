from flask import Blueprint, request

import speech_recognition as sr
from gtts import gTTS
import playsound
import time

bp = Blueprint('stt_tts',__name__,url_prefix='/stt_tts')

@bp.route('/', methods=['POST'])
def stt_tts_main():
    r = sr.Recognizer()
    try:
        while True:
            # 음성 입력 받기
            with sr.Microphone() as source:
                print('음성을 입력하세요')
                audio = r.listen(source)
                result = r.recognize_google(audio, language='ko-KR')
                print('음성: ' + result)

                # if '카카오' in result:  # '헤이 카카오'등 키워드를 추가하여 음성인식 서비스처럼 사용 가능
                tts = gTTS(text=result, lang='ko')
                path = 'test.mp3'  # 음성파일 경로설정
                tts.save(path)  # 음성파일 저장
                # time.sleep(3)  # 저장하는데 시간이 걸리면 잠시 대기
                playsound.playsound(path)  # 녹음파일 실행
    except:
        print('마이크로폰 에러')