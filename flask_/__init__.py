from flask import Flask

def create_app():
    app = Flask(__name__)

    from .views import main_views, csf_views, ocr_views, stt_tts_views, chatbot_views
    app.register_blueprint(csf_views.bp)
    app.register_blueprint(main_views.bp)
    app.register_blueprint(ocr_views.bp)
    app.register_blueprint(stt_tts_views.bp)
    app.register_blueprint(chatbot_views.bp)

    return app
