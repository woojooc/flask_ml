from flask import Blueprint


bp = Blueprint('main', __name__,url_prefix='/' )

@bp.route('/')
def main_index():

    return 'start'