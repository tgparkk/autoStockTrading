from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_socketio import SocketIO, emit
import os
import secrets
import sys
import time
import json
import threading
import logging
from datetime import datetime

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 자동매매 시스템 래퍼 임포트
from trading_system import TradingSystem

# Flask 앱 및 SocketIO 설정
app = Flask(__name__)
app.config['SECRET_KEY'] =  secrets.token_hex(16) #'your_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('logs/web_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 자동매매 시스템 초기화
trading_system = TradingSystem()

# 백그라운드 데이터 업데이트 스레드
# 백그라운드 데이터 업데이트 스레드
def background_updater():
    """백그라운드에서 주식 데이터를 업데이트하고 소켓으로 전송"""
    while True:
        try:
            # 주식 데이터 업데이트
            stock_data = trading_system.get_current_stock_data()
            socketio.emit('stock_update', stock_data)
            
            # 계좌 정보 업데이트
            account_data = trading_system.get_account_info()
            if account_data:
                logger.info(f"계좌 정보 업데이트: summary={len(account_data.get('account_summary', []))}건, stocks={len(account_data.get('stocks', []))}건")
            else:
                logger.warning("계좌 정보가 없습니다.")
            socketio.emit('account_update', account_data)
            
            # 시스템 상태 업데이트
            system_status = {
                'status': trading_system.get_status(),
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'running': trading_system.is_running
            }
            socketio.emit('system_status', system_status)
            
            # 로그 업데이트
            recent_logs = trading_system.get_recent_logs(10)
            socketio.emit('log_update', recent_logs)
            
            time.sleep(3)  # 3초마다 업데이트 (API 요청 한도 고려)
            
        except Exception as e:
            logger.error(f"Background updater error: {str(e)}")
            time.sleep(5)  # 오류 발생 시 5초 후 재시도

# 라우트 설정
@app.route('/')
def index():
    """홈 페이지"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """대시보드 페이지"""
    return render_template('dashboard.html')

@app.route('/settings')
def settings():
    """설정 페이지"""
    return render_template('settings.html', 
                          config=trading_system.get_strategy_config(),
                          target_stocks=trading_system.get_target_stocks(),
                          interval=trading_system.get_interval())
                          

@app.route('/logs')
def logs():
    """로그 페이지"""
    return render_template('logs.html', logs=trading_system.get_recent_logs(100))

# API 엔드포인트
@app.route('/api/start', methods=['POST'])
def start_trading():
    """매매 시작"""
    try:
        trading_system.start()
        return jsonify({'success': True, 'message': '자동매매를 시작했습니다.'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'오류 발생: {str(e)}'})

@app.route('/api/stop', methods=['POST'])
def stop_trading():
    """매매 중지"""
    try:
        trading_system.stop()
        return jsonify({'success': True, 'message': '자동매매를 중지했습니다.'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'오류 발생: {str(e)}'})

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """설정 업데이트"""
    try:
        data = request.json
        # 전략 설정 업데이트
        if 'strategy' in data:
            trading_system.update_strategy_config(data['strategy'])
        
        # 타겟 종목 업데이트
        if 'target_stocks' in data:
            trading_system.update_target_stocks(data['target_stocks'])
        
        # 실행 간격 업데이트
        if 'interval' in data:
            trading_system.update_interval(data['interval'])

        return jsonify({'success': True, 'message': '설정이 업데이트되었습니다.'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'오류 발생: {str(e)}'})

@app.route('/api/stock/<stock_code>/detail')
def stock_detail(stock_code):
    """종목 상세 정보"""
    try:
        days = request.args.get('days', 30, type=int)
        detail = trading_system.get_stock_detail(stock_code, days)
        return jsonify(detail)
    except Exception as e:
        return jsonify({'success': False, 'message': f'오류 발생: {str(e)}'})

# Socket.IO 이벤트
@socketio.on('connect')
def handle_connect():
    """클라이언트 연결 처리"""
    logger.info(f"Client connected: {request.sid}")
    # 초기 데이터 전송
    emit('stock_update', trading_system.get_current_stock_data())
    emit('account_update', trading_system.get_account_info())
    emit('system_status', {
        'status': trading_system.get_status(),
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'running': trading_system.is_running
    })

@socketio.on('disconnect')
def handle_disconnect():
    """클라이언트 연결 해제 처리"""
    logger.info(f"Client disconnected: {request.sid}")

@app.route('/api/account/info')
def get_account_info():
    """계좌 정보 API 엔드포인트"""
    try:
        account_data = trading_system.get_account_info()
        return jsonify(account_data)
    except Exception as e:
        logger.error(f"계좌 정보 API 오류: {str(e)}")
        return jsonify({
            'account_summary': [{
                'dnca_tot_amt': '12',  # 예수금
                'scts_evlu_amt': '0',  # 주식 평가금액
                'tot_evlu_amt': '12',  # 총 평가금액
                'pchs_amt_smtl_amt': '0',  # 매입금액
                'evlu_pfls_smtl_amt': '0',  # 평가손익
                'asst_icdc_erng_rt': '0.00'  # 수익률
            }],
            'stocks': []
        })

# 메인 실행
if __name__ == '__main__':
    # 디렉토리 생성
    os.makedirs('logs', exist_ok=True)
    
    # 백그라운드 스레드 시작
    updater_thread = threading.Thread(target=background_updater, daemon=True)
    updater_thread.start()
    
    # 앱 실행 (개발 시)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
    
    # 프로덕션 환경에서는 아래 방식으로 실행
    # gunicorn -k eventlet -w 1 app:app --bind 0.0.0.0:5000