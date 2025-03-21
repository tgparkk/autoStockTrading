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
            market_regime = None
            next_update = None
            
            # 통합 전략이 있는 경우 시장 국면 정보 가져오기
            if hasattr(trading_system.strategy, 'market_regime'):
                market_regime = trading_system.strategy.market_regime
            
            # 다음 월요일 8시
            from datetime import datetime, timedelta
            now = datetime.now()
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0 and now.hour >= 8:
                days_until_monday = 7
            next_monday = now + timedelta(days=days_until_monday)
            next_update = next_monday.replace(hour=8, minute=0, second=0).strftime('%Y-%m-%d %H:%M')
            
            system_status = {
                'status': trading_system.get_status(),
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'running': trading_system.is_running,
                'market_regime': market_regime,
                'next_update': next_update
            }
            socketio.emit('system_status', system_status)
            
            # 로그 업데이트
            recent_logs = trading_system.get_recent_logs(10)
            socketio.emit('log_update', recent_logs)
            
            # 선정된 종목 정보 업데이트
            stocks_info = []
            if hasattr(trading_system.strategy, 'selected_stocks') and trading_system.strategy.selected_stocks:
                # 통합 전략에서 선정된 종목 정보 가져오기
                for stock_code in trading_system.strategy.selected_stocks:
                    stock_info = {
                        'code': stock_code,
                        'selected_date': datetime.now().strftime('%Y-%m-%d'),
                        'score': None
                    }
                    
                    # 선정 점수 정보가 있는 경우
                    if hasattr(trading_system.strategy, 'stock_scores') and stock_code in trading_system.strategy.stock_scores:
                        stock_info['score'] = trading_system.strategy.stock_scores[stock_code]
                    
                    stocks_info.append(stock_info)
            else:
                # 기존 target_stocks 사용
                for stock_code in trading_system.target_stocks:
                    stocks_info.append({
                        'code': stock_code,
                        'selected_date': '-',
                        'score': None
                    })
            
            socketio.emit('selected_stocks_update', stocks_info)
            
            time.sleep(30)  # 30초마다 업데이트 (API 요청 한도 고려)
            
        except Exception as e:
            logger.error(f"Background updater error: {str(e)}")
            time.sleep(50)  # 오류 발생 시 50초 후 재시도

# 라우트 설정
@app.route('/')
def index():
    """홈 페이지"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """대시보드 페이지"""
    # 계좌 정보 조회
    account_data = trading_system.get_account_info()
    
    # 시장 국면 정보
    market_regime = None
    if hasattr(trading_system.strategy, 'market_regime'):
        market_regime = trading_system.strategy.market_regime
    
    # 선정된 종목 정보
    selected_stocks = []
    if hasattr(trading_system.strategy, 'selected_stocks'):
        selected_stocks = trading_system.strategy.selected_stocks
    else:
        selected_stocks = trading_system.target_stocks
    
    return render_template('dashboard.html', 
                          account_data=account_data,
                          market_regime=market_regime,
                          selected_stocks=selected_stocks)

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
                'dnca_tot_amt': '-1',  # 예수금
                'scts_evlu_amt': '0',  # 주식 평가금액
                'tot_evlu_amt': '-1',  # 총 평가금액
                'pchs_amt_smtl_amt': '0',  # 매입금액
                'evlu_pfls_smtl_amt': '0',  # 평가손익
                'asst_icdc_erng_rt': '0.00'  # 수익률
            }],
            'stocks': []
        })
    
# 종목 목록 조회 API
@app.route('/api/stocks/list')
def get_stocks_list():
    """선정된 종목 목록 정보"""
    try:
        stocks_info = []
        
        if hasattr(trading_system.strategy, 'selected_stocks') and trading_system.strategy.selected_stocks:
            # 통합 전략에서 선정된 종목 정보 가져오기
            for stock_code in trading_system.strategy.selected_stocks:
                stock_info = {
                    'code': stock_code,
                    'selected_date': None,
                    'score': None
                }
                
                # 선정 점수 정보가 있는 경우
                if hasattr(trading_system.strategy, 'stock_scores') and stock_code in trading_system.strategy.stock_scores:
                    stock_info['score'] = trading_system.strategy.stock_scores[stock_code]
                
                stocks_info.append(stock_info)
        else:
            # 기존 target_stocks 사용
            for stock_code in trading_system.target_stocks:
                stocks_info.append({
                    'code': stock_code,
                    'selected_date': '-',
                    'score': None
                })
        
        # 소켓으로 업데이트된 정보 전송
        socketio.emit('selected_stocks_update', stocks_info)
        
        return jsonify({'success': True, 'stocks': stocks_info})
    except Exception as e:
        logger.error(f"종목 목록 조회 중 오류: {str(e)}")
        return jsonify({'success': False, 'message': f'오류 발생: {str(e)}'})

# 종목 수동 갱신 API
@app.route('/api/stocks/update', methods=['POST'])
def update_stocks():
    """종목 목록 수동 갱신"""
    try:
        # 통합 전략이 있는 경우 weekly_update 호출
        if hasattr(trading_system.strategy, 'weekly_update'):
            success = trading_system.strategy.weekly_update()
            
            if success:
                # 선정된 종목 정보를 target_stocks에 복사
                if hasattr(trading_system.strategy, 'selected_stocks'):
                    trading_system.target_stocks = trading_system.strategy.selected_stocks.copy()
                
                # 파일에 저장
                with open(trading_system.stocks_path, 'w', encoding='utf-8') as f:
                    for stock in trading_system.target_stocks:
                        f.write(f"{stock}\n")
                
                # 종목 목록 정보 가져오기
                stocks_info = get_stocks_list().json.get('stocks', [])
                
                # 소켓으로 업데이트된 정보 전송
                socketio.emit('selected_stocks_update', stocks_info)
                
                return jsonify({'success': True, 'message': f'종목 목록이 갱신되었습니다. {len(trading_system.target_stocks)}개 종목이 선정되었습니다.'})
            else:
                return jsonify({'success': False, 'message': '종목 갱신에 실패했습니다.'})
        else:
            return jsonify({'success': False, 'message': '통합 전략이 적용되지 않았습니다.'})
    except Exception as e:
        logger.error(f"종목 갱신 중 오류: {str(e)}")
        return jsonify({'success': False, 'message': f'오류 발생: {str(e)}'})


# 시스템 상태 API에 시장 국면 정보 추가
@app.route('/api/system/status')
def get_system_status():
    """시스템 상태 정보"""
    try:
        market_regime = None
        next_update = None
        
        # 통합 전략이 있는 경우 시장 국면 정보 가져오기
        if hasattr(trading_system.strategy, 'market_regime'):
            market_regime = trading_system.strategy.market_regime
        
        # 다음 월요일 8시
        from datetime import datetime, timedelta
        now = datetime.now()
        days_until_monday = (7 - now.weekday()) % 7
        if days_until_monday == 0 and now.hour >= 8:
            days_until_monday = 7
        next_monday = now + timedelta(days=days_until_monday)
        next_update = next_monday.replace(hour=8, minute=0, second=0).strftime('%Y-%m-%d %H:%M')
        
        status = {
            'status': trading_system.get_status(),
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'running': trading_system.is_running,
            'market_regime': market_regime,
            'next_update': next_update
        }
        
        return jsonify(status)
    except Exception as e:
        logger.error(f"시스템 상태 조회 중 오류: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})
    

# 대시보드 데이터 API
@app.route('/api/dashboard/data')
def dashboard_data():
    """대시보드에 표시할 데이터 API"""
    try:
        # 계좌 정보 가져오기
        account_data = trading_system.get_account_info()
        
        # 시장 국면 정보
        market_regime = None
        regime_confidence = 0.75  # 기본값
        
        # 통합 전략이 있는 경우 시장 국면 정보 가져오기
        if hasattr(trading_system.strategy, 'market_regime'):
            market_regime = trading_system.strategy.market_regime
        
        # 모델 정보 (초기 샘플 데이터)
        model_info = {
            'model_type': 'RandomForest Classifier',
            'last_training': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'accuracy': 0.685,
            'f1_score': '0.72'
        }
        
        # 보유 종목 정보
        holdings_info = {
            'count': len(account_data.get('stocks', [])),
            'max': 20,
            'buy_signals': 5  # 샘플 데이터
        }
        
        # 종목 스코어 샘플 데이터
        stocks_data = []
        
        # 선정된 종목이 있으면 해당 종목 사용
        selected_stocks = []
        if hasattr(trading_system.strategy, 'selected_stocks') and trading_system.strategy.selected_stocks:
            selected_stocks = trading_system.strategy.selected_stocks
        else:
            selected_stocks = trading_system.target_stocks
        
        # 주요 종목 샘플 데이터 (실제 구현 시에는 실제 데이터로 대체)
        sample_stocks = [
            {'code': '005930', 'name': '삼성전자', 'momentum': 0.65, 'technical': 0.58, 'ml_prediction': 0.72, 'total_score': 0.65, 'signal': 'buy'},
            {'code': '000660', 'name': 'SK하이닉스', 'momentum': 0.55, 'technical': 0.62, 'ml_prediction': 0.64, 'total_score': 0.60, 'signal': 'buy'},
            {'code': '035420', 'name': 'NAVER', 'momentum': 0.48, 'technical': 0.52, 'ml_prediction': 0.55, 'total_score': 0.51, 'signal': 'neutral'},
            {'code': '035720', 'name': '카카오', 'momentum': 0.42, 'technical': 0.45, 'ml_prediction': 0.51, 'total_score': 0.46, 'signal': 'neutral'},
            {'code': '051910', 'name': 'LG화학', 'momentum': 0.38, 'technical': 0.41, 'ml_prediction': 0.35, 'total_score': 0.38, 'signal': 'neutral'}
        ]
        
        # 선정된 종목과 샘플 데이터 매핑
        for stock_code in selected_stocks[:10]:  # 상위 10개만 표시
            # 샘플 데이터에서 찾기
            found = False
            for sample in sample_stocks:
                if sample['code'] == stock_code:
                    stocks_data.append(sample)
                    found = True
                    break
            
            # 샘플에 없는 경우 임의 데이터 생성
            if not found:
                momentum = round(random.uniform(0.3, 0.7), 2)
                technical = round(random.uniform(0.2, 0.7), 2)
                ml_prediction = round(random.uniform(0.2, 0.8), 2)
                total_score = round(momentum * 0.35 + technical * 0.35 + ml_prediction * 0.3, 2)
                
                signal = 'neutral'
                if total_score > 0.6:
                    signal = 'buy'
                elif total_score < 0.3:
                    signal = 'sell'
                
                # 종목명 추정 (실제로는 API에서 가져와야 함)
                stock_name = f'종목{stock_code[-4:]}'
                
                stocks_data.append({
                    'code': stock_code,
                    'name': stock_name,
                    'momentum': momentum,
                    'technical': technical,
                    'ml_prediction': ml_prediction,
                    'total_score': total_score,
                    'signal': signal
                })
        
        # 특성 중요도 샘플 데이터
        feature_importance = {
            'labels': ['RSI', '볼린저밴드', 'MACD', '이동평균선', '거래량변화', '가격변동성', '모멘텀', 'ADX', '일목균형표', 'OBV'],
            'values': [0.18, 0.15, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.04]
        }
        
        # 모델 성능 히스토리 샘플 데이터
        performance_history = {
            'dates': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30, 0, -1)],
            'accuracy': [round(random.uniform(0.6, 0.8), 2) for _ in range(30)],
            'f1_score': [round(random.uniform(0.55, 0.75), 2) for _ in range(30)]
        }
        
        # 포트폴리오 배분 샘플 데이터
        portfolio_allocation = {
            'labels': ['현금', 'IT/소프트웨어', '반도체', '바이오/제약', '화학/소재', '금융'],
            'values': [40, 20, 15, 10, 8, 7]
        }
        
        # 포트폴리오 가치
        portfolio_value = 0
        portfolio_change = 0
        
        if account_data and 'account_summary' in account_data and account_data['account_summary']:
            summary = account_data['account_summary'][0]
            portfolio_value = int(summary.get('tot_evlu_amt', 0))
            portfolio_change = float(summary.get('asst_icdc_erng_rt', 0)) / 10000  # 수익률
        
        # 최종 응답 데이터
        response_data = {
            'portfolio': {
                'value': portfolio_value,
                'change': portfolio_change
            },
            'market_regime': market_regime,
            'regime_confidence': regime_confidence,
            'model': model_info,
            'holdings': holdings_info,
            'stocks': stocks_data,
            'feature_importance': feature_importance,
            'performance_history': performance_history,
            'portfolio_allocation': portfolio_allocation
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"대시보드 데이터 조회 중 오류: {str(e)}")
        return jsonify({'error': str(e)})

# 사용하기 위해서는 추가할 임포트
import random

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