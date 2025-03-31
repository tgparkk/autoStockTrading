from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
from flask_socketio import SocketIO, emit
import os
import secrets
import sys
import time
import json
import threading
import logging
from datetime import datetime, timedelta  
import glob
import re

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 자동매매 시스템 래퍼 임포트
from trading_system import TradingSystem

# Flask 앱 및 SocketIO 설정
app = Flask(__name__)
app.config['SECRET_KEY'] =  secrets.token_hex(16) #'your_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# 로깅 설정 강화
logging.basicConfig(
    level=logging.DEBUG,  # INFO에서 DEBUG로 변경
    format='[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f'logs/web_app_debug_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 자동매매 시스템 초기화
trading_system = TradingSystem()

# ML 모델 정보 업데이트 (15분마다 전송)
def update_ml_model_info():
    try:
        # ML 모델 정보 가져오기
        ml_model_info = trading_system.get_ml_model_info()
        
        # 소켓으로 전송
        if ml_model_info:
            socketio.emit('ml_model_update', ml_model_info)
            logger.info("ML 모델 정보 업데이트 전송됨")
    except Exception as e:
        logger.error(f"ML 모델 정보 업데이트 중 오류: {str(e)}")

# 백그라운드 데이터 업데이트 스레드
def background_updater():
    """백그라운드에서 주식 데이터를 업데이트하고 소켓으로 전송"""
    ml_update_counter = 0  # ML 정보 업데이트 카운터
    
    # signal_utils 임포트
    from src.utils.signal_utils import get_trading_signal
    
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
            
            # 선정된 종목 정보 업데이트 (신호 계산 추가)
            stocks_info = []
            
            if hasattr(trading_system.strategy, 'selected_stocks') and trading_system.strategy.selected_stocks:
                target_stocks = trading_system.strategy.selected_stocks
            else:
                target_stocks = trading_system.target_stocks
                
            for stock_code in target_stocks:
                # 통합 신호 계산 함수 사용
                signal_info = get_trading_signal(trading_system.market_data, stock_code)
                
                # 종목명 조회
                stock_name = "알 수 없음"
                current_price = None
                
                if stock_code in stock_data:
                    stock_name = stock_data[stock_code].get('prdt_name', '알 수 없음')
                    current_price = stock_data[stock_code].get('stck_prpr', '0')
                
                # 점수 정보
                score = None
                if hasattr(trading_system.strategy, 'stock_scores') and stock_code in trading_system.strategy.stock_scores:
                    score = trading_system.strategy.stock_scores[stock_code]
                elif 'score' in signal_info:
                    score = signal_info['score']
                
                # 선정일자
                selected_date = datetime.now().strftime('%Y-%m-%d')
                if hasattr(trading_system.strategy, 'selection_date'):
                    selected_date = trading_system.strategy.selection_date
                
                # 종목 정보 구성
                stock_info = {
                    'code': stock_code,
                    'name': stock_name,
                    'current_price': current_price,
                    'selected_date': selected_date,
                    'score': score,
                    'signal': signal_info['signal'],  # 통합 신호 사용
                    'signal_score': signal_info['score'],  # 신호 점수
                    'signal_reasons': signal_info.get('reasons', [])  # 신호 이유
                }
                
                # 로깅 추가 (디버깅용)
                logger.debug(f"UI 표시 신호 계산: {stock_code} - {signal_info['signal']} (점수: {signal_info['score']})")
                
                stocks_info.append(stock_info)
            
            # 소켓으로 업데이트된 정보 전송
            socketio.emit('selected_stocks_update', stocks_info)
            
            # ML 모델 정보 업데이트 (15분마다)
            ml_update_counter += 1
            if ml_update_counter >= 30:  # 30초 간격으로 30번 = 15분
                update_ml_model_info()
                ml_update_counter = 0
            
            time.sleep(30)  # 30초마다 업데이트 (API 요청 한도 고려)
            
        except Exception as e:
            logger.error(f"Background updater error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())  # 상세 스택 트레이스 기록
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
        logger.info("계좌 정보 API 요청 받음")
        
        # 1. API 호출 시도
        try:
            account_data = trading_system.market_data.get_account_balance()
            logger.info(f"계좌 API 응답 성공: {account_data is not None}")
        except Exception as api_error:
            logger.error(f"계좌 API 호출 오류: {str(api_error)}")
            account_data = None
        
        # 2. 응답 데이터 확인 및 보정
        if account_data and isinstance(account_data, dict):
            # 기존 로직 유지
            return jsonify(account_data)
        else:
            # 더미 데이터 반환
            logger.warning("유효한 계좌 데이터 없음, 더미 데이터 반환")
            dummy_data = {
                'account_summary': [{
                    'dnca_tot_amt': '500000',
                    'scts_evlu_amt': '500000',
                    'tot_evlu_amt': '1000000',
                    'pchs_amt_smtl_amt': '450000',
                    'evlu_pfls_smtl_amt': '50000',
                    'asst_icdc_erng_rt': '1000',
                    'is_dummy': True  # 클라이언트에서 확인 가능
                }],
                'stocks': []
            }
            return jsonify(dummy_data)
            
    except Exception as e:
        logger.error(f"계좌 정보 API 오류: {str(e)}", exc_info=True)
        # 심플한 오류 응답
        return jsonify({'error': str(e)}), 500  # 500 에러 코드 반환
    
# 종목 목록 조회 API 수정
@app.route('/api/stocks/list')
def get_stocks_list():
    """선정된 종목 목록 정보"""
    try:
        logger.debug("종목 목록 API 요청 받음")
        
        # signal_utils 임포트
        from src.utils.signal_utils import get_trading_signal

        stocks_info = []
        
        # 수정된 종목 정보 맵핑 함수 - 종목코드로 종목명 조회
        def get_stock_name(stock_code):
            # 기본 종목명 사전 (자주 사용되는 종목)
            default_stock_names = {
                '005930': '삼성전자',
                '000660': 'SK하이닉스',
                '035420': 'NAVER',
                '035720': '카카오',
                '051910': 'LG화학',
                '005380': '현대차',
                '012330': '현대모비스',
                '068270': '셀트리온',
                '006400': '삼성SDI'
                # 필요한 만큼 더 추가
            }
            
            # 먼저 기본 사전에서 찾기
            if stock_code in default_stock_names:
                return default_stock_names[stock_code]
            
            # 사전에 없으면 API로 조회
            try:
                current_data = trading_system.market_data.get_stock_current_price(stock_code)
                if current_data and 'prdt_name' in current_data:
                    return current_data['prdt_name']
            except Exception as e:
                logger.warning(f"종목명 조회 중 오류({stock_code}): {str(e)}")
            
            # 모두 실패하면 기본값으로 종목코드명 반환
            return f"종목{stock_code}"
        
        # 종목 정보 수집
        stock_source = []
        if hasattr(trading_system.strategy, 'selected_stocks') and trading_system.strategy.selected_stocks:
            stock_source = trading_system.strategy.selected_stocks
        else:
            stock_source = trading_system.target_stocks
        
        # 각 종목에 대한 정보 수집
        for stock_code in stock_source:
            # 종목명 조회
            stock_name = get_stock_name(stock_code)

             # 신호 계산 (통합 로직 사용)
            signal_info = get_trading_signal(trading_system.market_data, stock_code)
            logger.debug(f"API 응답: {stock_code} - 신호: {signal_info['signal']}, 점수: {signal_info['score']}")
            
            # 현재가 조회 (추가된 부분)
            current_price = None
            try:
                current_data = trading_system.market_data.get_stock_current_price(stock_code)
                if current_data and 'stck_prpr' in current_data:
                    current_price = current_data['stck_prpr']
            except Exception as e:
                logger.warning(f"현재가 조회 중 오류({stock_code}): {str(e)}")
            
            # 점수 정보
            score = None
            if hasattr(trading_system.strategy, 'stock_scores') and stock_code in trading_system.strategy.stock_scores:
                score = trading_system.strategy.stock_scores[stock_code]
            elif 'score' in signal_info:
                score = signal_info['score']
            
            # 선정일자
            selected_date = datetime.now().strftime('%Y-%m-%d')
            if hasattr(trading_system.strategy, 'selection_date'):
                selected_date = trading_system.strategy.selection_date
            
            stocks_info.append({
                'code': stock_code,
                'name': stock_name,
                'current_price': current_price,
                'selected_date': selected_date,
                'score': score,
                'signal': signal_info['signal'],
                'signal_reasons': signal_info.get('reasons', [])
            })
        
        # 소켓으로 업데이트된 정보 전송
        socketio.emit('selected_stocks_update', stocks_info)

        # 로그 추가
        logger.info(f"종목 목록 API 응답: {len(stocks_info)}개 종목")
        
        return jsonify({'success': True, 'stocks': stocks_info})
    except Exception as e:
        logger.error(f"종목 목록 조회 중 오류: {str(e)}")
        return jsonify({'success': False, 'message': f'오류 발생: {str(e)}'})
# 종목 수동 갱신 API
@app.route('/api/stocks/update', methods=['POST'])
def update_stocks():
    """종목 목록 수동 갱신"""
    try:
        # 통합 전략 체크 및 주간 업데이트 호출
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
                response = get_stocks_list()
                stocks_info = response.json.get('stocks', [])
                
                # 소켓으로 업데이트된 정보 전송
                socketio.emit('selected_stocks_update', stocks_info)
                
                # 종목 선정 기록 저장
                trading_system.save_selected_stocks_history(stocks_info)
                
                return jsonify({'success': True, 'message': f'종목 목록이 갱신되었습니다. {len(trading_system.target_stocks)}개 종목이 선정되었습니다.'})
            else:
                return jsonify({'success': False, 'message': '종목 갱신에 실패했습니다.'})
        elif hasattr(trading_system, 'strategy') and trading_system.strategy:
            # 일일 트레이딩 전략은 weekly_update가 없을 수 있음
            return jsonify({'success': True, 'message': '일일 트레이딩 전략이 적용되었습니다. 종목 갱신이 필요하지 않습니다.'})
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
    
# 로그 페이지
@app.route('/logs')
def logs():
    """로그 페이지"""
    log_files = get_log_files()
    return render_template('logs.html', log_files=log_files)

# 로그 파일 목록 가져오기
def get_log_files():
    """로그 파일 목록 가져오기"""
    log_dir = os.path.abspath('logs')
    log_files = []
    
    try:
        # 디렉토리 내 모든 .log 파일 찾기
        for file_name in os.listdir(log_dir):
            if file_name.endswith('.log'):
                file_path = os.path.join(log_dir, file_name)
                
                # 파일 크기 계산
                try:
                    file_size = os.path.getsize(file_path)
                    if file_size < 1024:
                        size_str = f"{file_size} B"
                    elif file_size < 1024 * 1024:
                        size_str = f"{file_size / 1024:.1f} KB"
                    else:
                        size_str = f"{file_size / (1024 * 1024):.1f} MB"
                except:
                    size_str = "N/A"
                
                # 날짜 추출 시도
                try:
                    date_pattern = re.compile(r'(\d{8})')
                    date_match = date_pattern.search(file_name)
                    if date_match:
                        date_str = date_match.group(1)
                        file_date = datetime.strptime(date_str, '%Y%m%d')
                        display_date = file_date.strftime('%Y-%m-%d')
                    else:
                        display_date = "날짜 없음"
                except:
                    display_date = "날짜 없음"
                
                log_files.append({
                    'file_path': file_path,
                    'file_name': file_name,
                    'display_name': f"{display_date} ({file_name})",
                    'size': size_str,
                    'date': display_date
                })
    except Exception as e:
        logger.error(f"로그 파일 목록 로드 오류: {str(e)}")
        # 기본 목록 반환
        return []
    
    # 날짜별로 정렬
    log_files.sort(key=lambda x: x['file_name'], reverse=True)
    return log_files

# 로그 내용 API
@app.route('/api/logs/content')
def get_log_content():
    """로그 내용 API"""
    file_path = request.args.get('file', '')
    
    if not file_path:
        return jsonify({'success': False, 'message': '파일 경로가 지정되지 않았습니다.'})
    
    # 로그 디렉토리 경로
    log_dir = os.path.abspath('logs')
    
    try:
        # 파일명만 추출
        file_name = os.path.basename(file_path)
        
        # 안전한 경로 구성
        safe_path = os.path.join(log_dir, file_name)
        
        # 파일 존재 확인
        if not os.path.exists(safe_path):
            return jsonify({
                'success': False, 
                'message': f'파일을 찾을 수 없습니다: {file_name}'
            })
        
        # 로그 파일 읽기
        try:
            with open(safe_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            return jsonify({'success': True, 'content': content})
        except UnicodeDecodeError:
            # UTF-8 디코딩 실패 시 다른 인코딩 시도
            try:
                with open(safe_path, 'r', encoding='cp949', errors='replace') as f:
                    content = f.read()
                return jsonify({'success': True, 'content': content})
            except:
                return jsonify({
                    'success': False, 
                    'message': '파일 인코딩을 처리할 수 없습니다.'
                })
    except Exception as e:
        logger.error(f"로그 파일 읽기 오류: {str(e)}")
        return jsonify({
            'success': False, 
            'message': f'로그 파일을 읽는 중 오류가 발생했습니다: {str(e)}'
        })

# 로그 파일 다운로드 API
@app.route('/api/logs/download')
def download_log():
    """로그 파일 다운로드"""
    file_path = request.args.get('file', '')
    
    if not file_path:
        return jsonify({'success': False, 'message': '파일 경로가 지정되지 않았습니다.'})
    
    # 로그 디렉토리 경로
    log_dir = os.path.abspath('logs')
    
    try:
        # 파일명만 추출
        file_name = os.path.basename(file_path)
        
        # 안전한 경로 구성
        safe_path = os.path.join(log_dir, file_name)
        
        # 파일 존재 확인
        if not os.path.exists(safe_path):
            return jsonify({
                'success': False, 
                'message': f'파일을 찾을 수 없습니다: {file_name}'
            })
        
        return send_file(safe_path, as_attachment=True, download_name=file_name)
    except Exception as e:
        logger.error(f"로그 파일 다운로드 오류: {str(e)}")
        return jsonify({
            'success': False, 
            'message': f'로그 파일 다운로드 중 오류가 발생했습니다: {str(e)}'
        })

# API 테스트 엔드포인트
@app.route('/api/logs/test')
def test_logs_api():
    """로그 API 테스트"""
    log_dir = os.path.abspath('logs')
    log_files = []
    
    try:
        # 디렉토리 내 .log 파일 수 확인
        if os.path.exists(log_dir) and os.path.isdir(log_dir):
            log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'로그 디렉토리 접근 오류: {str(e)}',
            'log_dir': log_dir,
            'error': str(e)
        })
    
    return jsonify({
        'success': True,
        'message': 'API가 정상적으로 작동합니다.',
        'log_dir': log_dir,
        'log_files_count': len(log_files),
        'log_files': log_files[:5]  # 최대 5개 파일명만 반환
    })

# 직접 파일 읽기 라우트
@app.route('/logs/raw/<path:filename>')
def raw_log_file(filename):
    """직접 로그 파일 읽기"""
    log_dir = os.path.abspath('logs')
    
    # 파일 경로 검증
    safe_filename = os.path.basename(filename)  # 파일명만 추출
    file_path = os.path.join(log_dir, safe_filename)
    
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return f"파일을 찾을 수 없습니다: {safe_filename}", 404
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # 텍스트 파일 형식으로 응답
        resp = app.response_class(
            response=content,
            status=200,
            mimetype='text/plain'
        )
        return resp
    except Exception as e:
        return f"파일 읽기 오류: {str(e)}", 500
    

@app.route('/api/stocks/history')
def get_stocks_history():
    """종목 선정 기록 조회"""
    try:
        # 기록 저장 디렉토리
        history_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history")
        
        # 모든 기록 파일
        all_history_file = os.path.join(history_dir, "all_selected_stocks.csv")
        
        if not os.path.exists(all_history_file):
            return jsonify({'success': False, 'message': '기록 파일이 없습니다.'})
        
        # CSV 파일 읽기
        history_data = []
        with open(all_history_file, 'r', encoding='utf-8') as f:
            import csv
            reader = csv.DictReader(f)
            for row in reader:
                history_data.append(row)
        
        return jsonify({'success': True, 'history': history_data})
    except Exception as e:
        logger.error(f"종목 선정 기록 조회 중 오류: {str(e)}")
        return jsonify({'success': False, 'message': f'오류 발생: {str(e)}'})
    

@socketio.on('selected_stocks_update')
def handle_stocks_update(data):
    if not data or not Array.isArray(data):
        return
    
    # 여기서 신호도 함께 계산하여 전송
    updated_data = []
    for stock in data:
        # 실시간 신호 계산 (백엔드와 동일한 로직 사용)
        signal_info = calculate_trading_signal(stock_code=stock['code'])
        
        # 신호 정보 추가
        stock['signal'] = signal_info['signal']
        stock['score'] = signal_info['score']
        updated_data.append(stock)
    
    socketio.emit('selected_stocks_update', updated_data)

# app.py 파일에 추가 (라우트 섹션)
@app.route('/api/debug/signal/<stock_code>')
def debug_signal(stock_code):
    """신호 계산 디버깅 API"""
    try:
        # signal_utils 임포트
        from src.utils.signal_utils import get_trading_signal
        
        # 1. 백엔드 전략으로 계산한 신호
        backend_signal = None
        if hasattr(trading_system.strategy, 'analyze_stock'):
            backend_signal = trading_system.strategy.analyze_stock(stock_code)
        
        # 2. 통합 유틸리티로 계산한 신호
        utils_signal = get_trading_signal(trading_system.market_data, stock_code)
        
        # 3. 일봉 데이터 조회
        try:
            daily_data = trading_system.market_data.get_stock_daily_price(stock_code, period=20)
            daily_json = None
            if not daily_data.empty:
                daily_json = daily_data.to_dict('records')
        except:
            daily_json = None
        
        # 4. 현재가 데이터 조회
        current_data = trading_system.market_data.get_stock_current_price(stock_code)
        
        # 5. 차이점 분석
        differences = []
        if backend_signal and backend_signal.get('signal') != utils_signal.get('signal'):
            differences.append(f"신호 불일치: 백엔드={backend_signal.get('signal')}, 유틸리티={utils_signal.get('signal')}")
        
        # 종목 정보 (UI에 표시되는 정보)
        ui_info = None
        if hasattr(trading_system.strategy, 'stock_scores') and stock_code in trading_system.strategy.stock_scores:
            score = trading_system.strategy.stock_scores[stock_code]
            ui_info = {
                'score': score,
                # 추가 UI 관련 정보...
            }
        
        # 비교 정보 반환
        result = {
            'backend_signal': backend_signal,
            'utils_signal': utils_signal,
            'differences': differences,
            'current_data': current_data,
            'daily_data': daily_json,
            'ui_info': ui_info,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 로깅
        logger.debug(f"디버그 신호 API ({stock_code}): 백엔드={backend_signal.get('signal', 'N/A')}, 유틸리티={utils_signal.get('signal', 'N/A')}")
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"디버그 신호 API 오류: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)})

# 추가 디버깅 API - 대시보드 데이터 비교
@app.route('/api/debug/dashboard')
def debug_dashboard():
    """대시보드 데이터 디버깅 API"""
    try:
        # 실제 백엔드에서 계산된 데이터
        backend_data = {
            'positions': trading_system.strategy.positions if hasattr(trading_system.strategy, 'positions') else {},
            'selected_stocks': trading_system.strategy.selected_stocks if hasattr(trading_system.strategy, 'selected_stocks') else [],
            'stock_scores': trading_system.strategy.stock_scores if hasattr(trading_system.strategy, 'stock_scores') else {}
        }
        
        # 대시보드에 표시되는 데이터
        dashboard_data = {}
        try:
            # 대시보드 API 호출
            dashboard_data = trading_system.get_dashboard_data() if hasattr(trading_system, 'get_dashboard_data') else {}
        except:
            dashboard_data = {"error": "대시보드 데이터 조회 실패"}
        
        # 차이점 분석
        differences = []
        
        # 예: 선정된 종목 수 비교
        backend_stock_count = len(backend_data['selected_stocks'])
        dashboard_stock_count = len(dashboard_data.get('stocks', []))
        
        if backend_stock_count != dashboard_stock_count:
            differences.append(f"선정된 종목 수 불일치: 백엔드={backend_stock_count}, 대시보드={dashboard_stock_count}")
        
        # 결과 반환
        return jsonify({
            'backend_data': backend_data,
            'dashboard_data': dashboard_data,
            'differences': differences,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        logger.error(f"대시보드 디버그 API 오류: {str(e)}", exc_info=True)
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