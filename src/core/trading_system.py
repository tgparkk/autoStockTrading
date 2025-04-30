import os
import sys
import yaml
import json
import logging
import threading
import time
from datetime import datetime, timedelta
import glob
import importlib

from src.strategy.factory import StrategyFactory
from src.ml.model import StockPredictionModel
from src.ml.training import train_model
from src.utils.data_utils import calculate_moving_average, calculate_rsi, calculate_bollinger_bands

logger = logging.getLogger(__name__)

class TradingSystem:
    """자동매매 시스템 래퍼 클래스
    
    이 클래스는 자동 주식 거래 시스템의 중앙 컨트롤러 역할을 합니다.
    API 클라이언트, 시장 데이터, 주문 API, 전략을 통합하여 거래를 실행합니다.
    
    Attributes:
        auth (KoreaInvestmentAuth): 인증 객체
        market_data (MarketData): 시장 데이터 객체
        order_api (OrderAPI): 주문 API 객체
        strategy (Strategy): 전략 객체
        config_manager (ConfigManager): 설정 관리 객체
        is_running (bool): 시스템 실행 상태
        target_stocks (list): 대상 종목 리스트
        positions (dict): 보유 종목 정보
    """
    
    def __init__(self, auth_client, market_data, order_api, config_manager, strategy_type="basic"):
        """시스템 초기화
        
        Args:
            auth_client (KoreaInvestmentAuth): 인증 클라이언트
            market_data (MarketData): 시장 데이터 객체
            order_api (OrderAPI): 주문 API 객체
            config_manager (ConfigManager): 설정 관리 객체
            strategy_type (str, optional): 전략 유형 ('basic', 'day_trading', 'high_frequency', 'ml_high_frequency')
        """
        # 로그 파일 경로 설정
        log_file = 'logs/trading_system.log'
        self.setup_logger(log_file)
        
        logger.info("Trading system initializing...")
        
        # 의존성 설정
        self.auth = auth_client
        self.market_data = market_data
        self.order_api = order_api
        self.config_manager = config_manager
        
        # 상태 관리
        self.is_running = False
        self.trading_thread = None
        
        # 종목 및 데이터 캐시
        self.target_stocks = config_manager.load_target_stocks()
        self.current_data = {}
        self.account_info = {}
        
        # ML 모델 로드 또는 학습
        self.ml_model = None
        self._load_or_train_model()
        
        # 전략 객체 생성
        self.strategy = None
        self._create_strategy(strategy_type)
        
        logger.info("Trading system initialized successfully")
    
    def setup_logger(self, log_file):
        """로거 설정
        
        Args:
            log_file (str): 로그 파일 경로
        """
        # 디렉토리 확인
        log_dir = os.path.dirname(os.path.abspath(log_file))
        os.makedirs(log_dir, exist_ok=True)
        
        # 로거 설정
        global logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # 파일 핸들러 설정
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 콘솔 핸들러 설정
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 포맷 설정
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 로거에 핸들러 추가
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    def _create_strategy(self, strategy_type):
        """전략 객체 생성
        
        Args:
            strategy_type (str): 전략 유형 ('basic', 'day_trading', 'high_frequency', 'ml_high_frequency')
        """
        strategy_config = self.config_manager.get_strategy_config()
        self.strategy = StrategyFactory.create_strategy(
            strategy_type=strategy_type, 
            market_data=self.market_data, 
            order_api=self.order_api, 
            config=strategy_config,
            ml_model=self.ml_model
        )
        logger.info(f"Strategy created: {strategy_type}")
    
    def _load_or_train_model(self):
        """ML 모델 로드 또는 학습"""
        model = StockPredictionModel()
        try:
            # 최신 모델 파일 찾기
            model_files = glob.glob(os.path.join("models", "stock_model_*.pkl"))
            
            if model_files:
                # 가장 최근 모델 로드
                latest_model = max(model_files)
                model.load(os.path.basename(latest_model))
                logger.info(f"ML 모델 로드됨: {latest_model}")
                self.ml_model = model
            else:
                # 모델 신규 학습
                logger.info("기존 모델이 없습니다. 새로운 모델을 학습합니다.")
                self.ml_model = train_model(self.market_data, self.target_stocks, days=300)
        except Exception as e:
            logger.error(f"ML 모델 로드/학습 실패: {str(e)}")
    
    def start(self):
        """매매 시작"""
        if self.is_running:
            logger.warning("거래 시스템이 이미 실행 중입니다.")
            return
        
        self.is_running = True
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        logger.info("Trading system started")
    
    def stop(self):
        """매매 중지"""
        if not self.is_running:
            logger.warning("거래 시스템이 이미 중지되었습니다.")
            return
            
        self.is_running = False
        logger.info("Trading system stopping...")
        
        # 스레드가 종료될 때까지 최대 10초 대기
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=10)
            
        logger.info("Trading system stopped")
    
    def _trading_loop(self):
        """매매 루프"""
        last_update_time = datetime.now()
        stock_update_interval = 10 * 60  # 10분마다 종목 갱신
        
        while self.is_running:
            try:
                # 토큰 갱신 확인
                self.auth.get_access_token()
                
                # 거래 시간 체크
                if self._is_trading_time():
                    # 주기적 종목 갱신
                    current_time = datetime.now()
                    if (current_time - last_update_time).total_seconds() > stock_update_interval:
                        logger.info("주기적 종목 리스트 갱신 시작")
                        
                        # 전략의 주간 업데이트 메서드 호출 (있는 경우)
                        if hasattr(self.strategy, 'weekly_update'):
                            self.strategy.weekly_update()
                            
                            # 전략이 종목을 선정한 경우 업데이트
                            if hasattr(self.strategy, 'selected_stocks'):
                                self.target_stocks = self.strategy.selected_stocks
                                
                                # 설정 파일에 저장
                                self.config_manager.save_target_stocks(self.target_stocks)
                                
                                logger.info(f"종목 리스트 갱신 완료: {len(self.target_stocks)}개 종목")
                        
                        last_update_time = current_time
                    
                    # 전략 실행에 사용할 종목 선택
                    stocks_to_use = []
                    if hasattr(self.strategy, 'selected_stocks') and self.strategy.selected_stocks:
                        stocks_to_use = self.strategy.selected_stocks
                    elif self.target_stocks:
                        stocks_to_use = self.target_stocks
                    
                    # 종목이 없는 경우 종목 선정 시도
                    if not stocks_to_use:
                        logger.warning("선정된 종목이 없습니다. 종목 선정을 시도합니다.")
                        if hasattr(self.strategy, 'weekly_update'):
                            self.strategy.weekly_update()
                            if hasattr(self.strategy, 'selected_stocks'):
                                stocks_to_use = self.strategy.selected_stocks
                                self.target_stocks = self.strategy.selected_stocks
                    
                    # 전략 실행
                    if stocks_to_use:
                        # 전략 유형에 따라 실행 방식 결정
                        if self.strategy.__class__.__name__ == 'HighFrequencyStrategy':
                            logger.info("고빈도 트레이딩 전략 실행 중...")
                            results = self.strategy.run()
                        else:
                            logger.info(f"{len(stocks_to_use)}개 종목으로 전략 실행 중")
                            results = self.strategy.run(stocks_to_use)
                        
                        # 결과 처리 및 로깅
                        self._handle_trading_results(results)
                    else:
                        logger.warning("선정된 종목이 없어 전략을 실행할 수 없습니다.")
                    
                    # 캐시 데이터 업데이트
                    self._update_cache()
                else:
                    logger.info("거래 시간이 아닙니다. 대기 중...")
                
                # 전략 유형에 따라 다른 대기 시간 적용
                if self.strategy.__class__.__name__ == 'HighFrequencyStrategy':
                    # 고빈도 전략은 더 짧은 간격(1분)으로 실행
                    time.sleep(60)
                else:
                    # 일반 전략은 설정된 간격으로 실행
                    interval_seconds = self.config_manager.get_interval() * 60
                    time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"매매 루프 오류: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                time.sleep(60)  # 오류 시 1분 대기
    
    def _handle_trading_results(self, results):
        """거래 결과 처리
        
        Args:
            results (dict): 전략 실행 결과
        """
        buys_count = len(results.get('buys', []))
        sells_count = len(results.get('sells', []))
        errors_count = len(results.get('errors', []))
        
        logger.info(f"매수: {buys_count}건, 매도: {sells_count}건, 오류: {errors_count}건")
        
        # 매수 결과 로깅
        if buys_count > 0:
            for buy in results['buys']:
                logger.info(f"매수 실행: {buy['stock_code']} - {buy.get('reason', '신호 없음')}")
        
        # 매도 결과 로깅
        if sells_count > 0:
            for sell in results['sells']:
                logger.info(f"매도 실행: {sell['stock_code']} - {sell.get('reason', '신호 없음')}")
        
        # 오류 로깅
        if errors_count > 0:
            for error in results['errors']:
                logger.error(f"오류 발생: {error.get('stock_code', 'N/A')} - {error.get('error', '알 수 없는 오류')}")
    
    def _is_trading_time(self):
        """거래 시간 여부 확인
        
        Returns:
            bool: 거래 시간이면 True, 아니면 False
        """
        now = datetime.now()
        
        # 주말 체크
        if now.weekday() >= 5:  # 토요일(5), 일요일(6)
            return False
        
        # 시간 체크 (9:00 ~ 15:30)
        market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def _update_cache(self):
        """캐시 데이터 업데이트"""
        try:
            # 계좌 정보 업데이트
            account_data = self.market_data.get_account_balance()
            if account_data:
                logger.debug(f"계좌 정보 API 응답: {account_data}")
                self.account_info = account_data
            else:
                logger.warning("계좌 정보를 가져오지 못했습니다.")
            
            # 현재 종목 데이터 업데이트
            for stock_code in self.target_stocks:
                data = self.market_data.get_stock_current_price(stock_code)
                if data:
                    self.current_data[stock_code] = data
        except Exception as e:
            logger.error(f"캐시 업데이트 중 오류: {str(e)}")
    
    def get_current_stock_data(self):
        """현재 종목 데이터 반환
        
        Returns:
            dict: 종목 코드별 현재 데이터
        """
        # 데이터가 없으면 업데이트
        if not self.current_data:
            self._update_cache()
        return self.current_data
    
    def get_account_info(self):
        """계좌 정보 반환
        
        Returns:
            dict: 계좌 정보
        """
        try:
            # 정보가 없으면 업데이트
            if not self.account_info:
                self._update_cache()

            # 장 외 시간에는 더미 데이터 반환 (테스트 용)
            if not self._is_trading_time() and not self.account_info:
                dummy_data = {
                    'account_summary': [{
                        'dnca_tot_amt': '500000',  # 예수금
                        'scts_evlu_amt': '500000',  # 주식 평가금액
                        'tot_evlu_amt': '1000000',  # 총 평가금액
                        'pchs_amt_smtl_amt': '450000',  # 매입금액
                        'evlu_pfls_smtl_amt': '50000',  # 평가손익
                        'asst_icdc_erng_rt': '10.00'  # 수익률
                    }],
                    'stocks': []
                }
                return dummy_data
            
            # 계좌 정보 구조 조정
            result = {'account_summary': [], 'stocks': []}
            
            # API 응답 구조에 맞게 데이터 정리
            if (self.account_info and 'account_summary' in self.account_info 
                and not self.account_info['account_summary'] 
                and 'stocks' in self.account_info 
                and len(self.account_info['stocks']) > 0):
                # 첫 번째 항목을 account_summary로 이동
                account_summary_item = self.account_info['stocks'][0].copy()
                result['account_summary'] = [account_summary_item]
                
                # stocks가 실제 주식 항목인지 확인 (첫 번째 항목은 계좌 요약이므로 제외)
                if len(self.account_info['stocks']) > 1:
                    result['stocks'] = self.account_info['stocks'][1:]
            else:
                # 원래 구조 유지
                result = self.account_info
            
            # 보유 종목에 종목명 추가
            if result and 'stocks' in result:
                for stock in result['stocks']:
                    # 현재가 정보가 없으면 추가
                    if 'prpr' not in stock and 'pdno' in stock and stock.get('pdno') in self.current_data:
                        current_stock = self.current_data[stock['pdno']]
                        stock['prpr'] = current_stock.get('stck_prpr', '0')
                        stock['prdt_name'] = current_stock.get('prdt_name', '알 수 없음')
            
            logger.debug(f"조정된 계좌 정보: {result}")
            return result
        except Exception as e:
            logger.error(f"계좌 정보 조회 중 오류: {str(e)}")
            return {'account_summary': [], 'stocks': []}  # 오류 시 빈 구조 반환
    
    def get_strategy_config(self):
        """전략 설정 반환
        
        Returns:
            dict: 현재 적용된 전략 설정
        """
        return self.strategy.config
    
    def update_strategy_config(self, config_updates):
        """전략 설정 업데이트
        
        Args:
            config_updates (dict): 업데이트할 설정
            
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            # 전략 객체의 설정 업데이트
            for key, value in config_updates.items():
                self.strategy.config[key] = value
            
            # 설정 파일 저장
            success = self.config_manager.update_strategy_config(self.strategy.config)
            
            if success:
                logger.info("전략 설정이 업데이트되었습니다.")
            else:
                logger.warning("전략 설정 저장에 실패했습니다.")
            
            return success
        except Exception as e:
            logger.error(f"전략 설정 업데이트 중 오류: {str(e)}")
            return False
    
    def get_target_stocks(self):
        """대상 종목 목록 반환
        
        Returns:
            list: 대상 종목 코드 리스트
        """
        return self.target_stocks
    
    def update_target_stocks(self, stocks):
        """대상 종목 업데이트
        
        Args:
            stocks (list): 업데이트할 종목 코드 리스트
            
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            # 종목 리스트 업데이트
            self.target_stocks = stocks
            
            # 설정 파일 저장
            success = self.config_manager.save_target_stocks(stocks)
            
            if success:
                logger.info(f"대상 종목이 업데이트되었습니다: {len(stocks)}개 종목")
            else:
                logger.warning("대상 종목 저장에 실패했습니다.")
            
            return success
        except Exception as e:
            logger.error(f"대상 종목 업데이트 중 오류: {str(e)}")
            return False
    
    def get_status(self):
        """시스템 상태 반환
        
        Returns:
            str: 시스템 상태 ('running', 'waiting', 'stopped')
        """
        if self.is_running:
            return "running" if self._is_trading_time() else "waiting"
        return "stopped"
    
    def get_recent_logs(self, count=10):
        """최근 로그 반환
        
        Args:
            count (int, optional): 가져올 로그 수. 기본값은 10.
            
        Returns:
            list: 최근 로그 리스트
        """
        logs = []
        try:
            with open('logs/trading_system.log', 'r', encoding='utf-8') as f:
                logs = f.readlines()
            return logs[-count:] if count < len(logs) else logs
        except Exception as e:
            logger.error(f"로그 파일 읽기 오류: {str(e)}")
            return []
    
    def get_stock_detail(self, stock_code, days=30):
        """종목 상세 정보 반환
        
        Args:
            stock_code (str): 종목 코드
            days (int, optional): 분석 기간(일). 기본값은 30.
            
        Returns:
            dict: 종목 상세 정보
        """
        try:
            # 일별 데이터 조회
            df = self.market_data.get_stock_daily_price(stock_code, period=days)
            
            if df.empty:
                return {'error': '데이터가 없습니다.'}
            
            # 이동평균선 계산
            df = calculate_moving_average(df)
            
            # RSI 계산
            df = calculate_rsi(df)
            
            # 볼린저 밴드 계산
            df = calculate_bollinger_bands(df)
            
            # JSON 변환을 위한 날짜 형식 변환
            df['date'] = df['stck_bsop_date'].dt.strftime('%Y-%m-%d')
            
            # 분석 결과
            analysis = self.strategy.analyze_stock(stock_code)
            
            # 결과 반환
            return {
                'code': stock_code,
                'data': df.to_dict('records'),
                'analysis': analysis,
                'current': self.current_data.get(stock_code, {})
            }
        except Exception as e:
            logger.error(f"종목 상세 정보 조회 중 오류: {str(e)}")
            return {'error': str(e)}
    
    def retrain_model(self):
        """ML 모델 재학습
        
        Returns:
            bool: 재학습 성공 여부
        """
        try:
            self.ml_model = train_model(self.market_data, self.target_stocks, days=300)
            logger.info("ML 모델 재학습 완료")
            return True
        except Exception as e:
            logger.error(f"모델 재학습 실패: {str(e)}")
            return False
    
    def get_ml_model_info(self):
        """ML 모델 정보 반환
        
        Returns:
            dict: ML 모델 정보
        """
        if not self.ml_model:
            return {
                'model_type': 'None',
                'last_training': 'Not available',
                'accuracy': 0.0,
                'f1_score': 0.0,
                'feature_importance': {
                    'labels': ['RSI', '볼린저밴드', 'MACD', '이동평균선', '거래량변화'],
                    'values': [0.2, 0.2, 0.2, 0.2, 0.2]
                },
                'performance_history': {
                    'dates': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(5, 0, -1)],
                    'accuracy': [0.65, 0.66, 0.67, 0.68, 0.69],
                    'f1_score': [0.62, 0.63, 0.64, 0.65, 0.66]
                }
            }
        
        try:
            # 개선된 get_model_info 메서드 호출
            return self.ml_model.get_model_info()
        except Exception as e:
            logger.error(f"ML 모델 정보 수집 중 오류: {str(e)}")
            return {
                'model_type': 'Error',
                'last_training': 'Error',
                'accuracy': 0.0,
                'f1_score': 0.0,
                'feature_importance': None,
                'performance_history': None
            }
    
    def save_selected_stocks_history(self, stocks_info):
        """선정된 종목 기록 저장
        
        Args:
            stocks_info (list): 종목 정보 리스트 (dicts)
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 기록 저장 디렉토리
            history_dir = "history"
            os.makedirs(history_dir, exist_ok=True)
            
            # 파일명은 날짜로 생성 (YYYYMMDD.csv)
            today = datetime.now().strftime('%Y%m%d')
            history_file = os.path.join(history_dir, f"selected_stocks_{today}.csv")
            
            # 종목 데이터 저장
            with open(history_file, 'w', encoding='utf-8') as f:
                # 헤더 추가
                f.write("선정일자,종목코드,종목명,선정점수\n")
                
                # 데이터 추가
                for stock in stocks_info:
                    selected_date = stock.get('selected_date', today)
                    stock_code = stock.get('code', '')
                    stock_name = stock.get('name', '')
                    score = stock.get('score', '')
                    
                    # 점수가 None인 경우 빈 문자열로 처리
                    if score is None:
                        score = ''
                    
                    f.write(f"{selected_date},{stock_code},{stock_name},{score}\n")
            
            # 통합 기록 파일에도 추가
            all_history_file = os.path.join(history_dir, "all_selected_stocks.csv")
            
            # 파일이 없으면 헤더 추가
            if not os.path.exists(all_history_file):
                with open(all_history_file, 'w', encoding='utf-8') as f:
                    f.write("선정일자,종목코드,종목명,선정점수\n")
            
            # 기존 파일에 데이터 추가 (append)
            with open(all_history_file, 'a', encoding='utf-8') as f:
                for stock in stocks_info:
                    selected_date = stock.get('selected_date', today)
                    stock_code = stock.get('code', '')
                    stock_name = stock.get('name', '')
                    score = stock.get('score', '')
                    
                    if score is None:
                        score = ''
                    
                    f.write(f"{selected_date},{stock_code},{stock_name},{score}\n")
            
            logger.info(f"종목 선정 기록이 저장되었습니다: {history_file}")
            return True
        except Exception as e:
            logger.error(f"종목 선정 기록 저장 중 오류: {str(e)}")
            return False
