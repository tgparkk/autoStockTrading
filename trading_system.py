import os
import sys
import yaml
import json
import logging
import threading
import time
from datetime import datetime
from ruamel.yaml import YAML

from src.ml.model import StockPredictionModel
from src.ml.features import create_features
from src.ml.training import train_model

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 기존 자동매매 시스템 모듈 임포트
from src.api.auth import KoreaInvestmentAuth
from src.api.market_data import MarketData
from src.api.order import OrderAPI
from src.strategy.basic_strategy import BasicStrategy
from src.utils.logger import setup_logger
from src.utils.data_utils import calculate_moving_average, calculate_rsi, calculate_bollinger_bands

class TradingSystem:
    """자동매매 시스템 래퍼 클래스"""
    
    def __init__(self, config_path='config/api_config.yaml', 
                strategy_path='config/trading_config.yaml',
                stocks_path='config/target_stocks.txt'):
        self.logger = setup_logger('logs/trading_system.log')
        self.logger.info("Trading system initializing...")
        
        # 설정 파일 경로
        self.config_path = os.path.abspath(config_path)
        self.strategy_path = os.path.abspath(strategy_path)
        self.stocks_path = os.path.abspath(stocks_path)
        
        # API 객체
        self.auth = None
        self.market_data = None
        self.order_api = None
        
        # 전략 객체
        self.strategy = None
        
        # 상태 관리
        self.is_running = False
        self.trading_thread = None
        
        # 종목 및 데이터 캐시
        self.target_stocks = []
        self.current_data = {}
        self.account_info = {}
        self.recent_logs = []
        
        # 초기화
        self._initialize()
    
    def _initialize(self):
        """시스템 초기화"""
        try:
            # 토큰 발급
            self.auth = KoreaInvestmentAuth(self.config_path)
            
            # API 객체 초기화
            self.market_data = MarketData(self.auth, self.config_path)
            self.order_api = OrderAPI(self.auth, self.config_path)
            
            # 전략 설정 로드
            strategy_config = {}
            if os.path.exists(self.strategy_path):
                with open(self.strategy_path, 'r', encoding='utf-8') as f:
                    strategy_config = yaml.safe_load(f).get('strategy', {})
            
            # 전통적 전략 대신 통합 전략 사용
            from src.ml.integrated_strategy import IntegratedStrategy
            self.strategy = IntegratedStrategy(self.market_data, self.order_api, strategy_config)
            
            # 주간 업데이트를 위한 스케줄 설정
            self._setup_weekly_update()
            
            ## 대상 종목 로드
            ##self._load_target_stocks()

            # ML 모델 로드
            self.ml_model = None
            self._load_or_train_model()
            
            self.logger.info("Trading system initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing trading system: {str(e)}")
            raise
    
    def _load_target_stocks(self):
        """타겟 종목 로드"""
        if os.path.exists(self.stocks_path):
            with open(self.stocks_path, 'r', encoding='utf-8') as f:
                self.target_stocks = [line.strip() for line in f if line.strip() 
                                     and not line.strip().startswith('#')]
            self.logger.info(f"Loaded {len(self.target_stocks)} target stocks")
    
    def start(self):
        """매매 시작"""
        if self.is_running:
            return
        
        self.is_running = True
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        self.logger.info("Trading system started")
    
    def stop(self):
        """매매 중지"""
        self.is_running = False
        self.logger.info("Trading system stopped")
    
    def _trading_loop(self):
        """매매 루프"""
        while self.is_running:
            try:
                # 토큰 갱신 확인
                self.auth.get_access_token()
                
                # 거래 시간 체크
                if self._is_trading_time():
                    # 전략 실행
                    results = self.strategy.run(self.target_stocks)
                    
                    # 결과 로깅
                    self.logger.info(f"Strategy execution - Buys: {len(results['buys'])}, "
                                   f"Sells: {len(results['sells'])}, Errors: {len(results['errors'])}")
                    
                    # 캐시 업데이트
                    self._update_cache()
                else:
                    self.logger.info("Not in trading hours. Waiting...")
                
                # 설정된 간격만큼 대기 (분 단위를 초 단위로 변환)
                interval_seconds = self.get_interval() * 60
                time.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(60)  # 오류 발생 시 1분 후 재시도
    
    def _is_trading_time(self):
        """거래 시간 여부 확인"""
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
                self.logger.info(f"원본 API 응답: {account_data}")
                self.account_info = account_data
            else:
                self.logger.warning("계좌 정보를 가져오지 못했습니다.")
            
            # 현재 종목 데이터 업데이트
            for stock_code in self.target_stocks:
                data = self.market_data.get_stock_current_price(stock_code)
                if data:
                    self.current_data[stock_code] = data
        except Exception as e:
            self.logger.error(f"Error updating cache: {str(e)}")
    
    def get_current_stock_data(self):
        """현재 종목 데이터 반환"""
        # 데이터가 없으면 업데이트
        if not self.current_data:
            self._update_cache()
        return self.current_data
    
    def get_account_info(self):
        """계좌 정보 반환"""
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
            
            # 계좌 정보 구조 조정: API 응답 구조에 맞게 조정
            result = {'account_summary': [], 'stocks': []}
            
            # account_summary가 비어있고 stocks에 계좌 요약 정보가 있는 경우 (로그에서 확인된 구조)
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
            
            # 로깅 (프로덕션에서는 삭제하거나 디버그 레벨로 변경)
            self.logger.info(f"조정된 계좌 정보: {result}")
            
            return result
        except Exception as e:
            self.logger.error(f"계좌 정보 조회 중 오류: {str(e)}")
            return {'account_summary': [], 'stocks': []}  # 오류 시 빈 구조 반환
    
    def get_strategy_config(self):
        """전략 설정 반환"""
        return self.strategy.config
    
    def update_strategy_config(self, config):
        """전략 설정 업데이트"""
        # 설정 업데이트
        for key, value in config.items():
            self.strategy.config[key] = value
        
        # 설정 파일 저장 (주석 유지)
        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.indent(mapping=2, sequence=4, offset=2)
        
        try:
            with open(self.strategy_path, 'r', encoding='utf-8') as f:
                strategy_yaml = yaml.load(f)
            
            # 전략 설정만 업데이트
            strategy_yaml['strategy'] = self.strategy.config
            
            with open(self.strategy_path, 'w', encoding='utf-8') as f:
                yaml.dump(strategy_yaml, f)
            
            self.logger.info("Strategy configuration updated")
        except Exception as e:
            self.logger.error(f"Error updating strategy config: {str(e)}")
            raise
    
    def get_target_stocks(self):
        """타겟 종목 목록 반환"""
        return self.target_stocks
    
    def update_target_stocks(self, stocks):
        """타겟 종목 업데이트"""
        self.target_stocks = stocks
        
        # 파일에 저장
        with open(self.stocks_path, 'w', encoding='utf-8') as f:
            for stock in stocks:
                f.write(f"{stock}\n")
        
        self.logger.info(f"Target stocks updated: {len(stocks)} stocks")
    
    def get_status(self):
        """시스템 상태 반환"""
        if self.is_running:
            return "running" if self._is_trading_time() else "waiting"
        return "stopped"
    
    def get_recent_logs(self, count=10):
        """최근 로그 반환"""
        logs = []
        try:
            with open('logs/trading_system.log', 'r', encoding='utf-8') as f:
                logs = f.readlines()
            return logs[-count:] if count < len(logs) else logs
        except Exception:
            return []
    
    def get_stock_detail(self, stock_code, days=30):
        """종목 상세 정보 반환"""
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
            self.logger.error(f"Error getting stock detail: {str(e)}")
            return {'error': str(e)}
        
    def get_interval(self):
        """작업 실행 간격 반환"""
        # 설정 파일에서 interval 값 읽기
        try:
            with open(self.strategy_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config.get('interval', 2)  # 기본값 2분
        except Exception as e:
            self.logger.error(f"Error reading interval: {str(e)}")
            return 2  # 기본값 반환

    def _save_config_with_comments(self, config):
        """주석이 유지되는 방식으로 설정 파일 저장"""
        try:
            # 1. 기존 파일 내용을 라인 단위로 읽기
            with open(self.strategy_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 2. 줄 단위로 설정 값 업데이트
            updated_lines = []
            in_strategy_section = False
            strategy_indent = ""
            
            for line in lines:
                # 빈 줄이나 주석은 그대로 유지
                if line.strip() == "" or line.strip().startswith('#'):
                    updated_lines.append(line)
                    continue
                
                # strategy 섹션 시작 감지
                if line.strip() == "strategy:" or line.strip().startswith("strategy:"):
                    in_strategy_section = True
                    strategy_indent = " " * (line.find("strategy:"))
                    updated_lines.append(line)
                    continue
                
                # 다른 최상위 섹션 시작 감지
                if ":" in line and not line.startswith(" ") and not line.startswith("\t"):
                    if line.split(":")[0].strip() != "strategy":
                        in_strategy_section = False
                
                # strategy 섹션 내부의 값들 업데이트
                if in_strategy_section and ":" in line:
                    key = line.split(":")[0].strip()
                    if key in config['strategy']:
                        # 주석 유지
                        comment = ""
                        if "#" in line:
                            comment = " " + line.split("#", 1)[1].rstrip("\n")
                        
                        # 들여쓰기 유지
                        indent = ""
                        for char in line:
                            if char == " " or char == "\t":
                                indent += char
                            else:
                                break
                        
                        # 새 라인 구성
                        updated_lines.append(f"{indent}{key}: {config['strategy'][key]}{comment}\n")
                        continue
                
                # interval 값 업데이트
                if not in_strategy_section and ":" in line:
                    key = line.split(":")[0].strip()
                    if key == "interval" and "interval" in config:
                        # 주석 유지
                        comment = ""
                        if "#" in line:
                            comment = " " + line.split("#", 1)[1].rstrip("\n")
                        
                        # 들여쓰기 유지
                        indent = ""
                        for char in line:
                            if char == " " or char == "\t":
                                indent += char
                            else:
                                break
                        
                        updated_lines.append(f"{indent}interval: {config['interval']}{comment}\n")
                        continue
                
                # 그 외 라인은 그대로 유지
                updated_lines.append(line)
            
            # 3. 파일에 다시 쓰기
            with open(self.strategy_path, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)
            
            self.logger.info("Configuration updated with comments preserved")
            return True
        except Exception as e:
            self.logger.error(f"Error saving config with comments: {str(e)}")
            return False

    def update_strategy_config(self, config_updates):
        """전략 설정 업데이트 (주석 유지)"""
        # 현재 설정 복사
        config = {'strategy': self.strategy.config.copy()}
        
        # 업데이트된 값만 변경
        for key, value in config_updates.items():
            config['strategy'][key] = value
        
        # 전략 객체의 설정도 업데이트
        for key, value in config_updates.items():
            self.strategy.config[key] = value
        
        # 주석이 유지되는 방식으로 저장
        if not self._save_config_with_comments(config):
            # 실패 시 기존 방식으로 저장 시도
            try:
                with open(self.strategy_path, 'r', encoding='utf-8') as f:
                    full_config = yaml.safe_load(f)
                
                full_config['strategy'] = config['strategy']
                
                with open(self.strategy_path, 'w', encoding='utf-8') as f:
                    yaml.dump(full_config, f, default_flow_style=False)
                
                self.logger.warning("Configuration updated but comments were lost")
            except Exception as e:
                self.logger.error(f"Error updating strategy config: {str(e)}")
                raise
        
        self.logger.info("Strategy configuration updated")

    def update_interval(self, interval):
        """작업 실행 간격 업데이트 (주석 유지)"""
        try:
            # 현재 설정 가져오기
            with open(self.strategy_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # interval 값 업데이트
            config['interval'] = interval
            
            # 주석이 유지되는 방식으로 저장
            if not self._save_config_with_comments(config):
                # 실패 시 기존 방식으로 저장 시도
                with open(self.strategy_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                self.logger.warning("Interval updated but comments were lost")
            
            self.logger.info(f"Updated execution interval: {interval} minutes")
            
            # 만약 실행 중이라면, 스케줄링 업데이트
            if hasattr(self, 'trading_thread') and self.trading_thread and self.is_running:
                self.stop()
                self.start()
        except Exception as e:
            self.logger.error(f"Error updating interval: {str(e)}")
            raise

    def _load_or_train_model(self):
        """ML 모델 로드 또는 학습"""
        model = StockPredictionModel()
        try:
            # 최신 모델 파일 찾기
            import glob
            import os
            model_files = glob.glob(os.path.join("models", "stock_model_*.pkl"))
            
            if model_files:
                # 가장 최근 모델 로드
                latest_model = max(model_files)
                model.load(os.path.basename(latest_model))
                self.logger.info(f"ML 모델 로드됨: {latest_model}")
                self.ml_model = model
            else:
                # 모델 신규 학습
                self.logger.info("기존 모델이 없습니다. 새로운 모델을 학습합니다.")
                self.ml_model = train_model(self.market_data, self.target_stocks, days=300)
        except Exception as e:
            self.logger.error(f"ML 모델 로드/학습 실패: {str(e)}")
    
    def retrain_model(self):
        """모델 재학습"""
        try:
            self.ml_model = train_model(self.market_data, self.target_stocks, days=300)
            return True
        except Exception as e:
            self.logger.error(f"모델 재학습 실패: {str(e)}")
            return False
        
    def _setup_weekly_update(self):
        """주간 업데이트 스케줄 설정"""
        import schedule
        
        # 매주 월요일 오전 8시 업데이트
        schedule.every().monday.at("08:00").do(self._weekly_update_job)
        
        # 업데이트 스레드 시작
        update_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        update_thread.start()

    def _run_scheduler(self):
        """스케줄러 실행"""
        import schedule
        import time
        
        while True:
            schedule.run_pending()
            time.sleep(60)

    def _weekly_update_job(self):
        """주간 업데이트 작업"""
        self.logger.info("주간 종목 업데이트 시작")
        
        # 통합 전략의 주간 업데이트 실행
        if hasattr(self.strategy, 'weekly_update'):
            success = self.strategy.weekly_update()
            
            if success and hasattr(self.strategy, 'selected_stocks'):
                # 선정된 종목으로 타겟 업데이트
                self.target_stocks = self.strategy.selected_stocks
                
                # 파일에 저장
                with open(self.stocks_path, 'w', encoding='utf-8') as f:
                    for stock in self.target_stocks:
                        f.write(f"{stock}\n")
                
                self.logger.info(f"타겟 종목 업데이트 완료: {len(self.target_stocks)}개 종목")
        else:
            self.logger.warning("통합 전략에 주간 업데이트 메서드가 없습니다.")
