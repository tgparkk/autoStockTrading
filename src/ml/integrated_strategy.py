# src/ml/integrated_strategy.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from src.strategy.basic_strategy import BasicStrategy
from src.ml.features import create_features
from src.ml.model import StockPredictionModel

logger = logging.getLogger(__name__)

class IntegratedStrategy(BasicStrategy):
    """통합 선별-집중 전략"""
    
    def __init__(self, market_data, order_api, config=None):
        super().__init__(market_data, order_api, config)
        
        # 추가 설정값
        self.additional_config = {
            'volume_lookback': 5,         # 거래량 확인 기간 (일)
            'stock_pool_size': 100,       # 초기 종목 풀 크기
            'final_selection_size': 20,   # 최종 선정 종목 수
            'momentum_period': 20,        # 모멘텀 계산 기간 (일)
            'volume_change_threshold': 1.5  # 거래량 증가 임계값 (배수)
        }
        
        # 머신러닝 모델 초기화
        self.stock_model = None
        self.market_regime_model = None
        self._load_models()
        
        # 시장 국면 및 스코어 캐시
        self.market_regime = 'neutral'
        self.stock_scores = {}
        self.last_regime_update = None
        self.selected_stocks = []
    
    def _load_models(self):
        """머신러닝 모델 로드"""
        try:
            # 주식 예측 모델 로드
            self.stock_model = StockPredictionModel()
            self.stock_model.load("stock_model_latest.pkl")
            
            # 시장 국면 예측 모델 로드
            self.market_regime_model = StockPredictionModel()
            self.market_regime_model.load("market_regime_model.pkl")
            
            logger.info("머신러닝 모델 로드 완료")
        except Exception as e:
            logger.error(f"모델 로드 실패: {str(e)}")
    
    def update_stock_pool(self, all_stocks=None):
        """거래량 기반 종목 풀 업데이트"""
        try:
            if not all_stocks:
                # 전체 종목 리스트는 별도 API나 파일에서 가져옴
                # 예시 코드만 제공
                all_stocks = self._get_all_market_stocks()
            
            # 거래량 데이터 수집
            volume_data = {}
            for stock_code in all_stocks:
                try:
                    # 일정 기간 데이터 조회
                    df = self.market_data.get_stock_daily_price(
                        stock_code, 
                        period=self.additional_config['volume_lookback'] + 30  # 추가 데이터 포함
                    )
                    
                    if not df.empty and len(df) >= self.additional_config['volume_lookback']:
                        # 최근 n일 거래량
                        recent_volume = df.head(self.additional_config['volume_lookback'])['acml_vol'].mean()
                        
                        # 이전 20일 거래량
                        prev_volume = df.iloc[self.additional_config['volume_lookback']:25]['acml_vol'].mean()
                        
                        # 거래량 증가율
                        if prev_volume > 0:
                            volume_change = recent_volume / prev_volume
                        else:
                            volume_change = 1.0
                        
                        # 거래량 데이터 저장
                        volume_data[stock_code] = {
                            'recent_volume': recent_volume,
                            'volume_change': volume_change,
                            'price': df.iloc[0]['stck_clpr'],  # 최근 종가
                            'volatility': df.head(20)['stck_clpr'].pct_change().std() * np.sqrt(250)  # 연간화 변동성
                        }
                except Exception as e:
                    logger.warning(f"{stock_code} 거래량 데이터 조회 중 오류: {str(e)}")
            
            # 1차 필터링: 거래량 증가율 기준
            filtered_stocks = {
                code: data for code, data in volume_data.items() 
                if data['volume_change'] >= self.additional_config['volume_change_threshold']
            }
            
            # 2차 필터링: 저가주 및 고변동성 종목 제외
            filtered_stocks = {
                code: data for code, data in filtered_stocks.items()
                if data['price'] >= 5000 and data['volatility'] <= 0.8  # 5,000원 이상, 변동성 80% 이하
            }
            
            # 거래량 기준 정렬 및 상위 N개 선택
            sorted_stocks = sorted(
                filtered_stocks.items(), 
                key=lambda x: x[1]['recent_volume'], 
                reverse=True
            )
            
            # 최종 종목 풀 선정
            pool_size = min(self.additional_config['stock_pool_size'], len(sorted_stocks))
            stock_pool = [code for code, _ in sorted_stocks[:pool_size]]
            
            logger.info(f"거래량 기준 {pool_size}개 종목 풀 선정 완료")
            return stock_pool
        
        except Exception as e:
            logger.error(f"종목 풀 업데이트 중 오류: {str(e)}")
            return []
    
    def select_stocks(self, stock_pool):
        """멀티팩터 기반 최종 종목 선정"""
        try:
            # 각 종목별 점수 계산
            for stock_code in stock_pool:
                if stock_code not in self.stock_scores:
                    score = self._calculate_stock_score(stock_code)
                    self.stock_scores[stock_code] = score
            
            # 점수 기준 정렬
            sorted_stocks = sorted(
                self.stock_scores.items(),
                key=lambda x: x[1],
                reverse=True  # 높은 점수 순
            )
            
            # 최종 종목 선정
            selection_size = min(self.additional_config['final_selection_size'], len(sorted_stocks))
            selected_stocks = [code for code, _ in sorted_stocks[:selection_size]]
            
            self.selected_stocks = selected_stocks
            logger.info(f"멀티팩터 분석으로 {selection_size}개 종목 최종 선정 완료")
            return selected_stocks
        
        except Exception as e:
            logger.error(f"종목 선정 중 오류: {str(e)}")
            return []
    
    def _calculate_stock_score(self, stock_code):
        """종목별 종합 점수 계산"""
        try:
            # 기본 데이터 로드
            df = self.market_data.get_stock_daily_price(stock_code, period=60)
            if df.empty:
                return 0
            
            # 1. 모멘텀 점수 (20일 수익률)
            momentum_period = self.additional_config['momentum_period']
            if len(df) >= momentum_period:
                recent_price = df.iloc[0]['stck_clpr']
                past_price = df.iloc[momentum_period-1]['stck_clpr']
                momentum = (recent_price / past_price - 1) if past_price > 0 else 0
                # 모멘텀 점수 범위 조정 (-1 ~ 1)
                momentum_score = min(max(momentum * 5, -1), 1)
            else:
                momentum_score = 0
            
            # 2. 기술적 지표 계산
            analysis = self.analyze_stock(stock_code)
            
            # 기술적 지표 점수 (-1 ~ 1)
            technical_score = 0
            if 'signal' in analysis:
                if analysis['signal'] == 'buy':
                    technical_score = 0.8
                elif analysis['signal'] == 'sell':
                    technical_score = -0.8
            
            # 3. ML 예측 점수
            ml_score = 0
            if self.stock_model:
                try:
                    # 특성 생성
                    features = create_features(df).iloc[0]
                    # ML 예측 (0: 하락, 1: 횡보, 2: 상승)
                    prediction = self.stock_model.predict_proba([features])
                    # 상승 확률에서 하락 확률을 뺀 값 (-1 ~ 1)
                    ml_score = prediction[0][2] - prediction[0][0]
                except:
                    pass
            
            # 종합 점수 계산 (각 점수의 가중 평균)
            weights = {
                'momentum': 0.35,
                'technical': 0.35,
                'ml': 0.30
            }
            
            total_score = (
                weights['momentum'] * momentum_score +
                weights['technical'] * technical_score +
                weights['ml'] * ml_score
            )
            
            return total_score
            
        except Exception as e:
            logger.error(f"{stock_code} 점수 계산 중 오류: {str(e)}")
            return 0
    
    def update_market_regime(self):
        """시장 국면 업데이트"""
        now = datetime.now()
        
        # 하루에 한 번만 업데이트
        if self.last_regime_update and (now - self.last_regime_update).days < 1:
            return self.market_regime
        
        try:
            # KOSPI 지수 데이터 가져오기
            kospi_data = self.market_data.get_stock_daily_price('KOSPI', period=60)
            
            if kospi_data.empty:
                return 'neutral'
            
            # 특성 생성
            features = create_features(kospi_data).iloc[0]
            
            # 머신러닝 모델로 국면 예측
            if self.market_regime_model:
                # 모델 예측 (0: 약세, 1: 중립, 2: 강세)
                prediction = self.market_regime_model.predict([features])[0]
                
                if prediction == 0:
                    self.market_regime = 'bearish'
                elif prediction == 2:
                    self.market_regime = 'bullish'
                else:
                    self.market_regime = 'neutral'
            else:
                # 모델이 없는 경우 간단한 규칙 기반 판단
                # 20일 이동평균선 대비 현재가 비율
                ma_20 = kospi_data['stck_clpr'].rolling(window=20).mean().iloc[0]
                current_price = kospi_data['stck_clpr'].iloc[0]
                
                if current_price > ma_20 * 1.05:
                    self.market_regime = 'bullish'
                elif current_price < ma_20 * 0.95:
                    self.market_regime = 'bearish'
                else:
                    self.market_regime = 'neutral'
            
            self.last_regime_update = now
            logger.info(f"시장 국면 업데이트: {self.market_regime}")
            
            # 시장 국면에 따라 전략 파라미터 조정
            self._adjust_parameters_by_regime()
            
            return self.market_regime
            
        except Exception as e:
            logger.error(f"시장 국면 업데이트 중 오류: {str(e)}")
            return 'neutral'
    
    def _adjust_parameters_by_regime(self):
        """시장 국면에 따라 파라미터 조정"""
        if self.market_regime == 'bullish':
            # 강세장 파라미터
            self.config['stop_loss'] = 0.05
            self.config['take_profit'] = 0.08
            self.config['position_size'] = 0.25
            self.config['max_position'] = 8
        elif self.market_regime == 'bearish':
            # 약세장 파라미터
            self.config['stop_loss'] = 0.03
            self.config['take_profit'] = 0.05
            self.config['position_size'] = 0.15
            self.config['max_position'] = 5
        else:
            # 중립 파라미터
            self.config['stop_loss'] = 0.04
            self.config['take_profit'] = 0.06
            self.config['position_size'] = 0.20
            self.config['max_position'] = 6
    
    def run(self, target_stocks=None):
        """통합 전략 실행"""
        logger.info("통합 선별-집중 전략 실행")
        
        # 주기적으로 시장 국면 업데이트
        self.update_market_regime()
        
        if not target_stocks or len(target_stocks) == 0:
            target_stocks = self.selected_stocks
        
        if not target_stocks or len(target_stocks) == 0:
            logger.warning("선정된 종목이 없습니다.")
            return {'buys': [], 'sells': [], 'errors': []}
        
        # 기존 BasicStrategy의 run 메서드 호출
        return super().run(target_stocks)
    
    def weekly_update(self):
        """주간 업데이트 수행"""
        try:
            # 1. 거래량 기반 종목 풀 업데이트
            stock_pool = self.update_stock_pool()
            
            # 2. 멀티팩터 기반 최종 종목 선정
            if stock_pool:
                self.select_stocks(stock_pool)
            
            # 3. 시장 국면 갱신
            self.update_market_regime()
            
            return True
        except Exception as e:
            logger.error(f"주간 업데이트 중 오류: {str(e)}")
            return False