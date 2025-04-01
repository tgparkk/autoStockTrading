import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from src.strategy.basic_strategy import BasicStrategy  # 상대경로에서 절대경로로 변경
from src.utils.data_utils import calculate_moving_average, calculate_rsi, calculate_bollinger_bands  # 상대경로에서 절대경로로 변경

logger = logging.getLogger(__name__)

class MLHighFrequencyStrategy(BasicStrategy):
    """머신러닝 기반 고빈도 매매 전략 클래스
    ML 모델의 예측과 기술적 지표를 결합하여 단기 매매 기회를 포착합니다.
    """
    
    def __init__(self, market_data, order_api, ml_model=None, config=None):
        super().__init__(market_data, order_api, config)
        
        # 머신러닝 모델
        self.ml_model = ml_model
        
        # 고빈도 전략 설정
        self.hf_config = {
            'scan_interval': 1,          # 스캐닝 간격 (분)
            'top_stock_limit': 150,      # 스캐닝 종목 수
            'score_threshold': 7.5,      # 매수 임계점수 (10점 만점)
            'volume_threshold': 1.3,     # 거래량 증가 임계값
            'max_price': 100000,         # 최대 매수 가격 (10만원)
            'min_price': 5000,           # 최소 매수 가격 (5천원)
            'stop_loss': 0.01,           # 손절 기준 (1%)
            'take_profit': 0.02,         # 익절 기준 (2%)
            'trailing_stop': 0.005,      # 트레일링 스탑 (0.5%)
            'position_size': 0.05,       # 포지션 크기 (자본금 대비 5%)
            'max_positions': 10,         # 최대 포지션 개수
            'entry_time_start': '09:05', # 매수 시작 시간
            'entry_time_end': '15:10',   # 매수 종료 시간
            'exit_time': '15:15',        # 청산 시간
            'max_holding_time': 60,      # 최대 보유 시간 (분)
            'api_call_delay': 0.2,       # API 호출 간 딜레이(초)
            'ml_weight': 0.4,            # ML 예측 가중치
            'technical_weight': 0.4,     # 기술적 지표 가중치
            'momentum_weight': 0.2,      # 모멘텀 가중치
        }
        
        # 기본 설정 덮어쓰기
        if config:
            for key, value in config.items():
                if key in self.hf_config:
                    self.hf_config[key] = value
        
        self.config['stop_loss'] = self.hf_config['stop_loss']
        self.config['take_profit'] = self.hf_config['take_profit']
        self.config['position_size'] = self.hf_config['position_size']
        self.config['max_position'] = self.hf_config['max_positions']
        
        # 후보 종목 및 선정 종목
        self.candidate_stocks = []  # (종목코드, 점수) 튜플 리스트
        self.selected_stocks = []   # 최종 선정된 종목 리스트
        self.stock_scores = {}      # 종목별 점수 {종목코드: 점수}
        self.selection_date = datetime.now().strftime('%Y-%m-%d')
        
        # 거래 이력
        self.trade_history = []     # 거래 이력 저장
        self.active_signals = {}    # 활성화된 매매 신호 {종목코드: {시간, 신호, 가격}}
        
        # 성능 모니터링
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0,
            'total_loss': 0,
            'max_drawdown': 0,
        }
        
        # 종목 분석 결과 캐시
        self.stock_analysis_cache = {}  # {종목코드: {'time': 분석시간, 'signal': 신호, 'data': 데이터}}
        self.last_scan_time = datetime.now() - timedelta(minutes=10)
        self.all_market_stocks = []  # 시장 내 모든 종목 리스트
        
        logger.info("ML 기반 고빈도 매매 전략이 초기화되었습니다.")
    
    def is_entry_time(self):
        """매수 가능 시간인지 확인"""
        now = datetime.now()
        current_time = now.strftime('%H:%M')
        return self.hf_config['entry_time_start'] <= current_time <= self.hf_config['entry_time_end']
    
    def is_exit_time(self):
        """청산 시간인지 확인"""
        now = datetime.now()
        current_time = now.strftime('%H:%M')
        return current_time >= self.hf_config['exit_time']
    
    def _get_all_market_stocks(self):
        """시장 내 모든 종목 조회 및 캐싱"""
        # 캐시 유효성 체크 (하루에 한 번만 업데이트)
        now = datetime.now()
        if self.all_market_stocks and (now - self.last_scan_time).days < 1:
            return self.all_market_stocks
        
        logger.info("시장 종목 목록 조회 중...")
        
        # 기본 종목 리스트 (API 실패 시 폴백)
        default_stocks = [
            '005930', '000660', '035420', '035720', '051910',  # 삼성전자, SK하이닉스, NAVER, 카카오, LG화학
            '005380', '012330', '068270', '207940', '006400',  # 현대차, 현대모비스, 셀트리온, 삼성바이오로직스, 삼성SDI
            '018260', '003550', '036570', '028260', '033780',  # 삼성SDS, LG, 엔씨소프트, 삼성물산, KT&G
            '015760', '017670', '096770', '039490', '016360',  # 한국전력, SK텔레콤, SK이노베이션, 키움증권, 삼성증권
            '009540', '010130', '011200', '011170', '055550',  # 한국조선해양, 고려아연, HMM, 롯데케미칼, 신한지주
        ]
        
        try:
            import requests
            
            # 코스피 종목 리스트 조회 시도
            all_stocks = []
            
            for market_code in ["J", "Q"]:  # J: 코스피, Q: 코스닥
                try:
                    # API URL과 헤더 설정
                    url = f"{self.market_data.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
                    
                    # 헤더 설정
                    headers = self.market_data.auth.get_auth_headers()
                    
                    # 요청 파라미터
                    params = {
                        "fid_cond_mrkt_div_code": market_code,
                        "fid_input_iscd": "",
                    }
                    
                    response = requests.get(url, params=params, headers=headers)
                    if response.status_code == 200:
                        data = response.json()
                        
                        # 데이터 추출 방식은 실제 API 응답 구조에 맞게 조정 필요
                        if 'output' in data and isinstance(data['output'], list):
                            for item in data['output']:
                                if 'code' in item:
                                    all_stocks.append(item['code'])
                    
                    logger.info(f"{market_code} 시장 종목 조회 완료: {len(all_stocks)}개")
                except Exception as e:
                    logger.warning(f"시장 {market_code} 종목 조회 중 오류: {str(e)}")
            
            # 종목이 조회되면 해당 목록 반환, 그렇지 않으면 기본 목록 사용
            if all_stocks:
                self.all_market_stocks = all_stocks
                self.last_scan_time = now
                return all_stocks
            else:
                logger.warning("API에서 종목을 가져오지 못했습니다. 기본 목록 사용.")
                self.all_market_stocks = default_stocks
                self.last_scan_time = now
                return default_stocks
                
        except Exception as e:
            logger.error(f"시장 종목 목록 조회 중 오류: {str(e)}")
            # 오류 발생 시 기본 목록 반환
            self.all_market_stocks = default_stocks
            self.last_scan_time = now
            return default_stocks
    
    def filter_candidate_stocks(self):
        """후보 종목 필터링 - 1차 스크리닝"""
        logger.info("후보 종목 필터링 시작...")
        
        # 전체 시장 종목 가져오기
        all_stocks = self._get_all_market_stocks()
        
        # 상위 종목만 필터링 (제한된 수)
        target_stocks = all_stocks[:self.hf_config['top_stock_limit']]
        
        # 후보 종목 저장 리스트
        candidates = []
        
        # 각 종목 분석
        for stock_code in target_stocks:
            try:
                # 현재가 데이터 조회
                current_data = self.market_data.get_stock_current_price(stock_code)
                if not current_data:
                    continue
                
                # 기본 필터링 조건
                # 1. 가격 범위 체크
                price = float(current_data.get('stck_prpr', '0').replace(',', ''))
                if price < self.hf_config['min_price'] or price > self.hf_config['max_price']:
                    continue
                
                # 2. 거래량 체크
                volume_ratio = float(current_data.get('prdy_vrss_vol_rate', '0').replace(',', '')) / 100
                if volume_ratio < self.hf_config['volume_threshold']:
                    continue
                
                # 3. 가격 변동 체크
                price_change = float(current_data.get('prdy_ctrt', '0').replace(',', '')) / 100
                if price_change <= 0:  # 상승 중인 종목만 선택
                    continue
                
                # 초기 점수 계산 (0-10 사이 값)
                base_score = 0
                
                # 가격 변동 점수 (최대 3점)
                if price_change > 0.03:  # 3% 이상 상승
                    base_score += 3
                elif price_change > 0.02:  # 2% 이상 상승
                    base_score += 2
                elif price_change > 0.01:  # 1% 이상 상승
                    base_score += 1
                
                # 거래량 점수 (최대 3점)
                if volume_ratio > 3:  # 평균 대비 300%
                    base_score += 3
                elif volume_ratio > 2:  # 평균 대비 200%
                    base_score += 2
                elif volume_ratio > 1.5:  # 평균 대비 150%
                    base_score += 1.5
                elif volume_ratio > 1:  # 평균 대비 100%
                    base_score += 1
                
                # 스코어가 충분히 높은 종목만 후보로 선정
                if base_score >= 3:
                    candidates.append((stock_code, base_score))
                    
                    # 캐시 업데이트
                    self.stock_analysis_cache[stock_code] = {
                        'time': datetime.now(),
                        'signal': 'potential',
                        'score': base_score,
                        'data': {
                            'price': price,
                            'price_change': price_change,
                            'volume_ratio': volume_ratio
                        }
                    }
                
                # API 호출 제한 방지
                time.sleep(self.hf_config['api_call_delay'])
                
            except Exception as e:
                logger.warning(f"종목 {stock_code} 필터링 중 오류: {str(e)}")
        
        # 결과 정렬 (점수 기준 내림차순)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"1차 필터링 완료: {len(candidates)}개 후보 선정")
        return candidates
    
    def analyze_with_ml_and_indicators(self, stock_code, base_score=0):
        """머신러닝과 기술적 지표를 결합한 종목 분석"""
        try:
            # 캐시 확인 (2분 이내 분석 결과 재사용)
            if stock_code in self.stock_analysis_cache:
                cache_entry = self.stock_analysis_cache[stock_code]
                cache_age = (datetime.now() - cache_entry['time']).total_seconds()
                if cache_age < 120 and 'ml_score' in cache_entry:
                    logger.debug(f"{stock_code} ML 분석 캐시 사용 (캐시 유효기간: {cache_age:.1f}초)")
                    return cache_entry.get('final_score', 0)
            
            logger.info(f"{stock_code} ML 및 기술적 지표 분석 시작")
            
            # 일봉 데이터 가져오기
            df = self.market_data.get_stock_daily_price(stock_code, period=30)
            if df.empty:
                logger.warning(f"{stock_code} 일봉 데이터 없음")
                return 0
            
            # 분봉 데이터 가져오기
            intraday_df = None
            try:
                intraday_df = self.market_data.get_intraday_data(stock_code, interval='1', count=30)
            except Exception as e:
                logger.warning(f"{stock_code} 분봉 데이터 조회 실패: {str(e)}")
            
            # 점수 초기화 (기본 점수 반영)
            total_score = base_score
            
            # 1. 기술적 지표 분석 (최대 4점)
            technical_score = self._analyze_technical_indicators(df, intraday_df)
            
            # 2. ML 모델 예측 (최대 4점)
            ml_score = self._get_ml_prediction(df)
            
            # 3. 모멘텀 분석 (최대 2점)
            momentum_score = self._analyze_momentum(df)
            
            # 가중치 적용하여 최종 점수 계산 (최대 10점)
            final_score = (
                technical_score * self.hf_config['technical_weight'] + 
                ml_score * self.hf_config['ml_weight'] + 
                momentum_score * self.hf_config['momentum_weight']
            )
            
            # 점수 정규화 (0-10 범위로)
            final_score = min(10, max(0, final_score * 10))
            
            # 캐시 업데이트
            self.stock_analysis_cache[stock_code] = {
                'time': datetime.now(),
                'technical_score': technical_score,
                'ml_score': ml_score,
                'momentum_score': momentum_score,
                'final_score': final_score,
                'signal': 'buy' if final_score >= self.hf_config['score_threshold'] else 'neutral'
            }

            # 각 단계마다 점수 로깅 추가
            logger.info(f"{stock_code} 기술지표 점수: {technical_score:.2f}")
            logger.info(f"{stock_code} ML 점수: {ml_score:.2f}")
            logger.info(f"{stock_code} 모멘텀 점수: {momentum_score:.2f}")
            
            logger.info(f"{stock_code} 분석 완료: 기술지표={technical_score:.2f}, ML={ml_score:.2f}, 모멘텀={momentum_score:.2f}, 최종={final_score:.2f}/10점")
            
            return final_score
            
        except Exception as e:
            logger.error(f"{stock_code} 종합 분석 중 오류: {str(e)}")
            return 0
    
    def _analyze_technical_indicators(self, df, intraday_df=None):
        """기술적 지표 분석 - 0~1 사이 점수 반환"""
        if df.empty:
            return 0
        
        indicators_score = 0
        reasons = []
        
        # 1. 이동평균선 분석
        try:
            df = calculate_moving_average(df, windows=[5, 10, 20, 60])
            
            # 5일선 > 10일선 > 20일선 (강한 상승추세)
            if len(df) >= 2 and 'ma_5' in df.columns and 'ma_10' in df.columns and 'ma_20' in df.columns:
                if df['ma_5'].iloc[0] > df['ma_10'].iloc[0] > df['ma_20'].iloc[0]:
                    indicators_score += 0.2
                    reasons.append("강한 상승추세")
                # 5일선이 20일선 상향돌파 (골든크로스)
                elif df['ma_5'].iloc[0] > df['ma_20'].iloc[0] and df['ma_5'].iloc[1] <= df['ma_20'].iloc[1]:
                    indicators_score += 0.25
                    reasons.append("골든크로스 감지")
                # 5일선 > 20일선 (약한 상승추세)
                elif df['ma_5'].iloc[0] > df['ma_20'].iloc[0]:
                    indicators_score += 0.15
                    reasons.append("약한 상승추세")
        except Exception as e:
            logger.debug(f"이동평균선 분석 오류: {str(e)}")
        
        # 2. RSI 분석
        try:
            df = calculate_rsi(df)
            if 'rsi' in df.columns:
                rsi_value = df['rsi'].iloc[0]
                
                # RSI가 30-70 구간에서 상승 중 (좋은 매수 지점)
                if 30 <= rsi_value <= 45:
                    indicators_score += 0.25
                    reasons.append(f"RSI 과매도 탈출 ({rsi_value:.1f})")
                elif 45 < rsi_value <= 60:
                    indicators_score += 0.15
                    reasons.append(f"RSI 상승 구간 ({rsi_value:.1f})")
        except Exception as e:
            logger.debug(f"RSI 분석 오류: {str(e)}")
        
        # 3. 볼린저 밴드 분석
        try:
            df = calculate_bollinger_bands(df)
            if 'bb_lower' in df.columns and 'bb_upper' in df.columns:
                current_price = df['stck_clpr'].iloc[0]
                
                # 볼린저 밴드 하단 반등 신호 (강한 매수 신호)
                if current_price < df['bb_lower'].iloc[0] * 1.02:
                    indicators_score += 0.25
                    reasons.append("볼린저 하단 지지")
                
                # 볼린저 중간 밴드 돌파 (상승 추세 신호)
                elif df['stck_clpr'].iloc[1] < df['bb_ma'].iloc[1] and current_price > df['bb_ma'].iloc[0]:
                    indicators_score += 0.15
                    reasons.append("볼린저 중앙선 상향돌파")
        except Exception as e:
            logger.debug(f"볼린저 밴드 분석 오류: {str(e)}")
        
        # 4. 거래량 급증 분석
        try:
            if len(df) >= 6 and 'acml_vol' in df.columns:
                avg_volume = df['acml_vol'].iloc[1:6].mean()  # 최근 5일 평균 거래량
                current_volume = df['acml_vol'].iloc[0]
                
                # 거래량 급증 신호
                if avg_volume > 0:
                    volume_ratio = current_volume / avg_volume
                    if volume_ratio > 2:
                        indicators_score += 0.25
                        reasons.append(f"거래량 급증 (평균대비 {volume_ratio:.1f}배)")
                    elif volume_ratio > 1.5:
                        indicators_score += 0.15
                        reasons.append(f"거래량 증가 (평균대비 {volume_ratio:.1f}배)")
        except Exception as e:
            logger.debug(f"거래량 분석 오류: {str(e)}")
        
        # 5. 분봉 데이터 분석 (스윙 모멘텀)
        if intraday_df is not None and not intraday_df.empty:
            try:
                # 최근 30분 동안의 상승봉 비율
                up_candles = sum(1 for i in range(min(30, len(intraday_df))) 
                               if intraday_df['stck_clpr'].iloc[i] > intraday_df['stck_oprc'].iloc[i])
                up_ratio = up_candles / min(30, len(intraday_df))
                
                if up_ratio > 0.7:  # 70% 이상이 상승봉
                    indicators_score += 0.25
                    reasons.append(f"단기 상승 모멘텀 강함 ({up_ratio:.1%})")
                elif up_ratio > 0.6:  # 60% 이상이 상승봉
                    indicators_score += 0.15
                    reasons.append(f"단기 상승 모멘텀 있음 ({up_ratio:.1%})")
            except Exception as e:
                logger.debug(f"분봉 분석 오류: {str(e)}")
        
        # 최대 점수 제한
        return min(1.0, indicators_score)
    
    def _get_ml_prediction(self, df):
        """머신러닝 모델을 사용한 예측 - 0~1 사이 점수 반환"""
        if not self.ml_model or df.empty:
            return 0.5  # 모델이 없으면 중립
        
        try:
            from src.ml.features import create_features
            
            # 특성 생성
            features_df = create_features(df)
            
            if features_df.empty:
                return 0.5
            
            # 예측용 특성 준비 - target과 stck_bsop_date 컬럼 제거
            features = features_df.drop(['target', 'stck_bsop_date'], axis=1, errors='ignore')
            
            # 특성 중 NaN이 있으면 0으로 대체
            features = features.fillna(0)
            
            # 모델 예측 (클래스별 확률)
            proba = self.ml_model.predict_proba(features.iloc[0:1])
            
            # 예측 결과 해석
            if len(proba[0]) >= 3:  # 3개 클래스
                ml_score = proba[0][2] - proba[0][0]  # 상승 - 하락
                normalized_score = (ml_score + 1) / 2
                return normalized_score
            else:
                return proba[0][1]  # 상승 확률
                
        except Exception as e:
            logger.error(f"ML 예측 중 오류: {str(e)}")
            return 0.5  # 오류 시 중립
    
    def _analyze_momentum(self, df):
        """가격 모멘텀 분석 - 0~1 사이 점수 반환"""
        if df.empty or len(df) < 10:
            return 0.5
        
        try:
            # 최근 가격 변화 계산
            current_price = df['stck_clpr'].iloc[0]
            price_5d_ago = df['stck_clpr'].iloc[min(5, len(df)-1)]
            price_10d_ago = df['stck_clpr'].iloc[min(10, len(df)-1)]
            
            # 5일 변화율
            change_5d = (current_price / price_5d_ago - 1) if price_5d_ago > 0 else 0
            
            # 10일 변화율
            change_10d = (current_price / price_10d_ago - 1) if price_10d_ago > 0 else 0
            
            # 가속 모멘텀 (최근 5일 변화율이 이전 5일보다 큰지)
            is_accelerating = change_5d > (change_10d - change_5d)
            
            # 점수 계산 (1일, 5일 변화율 가중 평균)
            daily_change = df['stck_clpr'].pct_change().iloc[0]
            
            # 일간 변화, 5일 변화 모두 고려
            momentum_score = 0.5  # 기본값 (중립)
            
            if change_5d > 0.05:  # 5% 이상 상승
                momentum_score += 0.3
            elif change_5d > 0.03:  # 3% 이상 상승
                momentum_score += 0.2
            elif change_5d > 0.01:  # 1% 이상 상승
                momentum_score += 0.1
            elif change_5d < -0.05:  # 5% 이상 하락
                momentum_score -= 0.3
            elif change_5d < -0.03:  # 3% 이상 하락
                momentum_score -= 0.2
            elif change_5d < -0.01:  # 1% 이상 하락
                momentum_score -= 0.1
            
            # 일간 모멘텀 추가
            if daily_change > 0.02:  # 2% 이상 상승
                momentum_score += 0.2
            elif daily_change > 0.01:  # 1% 이상 상승
                momentum_score += 0.1
            elif daily_change < -0.02:  # 2% 이상 하락
                momentum_score -= 0.2
            elif daily_change < -0.01:  # 1% 이상 하락
                momentum_score -= 0.1
            
            # 가속 모멘텀 보너스
            if is_accelerating and change_5d > 0:
                momentum_score += 0.1
            
            # 점수 제한 (0-1 범위)
            return max(0, min(1, momentum_score))
            
        except Exception as e:
            logger.error(f"모멘텀 분석 중 오류: {str(e)}")
            return 0.5  # 오류 시 중립
    
    def select_trading_candidates(self):
        """매매 대상 종목 선정"""
        logger.info("매매 후보 종목 선정 시작...")
        
        # 1. 1차 필터링 (기본 기준으로 후보 종목 추출)
        initial_candidates = self.filter_candidate_stocks()
        
        # 2. 상세 분석 및 우선순위 부여
        final_candidates = []
        
        for stock_code, base_score in initial_candidates:
            # ML 및 기술적 지표를 결합한 종합 점수 계산
            final_score = self.analyze_with_ml_and_indicators(stock_code, base_score)
            
            # 임계점 이상만 최종 후보로 선정
            if final_score >= self.hf_config['score_threshold']:
                final_candidates.append((stock_code, final_score))
                self.stock_scores[stock_code] = final_score
        
        # 3. 점수 기준 정렬
        final_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 최종 선정된 종목 리스트 업데이트 (코드만 추출)
        self.selected_stocks = [code for code, _ in final_candidates]
        self.selection_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"최종 선정된 매매 후보: {len(final_candidates)}개 종목")
        for i, (code, score) in enumerate(final_candidates[:10], 1):  # 상위 10개만 로깅
            logger.info(f"  {i}. {code}: {score:.2f}/10점")
        
        return self.selected_stocks
    
    def should_buy(self, stock_code):
        """매수 결정
        
        Args:
            stock_code (str): 종목 코드
            
        Returns:
            tuple: (매수 여부, 사유)
        """
        # 포지션 개수 확인
        if len(self.positions) >= self.hf_config['max_positions']:
            return False, "최대 포지션 개수 초과"
        
        # 매수 가능 시간이 아니면 매수하지 않음
        if not self.is_entry_time():
            return False, "매수 가능 시간이 아님"
        
        # 종목 점수 확인
        score = self.stock_scores.get(stock_code, 0)
        if score < self.hf_config['score_threshold']:
            # 점수가 없거나 임계값 미만이면 상세 분석 실행
            score = self.analyze_with_ml_and_indicators(stock_code)
            self.stock_scores[stock_code] = score
        
        # 임계점 이상이면 매수 신호
        if score >= self.hf_config['score_threshold']:
            reason = f"종합 점수 {score:.2f}/10 (임계값: {self.hf_config['score_threshold']})"
            
            # 현재가 확인
            current_data = self.market_data.get_stock_current_price(stock_code)
            if not current_data:
                return False, "현재가 조회 실패"
            
            # 현재 가격이 매수 가능 범위인지 확인
            current_price = float(current_data['stck_prpr'])
            if current_price < self.hf_config['min_price'] or current_price > self.hf_config['max_price']:
                return False, f"가격 범위 초과 ({current_price}원)"
            
            return True, reason
        
        return False, f"점수 미달 ({score:.2f} < {self.hf_config['score_threshold']})"
    
    def should_sell(self, stock_code):
        """매도 결정
        
        Args:
            stock_code (str): 종목 코드
            
        Returns:
            tuple: (매도 여부, 사유)
        """
        # 보유 여부 확인
        if stock_code not in self.positions:
            return False, "보유 종목 아님"
        
        # 청산 시간 확인 (장 마감 전 청산)
        if self.is_exit_time():
            return True, "장 마감 전 청산 시간"
        
        # 해당 종목 포지션 정보
        position = self.positions[stock_code]
        avg_price = position['avg_price']
        buy_time = position['buy_time']
        
        # 현재가 조회
        current_data = self.market_data.get_stock_current_price(stock_code)
        if not current_data:
            return False, "현재가 조회 실패"
        
        current_price = float(current_data['stck_prpr'])
        
        # 수익률 계산
        profit_ratio = (current_price - avg_price) / avg_price
        
        # 1. 익절 조건 확인
        if profit_ratio >= self.hf_config['take_profit']:
            return True, f"익절 조건 충족 (수익률: {profit_ratio:.2%})"
        
        # 2. 손절 조건 확인
        if profit_ratio <= -self.hf_config['stop_loss']:
            return True, f"손절 조건 충족 (손실률: {profit_ratio:.2%})"
        
        # 3. 최대 보유 시간 초과 확인
        holding_time = (datetime.now() - buy_time).total_seconds() / 60  # 분 단위
        if holding_time > self.hf_config['max_holding_time']:
            # 수익 중이면 청산, 손실 중이면 계속 보유
            if profit_ratio > 0:
                return True, f"최대 보유 시간 초과 (수익 중: {profit_ratio:.2%})"
            else:
                # 큰 손실 중이면 청산 고려
                if profit_ratio < -self.hf_config['stop_loss'] * 0.7:  # 손절선의 70%
                    return True, f"최대 보유 시간 초과 + 큰 손실 (손실률: {profit_ratio:.2%})"
        
        # 4. 실시간 신호 확인 (큰 하락 감지 시)
        price_change = float(current_data.get('prdy_ctrt', '0').replace(',', '')) / 100
        if price_change < -0.03:  # 3% 이상 급락
            return True, f"당일 급락 감지 (하락률: {price_change:.2%})"
        
        # 5. ML 재평가 (주기적으로 수행)
        cache_entry = self.stock_analysis_cache.get(stock_code, {})
        cache_age = (datetime.now() - cache_entry.get('time', datetime.min)).total_seconds() if 'time' in cache_entry else float('inf')
        
        if cache_age > 300:  # 5분 이상 경과
            score = self.analyze_with_ml_and_indicators(stock_code)
            if score < self.hf_config['score_threshold'] * 0.7:  # 임계값의 70% 미만
                return True, f"ML 신호 악화 (현재 점수: {score:.2f})"
        
        return False, "매도 신호 없음"
    
    def execute_buy(self, stock_code, investment_amount=None):
        """매수 실행 (기본 메서드 오버라이드)"""
        logger.info(f"ML 전략 매수 실행: {stock_code}")
        
        try:
            # 현재가 조회
            current_data = self.market_data.get_stock_current_price(stock_code)
            if not current_data:
                logger.error(f"현재가 조회 실패: {stock_code}")
                return None
            
            current_price = float(current_data['stck_prpr'])
            logger.info(f"현재가: {current_price}원")
            
            # 계좌 잔고 조회
            balance = self.market_data.get_account_balance()
            if not balance:
                logger.error("계좌 잔고 조회 실패")
                return None

            # 아래 부분 수정 - account_summary 항목이 있는지만 확인
            if not balance.get('account_summary'):
                logger.error("계좌 요약 정보가 없습니다")
                return None

            # 가용 현금 (조회된 데이터가 문자열이므로 변환)
            try:
                available_cash = float(balance['account_summary'][0].get('dnca_tot_amt', '0'))
                logger.info(f"가용 현금: {available_cash:,.0f}원")
            except (ValueError, IndexError, KeyError) as e:
                logger.error(f"가용 현금 확인 오류: {str(e)}")
                return None
            
            # 투자 금액 계산
            if not investment_amount:
                # 기본 설정: 자본금의 5%로 제한 (최소 10,000원)
                investment_amount = max(
                    10000,  # 최소 1만원
                    min(
                        available_cash * self.hf_config['position_size'],
                        available_cash * 0.2  # 최대 20%로 제한
                    )
                )

            logger.info(f"투자 예정 금액: {investment_amount:,.0f}원")
            
            # 매수 수량 계산
            quantity = int(investment_amount / current_price)
            
            # 최소 수량 확인
            if quantity <= 0:
                logger.warning(f"계산된 매수 수량이 0 이하: {quantity}")
                return None
            
            logger.info(f"매수 수량: {quantity}주, 예상 금액: {quantity * current_price:,.0f}원")
            
            # 주문 실행
            order_result = self.order_api.place_order(
                stock_code=stock_code,
                order_type="02",  # 매수
                quantity=quantity,
                price=current_price,
                order_division="00"  # 지정가
            )
            
            if order_result:
                # 포지션 정보 업데이트
                self.positions[stock_code] = {
                    'quantity': quantity,
                    'avg_price': current_price,
                    'buy_time': datetime.now(),
                    'investment': quantity * current_price,
                    'score': self.stock_scores.get(stock_code, 0)
                }
                
                # 거래 이력 추가
                self.trade_history.append({
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'code': stock_code,
                    'action': 'buy',
                    'price': current_price,
                    'quantity': quantity,
                    'amount': quantity * current_price,
                    'reason': f"점수: {self.stock_scores.get(stock_code, 0):.2f}"
                })
                
                # 성능 지표 업데이트
                self.performance_metrics['total_trades'] += 1
                
                logger.info(f"매수 주문 성공: {stock_code}, {quantity}주 @ {current_price:,.0f}원")
            else:
                logger.error(f"매수 주문 실패: {stock_code}")
            
            return order_result
        
        except Exception as e:
            logger.error(f"매수 실행 중 오류: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def execute_sell(self, stock_code, quantity=None):
        """매도 실행 (기본 메서드 오버라이드)"""
        logger.info(f"ML 전략 매도 실행: {stock_code}")
        
        try:
            # 보유 여부 확인
            if stock_code not in self.positions:
                logger.warning(f"보유하지 않은 종목: {stock_code}")
                return None
            
            # 포지션 정보
            position = self.positions[stock_code]
            buy_price = position['avg_price']
            buy_quantity = position['quantity']
            buy_time = position['buy_time']
            
            # 매도 수량 설정 (기본: 전량 매도)
            if not quantity:
                quantity = buy_quantity
            
            # 현재가 조회
            current_data = self.market_data.get_stock_current_price(stock_code)
            if not current_data:
                logger.error(f"현재가 조회 실패: {stock_code}")
                return None
            
            current_price = float(current_data['stck_prpr'])
            
            # 주문 실행
            order_result = self.order_api.place_order(
                stock_code=stock_code,
                order_type="01",  # 매도
                quantity=quantity,
                price=current_price,
                order_division="00"  # 지정가
            )
            
            if order_result:
                # 수익률 계산
                profit_ratio = (current_price - buy_price) / buy_price
                profit_amount = (current_price - buy_price) * quantity
                
                # 거래 이력 추가
                self.trade_history.append({
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'code': stock_code,
                    'action': 'sell',
                    'price': current_price,
                    'quantity': quantity,
                    'amount': quantity * current_price,
                    'profit_ratio': profit_ratio,
                    'profit_amount': profit_amount,
                    'holding_time': (datetime.now() - buy_time).total_seconds() / 60  # 분 단위
                })
                
                # 성능 지표 업데이트
                if profit_ratio > 0:
                    self.performance_metrics['winning_trades'] += 1
                    self.performance_metrics['total_profit'] += profit_amount
                else:
                    self.performance_metrics['losing_trades'] += 1
                    self.performance_metrics['total_loss'] += abs(profit_amount)
                
                # 포지션 정보 업데이트
                if quantity >= buy_quantity:
                    # 전량 매도
                    del self.positions[stock_code]
                else:
                    # 일부 매도
                    self.positions[stock_code]['quantity'] -= quantity
                
                profit_text = f"수익: {profit_amount:,.0f}원 ({profit_ratio:.2%})"
                logger.info(f"매도 주문 성공: {stock_code}, {quantity}주 @ {current_price:,.0f}원 → {profit_text}")
            else:
                logger.error(f"매도 주문 실패: {stock_code}")
            
            return order_result
            
        except Exception as e:
            logger.error(f"매도 실행 중 오류: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def run(self, target_stocks=None):
        """전략 실행"""
        logger.info("ML 기반 고빈도 매매 전략 실행")
        
        # 전략 실행 결과
        results = {
            'buys': [],
            'sells': [],
            'errors': []
        }
        
        try:
            # 시간 차이 계산
            time_since_last_scan = (datetime.now() - self.last_scan_time).total_seconds() / 60
            
            # 주기적으로 매매 후보 갱신 (10분마다)
            if time_since_last_scan >= 10:
                logger.info(f"주기적 매매 후보 갱신 ({time_since_last_scan:.1f}분 경과)")
                self.select_trading_candidates()
                self.last_scan_time = datetime.now()
            
            # 대상 종목 결정
            stocks_to_trade = []
            if target_stocks:
                stocks_to_trade = target_stocks
            elif self.selected_stocks:
                stocks_to_trade = self.selected_stocks
            else:
                # 후보가 없으면 새로 선정
                stocks_to_trade = self.select_trading_candidates()
            
            if not stocks_to_trade:
                logger.warning("거래할 종목이 없습니다.")
                return results
            
            # 포지션 정보 업데이트 (API로부터 최신 정보 가져오기)
            self.update_positions()
            
            # 장 마감 전 청산 체크
            if self.is_exit_time():
                logger.info("장 마감 전 청산 시간")
                
                for stock_code in list(self.positions.keys()):
                    logger.info(f"장 마감 전 청산: {stock_code}")
                    result = self.execute_sell(stock_code)
                    
                    if result:
                        results['sells'].append({
                            'stock_code': stock_code,
                            'reason': '장 마감 전 청산',
                            'result': result
                        })
                    else:
                        results['errors'].append({
                            'stock_code': stock_code,
                            'action': 'sell',
                            'reason': '장 마감 전 청산',
                            'error': '주문 실패'
                        })
                
                return results
            
            # 1. 매도 검사 (보유 종목)
            for stock_code in list(self.positions.keys()):
                should_sell, reason = self.should_sell(stock_code)
                
                if should_sell:
                    logger.info(f"매도 신호: {stock_code} - {reason}")
                    result = self.execute_sell(stock_code)
                    
                    if result:
                        results['sells'].append({
                            'stock_code': stock_code,
                            'reason': reason,
                            'result': result
                        })
                    else:
                        results['errors'].append({
                            'stock_code': stock_code,
                            'action': 'sell',
                            'reason': reason,
                            'error': '주문 실패'
                        })
            
            # 2. 매수 검사 (후보 종목)
            if self.is_entry_time():  # 매수 가능 시간일 때만
                # 최대 포지션 수 확인
                if len(self.positions) < self.hf_config['max_positions']:
                    # 높은 점수 순으로 정렬
                    sorted_stocks = sorted(
                        [(code, self.stock_scores.get(code, 0)) for code in stocks_to_trade],
                        key=lambda x: x[1], reverse=True
                    )
                    
                    # 상위 종목만 검사 (최대 5개)
                    for stock_code, score in sorted_stocks[:5]:
                        # 이미 보유 중인 종목은 스킵
                        if stock_code in self.positions:
                            continue
                        
                        should_buy, reason = self.should_buy(stock_code)
                        
                        if should_buy:
                            logger.info(f"매수 신호: {stock_code} - {reason}")
                            result = self.execute_buy(stock_code)
                            
                            if result:
                                results['buys'].append({
                                    'stock_code': stock_code,
                                    'reason': reason,
                                    'result': result
                                })
                                
                                # 최대 포지션 수 도달 시 매수 중단
                                if len(self.positions) >= self.hf_config['max_positions']:
                                    logger.info(f"최대 포지션 수 도달: {len(self.positions)}/{self.hf_config['max_positions']}")
                                    break
                            else:
                                results['errors'].append({
                                    'stock_code': stock_code,
                                    'action': 'buy',
                                    'reason': reason,
                                    'error': '주문 실패'
                                })
            else:
                logger.info("매수 가능 시간이 아닙니다.")
            
            # 거래 건수 로깅
            buys_count = len(results['buys'])
            sells_count = len(results['sells'])
            errors_count = len(results['errors'])
            
            logger.info(f"전략 실행 결과: 매수 {buys_count}건, 매도 {sells_count}건, 오류 {errors_count}건")
            
            return results
            
        except Exception as e:
            logger.error(f"전략 실행 중 오류: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return results
    
    def weekly_update(self):
        """주간 업데이트 (기존 선정 종목 갱신)"""
        logger.info("ML 기반 고빈도 전략: 주간 종목 갱신 시작")
        
        try:
            # 종목 선정 실행
            self.select_trading_candidates()
            
            # 선정 결과 로깅
            if self.selected_stocks:
                logger.info(f"주간 종목 갱신 완료: {len(self.selected_stocks)}개 종목 선정")
                return True
            else:
                logger.warning("선정된 종목이 없습니다.")
                return False
        except Exception as e:
            logger.error(f"주간 업데이트 중 오류: {str(e)}")
            return False
    
    def get_performance_metrics(self):
        """성능 지표 반환"""
        metrics = self.performance_metrics.copy()
        
        # 추가 지표 계산
        total_trades = metrics['total_trades']
        if total_trades > 0:
            # 승률
            win_rate = metrics['winning_trades'] / total_trades if total_trades > 0 else 0
            metrics['win_rate'] = win_rate
            
            # 손익비
            avg_profit = metrics['total_profit'] / metrics['winning_trades'] if metrics['winning_trades'] > 0 else 0
            avg_loss = metrics['total_loss'] / metrics['losing_trades'] if metrics['losing_trades'] > 0 else 0
            metrics['profit_loss_ratio'] = avg_profit / avg_loss if avg_loss > 0 else 0
            
            # 기대수익
            expected_return = (win_rate * avg_profit - (1 - win_rate) * avg_loss) if total_trades > 0 else 0
            metrics['expected_return'] = expected_return
        
        # 현재 포지션 정보 추가
        metrics['current_positions'] = len(self.positions)
        
        return metrics