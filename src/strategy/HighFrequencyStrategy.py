import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from .basic_strategy import BasicStrategy
from ..utils.data_utils import calculate_moving_average, calculate_rsi, calculate_bollinger_bands

logger = logging.getLogger(__name__)

class HighFrequencyStrategy(BasicStrategy):
    """고빈도 스캐닝 전략 - 1분 간격으로 코스피 종목 분석"""
    
    def __init__(self, market_data, order_api, config=None):
        super().__init__(market_data, order_api, config)
        
        # 고빈도 전략 설정
        self.hf_config = {
            'scan_interval': 1,          # 기본 스캐닝 간격 (분)
            'top_stock_limit': 100,      # 상위 종목 스캐닝 개수
            'mid_stock_limit': 300,      # 중위 종목 스캐닝 개수
            'candidate_limit': 10,       # 최대 매수 후보 수
            'score_threshold': 5.0,      # 매수 후보 최소 점수 (10점 만점)
            'api_call_delay': 0.2,       # API 호출 간 딜레이(초)
            'chunk_size': 50,            # 한 번에 처리할 종목 수
            'entry_time_start': '09:00', # 매수 시작 시간
            'entry_time_end': '15:20',   # 매수 종료 시간
            'exit_time': '15:25',        # 청산 시간
        }
        
        # 매수 후보 종목 리스트 및 점수
        self.candidate_stocks = []  # (종목코드, 점수) 튜플 리스트
        self.last_scan_time = datetime.now() - timedelta(minutes=10)  # 초기값
        self.all_market_stocks = []  # 전체 종목 리스트 캐시
        self.last_stock_list_update = datetime.now() - timedelta(days=1)  # 초기값
        
        # 종목 분석 결과 캐시
        self.stock_analysis_cache = {}  # {종목코드: {'time': 분석시간, 'signal': 신호, 'data': 데이터}}
        
        logger.info("고빈도 스캐닝 전략이 초기화되었습니다.")
    
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
    
    def _get_all_market_stocks(self, market_type='J'):
        """시장 내 모든 종목 조회 및 캐싱
        
        Args:
            market_type (str): 시장 구분 (J: 코스피, K: 코스닥)
            
        Returns:
            list: 종목 코드 리스트
        """
        # 캐시 유효성 체크 (하루에 한 번만 업데이트)
        now = datetime.now()
        if self.all_market_stocks and (now - self.last_stock_list_update).days < 1:
            logger.debug("캐시된 종목 목록 사용")
            return self.all_market_stocks
        
        logger.info(f"{market_type} 시장 종목 목록 조회 중...")
        
        # 기본 종목 리스트 (API 실패 시 폴백)
        default_stocks = [
            # 코스피 대표 종목들
            '005930', '000660', '035420', '035720', '051910',  # 삼성전자, SK하이닉스, NAVER, 카카오, LG화학
            '005380', '012330', '068270', '207940', '006400',  # 현대차, 현대모비스, 셀트리온, 삼성바이오로직스, 삼성SDI
            '018260', '003550', '036570', '028260', '033780',  # 삼성SDS, LG, 엔씨소프트, 삼성물산, KT&G
            '015760', '017670', '096770', '012330', '039490',  # 한국전력, SK텔레콤, SK이노베이션, 현대모비스, 키움증권
            '016360', '009540', '010130', '011200', '011170',  # 삼성증권, 한국조선해양, 고려아연, HMM, 롯데케미칼
            '055550', '086280', '024110', '003490', '032830',  # 신한지주, 현대글로비스, 기업은행, 대한항공, 삼성생명
        ]
        
        try:
            import requests
            
            # API URL과 헤더 설정
            url = f"{self.market_data.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
            
            # 헤더 설정
            headers = self.market_data.auth.get_auth_headers()
            
            # API로 전체 종목 리스트 조회
            all_stocks = []
            
            # 시가총액 기준 정렬 요청
            params = {
                "fid_cond_mrkt_div_code": market_type,
                "fid_input_iscd": "",
                "fid_org_adj_prc": "1",
                "fid_period_div_code": "D"
            }
            
            response = requests.get(url, params=params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if 'output' in data:
                    # 응답에서 종목 코드 추출
                    for item in data['output']:
                        if 'code' in item:
                            all_stocks.append(item['code'])
                    
                    logger.info(f"{market_type} 시장에서 {len(all_stocks)}개 종목 조회됨")
            
            # 종목이 조회되면 해당 목록 반환, 그렇지 않으면 기본 목록 사용
            if all_stocks:
                self.all_market_stocks = all_stocks
                self.last_stock_list_update = now
                logger.info(f"API에서 총 {len(all_stocks)}개 종목 조회됨")
                return all_stocks
            else:
                logger.warning("API에서 종목을 가져오지 못했습니다. 기본 목록 사용.")
                self.all_market_stocks = default_stocks
                self.last_stock_list_update = now
                return default_stocks
                
        except Exception as e:
            logger.error(f"시장 종목 목록 조회 중 오류: {str(e)}")
            # 오류 발생 시 기본 목록 반환
            self.all_market_stocks = default_stocks
            self.last_stock_list_update = now
            logger.info(f"기본 종목 목록 {len(default_stocks)}개 반환")
            return default_stocks
    
    def _chunk_stocks(self, stocks, chunk_size):
        """종목 리스트를 청크 단위로 분할"""
        for i in range(0, len(stocks), chunk_size):
            yield stocks[i:i + chunk_size]
    
    def scan_rising_stocks(self):
        """상승 중인 종목 스캐닝 (1차 빠른 필터링)"""
        logger.info("상승 종목 스캐닝 시작...")
        
        # 시간대별 스캔 강도 조절
        now = datetime.now()
        current_hour = now.hour
        current_minute = now.minute
        
        # 시장 전체 종목 목록 가져오기
        all_stocks = self._get_all_market_stocks('J')  # J: 코스피
        
        # 시간대별 스캔 범위 조절
        if (current_hour == 9 and current_minute < 30) or \
           (current_hour >= 14 and current_minute >= 30):
            # 장 초반 또는 후반에는 더 많은 종목 스캔
            target_stocks = all_stocks[:self.hf_config['mid_stock_limit']]
            logger.info(f"장 초반/후반 집중 스캔: {len(target_stocks)}개 종목")
        else:
            # 일반 시간대에는 상위 종목만 스캔
            target_stocks = all_stocks[:self.hf_config['top_stock_limit']]
            logger.info(f"일반 시간대 스캔: {len(target_stocks)}개 종목")
        
        # 1차 필터링
        candidates = []
        for chunk in self._chunk_stocks(target_stocks, self.hf_config['chunk_size']):
            for stock_code in chunk:
                try:
                    # 30초 이내에 이미 분석한 종목은 캐시된 결과 사용
                    if stock_code in self.stock_analysis_cache:
                        cache_entry = self.stock_analysis_cache[stock_code]
                        cache_age = (datetime.now() - cache_entry['time']).total_seconds()
                        if cache_age < 30:  # 30초 이내 캐시
                            if cache_entry['signal'] == 'buy':
                                candidates.append((stock_code, cache_entry['score']))
                            continue
                    
                    # 현재가 데이터 조회
                    data = self.market_data.get_stock_current_price(stock_code)
                    if not data:
                        continue
                    
                    # 간단한 필터링: 상승 중이고 거래량이 많은 종목
                    price_change = float(data.get('prdy_ctrt', '0').replace(',', '')) / 100  # % -> 소수점
                    volume_ratio = float(data.get('prdy_vrss_vol_rate', '0').replace(',', '')) / 100  # % -> 소수점
                    
                    # 가격, 거래량 기반 1차 스코어링
                    base_score = 0
                    
                    # 가격 변동 점수 (최대 3점)
                    if price_change > 0.03:  # 3% 이상 상승
                        base_score += 3
                    elif price_change > 0.01:  # 1% 이상 상승
                        base_score += 2
                    elif price_change > 0:  # 상승 중
                        base_score += 1
                    
                    # 거래량 점수 (최대 3점)
                    if volume_ratio > 3:  # 평균 대비 300%
                        base_score += 3
                    elif volume_ratio > 2:  # 평균 대비 200%
                        base_score += 2
                    elif volume_ratio > 1:  # 평균 대비 100%
                        base_score += 1
                    
                    # 4점 이상인 종목만 후보로 선정
                    if base_score >= 4:
                        candidates.append((stock_code, base_score))
                        
                        # 캐시 업데이트
                        self.stock_analysis_cache[stock_code] = {
                            'time': datetime.now(),
                            'signal': 'buy',
                            'score': base_score,
                            'data': {
                                'price_change': price_change,
                                'volume_ratio': volume_ratio
                            }
                        }
                
                except Exception as e:
                    logger.warning(f"종목 필터링 중 오류 ({stock_code}): {str(e)}")
            
            # API 호출 제한 방지
            time.sleep(self.hf_config['api_call_delay'])
        
        # 결과 정렬 (점수 기준 내림차순)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 후보만 반환
        top_candidates = candidates[:self.hf_config['candidate_limit']]
        
        logger.info(f"1차 스캐닝 결과: {len(candidates)}개 후보 중 상위 {len(top_candidates)}개 선정")
        return top_candidates
    
    def analyze_with_multiple_indicators(self, stock_code, base_score=0):
        """다양한 지표를 사용한 상세 종목 분석"""
        try:
            # 캐시 확인 (1분 이내에 이미 분석한 종목은 캐시된 결과 사용)
            if stock_code in self.stock_analysis_cache:
                cache_entry = self.stock_analysis_cache[stock_code]
                cache_age = (datetime.now() - cache_entry['time']).total_seconds()
                if cache_age < 60 and 'detailed_score' in cache_entry:  # 1분 이내 상세 분석 캐시
                    logger.debug(f"{stock_code} 상세 분석 캐시 사용 (나이: {cache_age:.1f}초)")
                    return cache_entry['detailed_score']
            
            logger.debug(f"{stock_code} 상세 분석 시작...")
            
            # 일봉 데이터
            df = self.market_data.get_stock_daily_price(stock_code, period=20)
            if df.empty:
                logger.warning(f"{stock_code} 일봉 데이터 없음")
                return 0
                
            # 분봉 데이터 (옵션)
            intraday_df = None
            try:
                intraday_df = self.market_data.get_intraday_data(stock_code, interval='1', count=30)
            except:
                logger.debug(f"{stock_code} 분봉 데이터 조회 실패")
            
            # 점수 초기화 (기본 점수 반영)
            score = base_score
            
            # 1. 이동평균선 분석
            df = calculate_moving_average(df, windows=[5, 10, 20, 60])
            # 5일선이 20일선 상향돌파
            if len(df) >= 2 and 'ma_5' in df.columns and 'ma_20' in df.columns:
                if df['ma_5'].iloc[0] > df['ma_20'].iloc[0] and df['ma_5'].iloc[1] <= df['ma_20'].iloc[1]:
                    score += 2
                    logger.debug(f"{stock_code} 골든크로스 감지")
                # 5일선이 10일선 위에 있음
                if 'ma_10' in df.columns and df['ma_5'].iloc[0] > df['ma_10'].iloc[0]:
                    score += 0.5
            
            # 2. 볼린저 밴드 분석
            df = calculate_bollinger_bands(df)
            if 'bb_lower' in df.columns and 'bb_upper' in df.columns:
                current_price = df['stck_clpr'].iloc[0]
                # 볼린저 밴드 하단 근처(5% 이내)
                if current_price < df['bb_lower'].iloc[0] * 1.05:
                    score += 1.5
                    logger.debug(f"{stock_code} 볼린저 밴드 하단 지지 감지")
                # 볼린저 밴드 상단 돌파
                elif current_price > df['bb_upper'].iloc[0]:
                    score += 1
                    logger.debug(f"{stock_code} 볼린저 밴드 상단 돌파 감지")
            
            # 3. RSI 분석
            df = calculate_rsi(df)
            if 'rsi' in df.columns:
                rsi_value = df['rsi'].iloc[0]
                # RSI가 과매도 탈출 구간
                if 30 <= rsi_value <= 50:
                    score += 1.5
                    logger.debug(f"{stock_code} RSI 과매도 탈출 감지 (RSI: {rsi_value:.1f})")
                # RSI가 상승 구간
                elif 50 <= rsi_value <= 70:
                    score += 1
                    logger.debug(f"{stock_code} RSI 상승 구간 감지 (RSI: {rsi_value:.1f})")
            
            # 4. 거래량 급증 체크
            if len(df) >= 6 and 'acml_vol' in df.columns:
                avg_volume = df['acml_vol'].iloc[1:6].mean()  # 최근 5일 평균 거래량
                current_volume = df['acml_vol'].iloc[0]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                # 거래량 2배 이상 급증
                if volume_ratio > 2:
                    score += 2
                    logger.debug(f"{stock_code} 거래량 급증 감지 (평균 대비 {volume_ratio:.1f}배)")
                # 거래량 1.5배 이상 증가
                elif volume_ratio > 1.5:
                    score += 1
            
            # 5. 일중 강도 체크 (분봉 데이터)
            if intraday_df is not None and not intraday_df.empty:
                # 분봉 기준 상승 비율
                up_candles = len(intraday_df[intraday_df['prdy_vrss'] > 0])
                strength = up_candles / len(intraday_df) if len(intraday_df) > 0 else 0
                score += strength * 2  # 최대 2점
                logger.debug(f"{stock_code} 일중 강도: {strength:.2f} (상승봉 비율)")
            
            # 점수 정규화 (최대 10점)
            final_score = min(10, score)
            
            # 캐시 업데이트
            if stock_code in self.stock_analysis_cache:
                self.stock_analysis_cache[stock_code]['detailed_score'] = final_score
                self.stock_analysis_cache[stock_code]['time'] = datetime.now()
            else:
                self.stock_analysis_cache[stock_code] = {
                    'time': datetime.now(),
                    'signal': 'buy' if final_score >= self.hf_config['score_threshold'] else 'neutral',
                    'score': final_score,
                    'detailed_score': final_score
                }
            
            logger.debug(f"{stock_code} 상세 분석 완료: 점수 {final_score:.1f}/10")
            return final_score
            
        except Exception as e:
            logger.error(f"상세 지표 분석 중 오류 ({stock_code}): {str(e)}")
            return 0
    
    def select_candidates(self):
        """매수 후보 종목 선정"""
        # 1. 1차 빠른 스캐닝으로 후보 종목 선별
        candidates = self.scan_rising_stocks()
        
        # 2. 후보 종목에 대해 상세 분석
        final_candidates = []
        for stock_code, base_score in candidates:
            # 상세 분석
            score = self.analyze_with_multiple_indicators(stock_code, base_score)
            
            # 임계값 이상인 종목만 최종 후보로 선정
            if score >= self.hf_config['score_threshold']:
                final_candidates.append((stock_code, score))
        
        # 3. 점수순 정렬
        final_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 최종 선정 종목 반환 (점수와 함께)
        self.candidate_stocks = final_candidates
        
        # 코드만 리스트로 추출
        selected_stocks = [code for code, score in final_candidates]
        
        if selected_stocks:
            logger.info(f"최종 매수 후보: {len(selected_stocks)}개 종목 선정됨")
            for idx, (code, score) in enumerate(final_candidates):
                logger.info(f"  {idx+1}. {code}: {score:.1f}/10점")
        else:
            logger.info("매수 후보 종목이 없습니다.")
        
        return selected_stocks
    
    def weekly_update(self):
        """주간 업데이트 - 여기서는 매수 후보 갱신"""
        logger.info("고빈도 전략: 매수 후보 갱신 시작")
        
        # 매수 후보 선정
        selected_stocks = self.select_candidates()
        
        # 선정된 종목이 있으면 속성에 저장
        if selected_stocks:
            self.selected_stocks = selected_stocks
            self.stock_scores = {code: score for code, score in self.candidate_stocks}
            self.selection_date = datetime.now().strftime('%Y-%m-%d')
            return True
        else:
            # 선정된 종목이 없으면 기존 목록 유지
            logger.warning("선정된 종목이 없어 기존 목록을 유지합니다.")
            if not hasattr(self, 'selected_stocks') or not self.selected_stocks:
                # 기본 종목 설정
                default_stocks = ['005930', '000660', '035420']  # 삼성전자, SK하이닉스, NAVER
                logger.info(f"기본 종목 {len(default_stocks)}개 설정")
                self.selected_stocks = default_stocks
                self.stock_scores = {code: 5.0 for code in default_stocks}  # 기본 점수
            return False
    
    def run(self, target_stocks=None):
        """전략 실행"""
        # 주요 로직을 try-except로 감싸서 오류 대응
        try:
            now = datetime.now()
            logger.info(f"고빈도 트레이딩 전략 실행 시작: {now.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 매수 후보 갱신 (1분 간격)
            time_since_last_scan = (now - self.last_scan_time).total_seconds() / 60
            if time_since_last_scan >= self.hf_config['scan_interval']:
                logger.info(f"매수 후보 갱신 ({time_since_last_scan:.1f}분 경과)")
                self.select_candidates()
                self.last_scan_time = now
            
            # 최신 매수 후보 목록 또는 지정된 종목 사용
            stocks_to_use = target_stocks if target_stocks else [code for code, _ in self.candidate_stocks]
            
            if not stocks_to_use:
                logger.warning("분석할 종목이 없습니다.")
                return {'buys': [], 'sells': [], 'errors': []}
            
            # 포지션 정보 업데이트
            self.update_positions()
            
            results = {
                'buys': [],
                'sells': [],
                'errors': []
            }
            
            # 장 마감 청산 체크
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
            
            # 매도 검사 (보유 종목)
            for stock_code in list(self.positions.keys()):
                try:
                    # 현재가 데이터 조회
                    current_data = self.market_data.get_stock_current_price(stock_code)
                    if not current_data:
                        continue
                    
                    # 손익률 계산
                    position = self.positions[stock_code]
                    avg_price = position['avg_price']
                    current_price = float(current_data['stck_prpr'])
                    profit_ratio = (current_price - avg_price) / avg_price
                    
                    # 매도 신호 판단
                    should_sell = False
                    reason = ""
                    
                    # 1. 익절 조건
                    if profit_ratio >= self.config['take_profit']:
                        should_sell = True
                        reason = f"익절 조건 충족 ({profit_ratio:.1%})"
                    
                    # 2. 손절 조건
                    elif profit_ratio <= -self.config['stop_loss']:
                        should_sell = True
                        reason = f"손절 조건 충족 ({profit_ratio:.1%})"
                    
                    # 3. 추세 반전 신호
                    elif float(current_data.get('prdy_ctrt', '0').replace(',', '')) < -2:  # 2% 이상 하락
                        should_sell = True
                        reason = f"하락 추세 감지 (전일대비 {float(current_data.get('prdy_ctrt', '0').replace(',', '')):.1f}%)"
                    
                    # 매도 실행
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
                except Exception as e:
                    logger.error(f"매도 처리 중 오류 ({stock_code}): {str(e)}")
                    results['errors'].append({
                        'stock_code': stock_code,
                        'action': 'sell',
                        'error': str(e)
                    })
            
            # 매수 진입 시간 체크
            if not self.is_entry_time():
                logger.info("매수 가능 시간이 아닙니다.")
                return results
            
            # 최대 포지션 수 체크
            if len(self.positions) >= self.config['max_position']:
                logger.info(f"최대 포지션 수 도달: {len(self.positions)}/{self.config['max_position']}")
                return results
            
            # 매수 검사 (매수 후보 종목)
            for stock_code in stocks_to_use:
                # 이미 보유중인 종목 스킵
                if stock_code in self.positions:
                    continue
                
                try:
                    # 후보 점수 가져오기
                    score = 0
                    for code, s in self.candidate_stocks:
                        if code == stock_code:
                            score = s
                            break
                    
                    # 임계점 이상인 경우만 매수
                    if score >= self.hf_config['score_threshold']:
                        logger.info(f"매수 신호: {stock_code} - 점수: {score:.1f}/10")
                        result = self.execute_buy(stock_code)
                        
                        if result:
                            results['buys'].append({
                                'stock_code': stock_code,
                                'reason': f"고빈도 전략 매수 (점수: {score:.1f}/10)",
                                'result': result
                            })
                            
                            # 최대 포지션 수 도달 시 매수 중단
                            if len(self.positions) >= self.config['max_position']:
                                logger.info(f"최대 포지션 수 도달: {len(self.positions)}/{self.config['max_position']}")
                                break
                        else:
                            results['errors'].append({
                                'stock_code': stock_code,
                                'action': 'buy',
                                'reason': f"고빈도 전략 매수 (점수: {score:.1f}/10)",
                                'error': '주문 실패'
                            })
                except Exception as e:
                    logger.error(f"매수 처리 중 오류 ({stock_code}): {str(e)}")
                    results['errors'].append({
                        'stock_code': stock_code,
                        'action': 'buy',
                        'error': str(e)
                    })
            
            # 결과 요약 로그
            logger.info(f"전략 실행 결과: 매수 {len(results['buys'])}건, 매도 {len(results['sells'])}건, 오류 {len(results['errors'])}건")
            return results
            
        except Exception as e:
            logger.error(f"전략 실행 중 예상치 못한 오류: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())  # 자세한 스택 트레이스 기록
            return {'buys': [], 'sells': [], 'errors': [{'error': str(e)}]}