import logging
import numpy as np
import pandas as pd
from datetime import datetime

# 로깅 설정
logger = logging.getLogger(__name__)

def calculate_trading_signal(stock_data, technical_indicators=None, config=None, stock_code=None, historical_data=None):
    """통합 매매 신호 생성 함수 (개선된 버전)
    
    Args:
        stock_data (dict): 주가 데이터 (API 응답 형식)
        technical_indicators (dict, optional): 기술적 지표 데이터
        config (dict, optional): 신호 계산 설정
        stock_code (str, optional): 종목 코드 - 로깅용
        historical_data (pd.DataFrame, optional): 과거 데이터 - 추가 분석용
        
    Returns:
        dict: {
            'signal': 'buy'/'sell'/'neutral', 
            'score': -100~100 (점수),
            'reasons': [...] (이유 목록),
            'confidence': 0.0~1.0 (신뢰도),
            'details': {...} (세부 데이터)
        }
    """
    # 기본 설정값
    if config is None:
        config = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volume_threshold': 1.5,
            'price_change_threshold': 0.01,
            'score_threshold': 50,  # 50점 이상이면 매수 신호
            'confidence_threshold': 0.6  # 60% 이상 신뢰도일 때 신호 더 강력하게 제공
        }
    
    # 결과 초기화
    result = {
        'signal': 'neutral',
        'score': 0,
        'reasons': [],
        'confidence': 0.5,  # 기본 신뢰도 50%
        'details': {},
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    if stock_code:
        result['stock_code'] = stock_code
    
    try:
        # 1. 기본 데이터 추출
        if not stock_data:
            logger.warning(f"{stock_code if stock_code else ''} 주가 데이터가 없습니다.")
            return result
            
        # 가격 데이터 추출
        try:
            current_price = float(stock_data.get('stck_prpr', '0').replace(',', ''))
            prev_price = float(stock_data.get('stck_sdpr', '0').replace(',', ''))  # 전일 종가
            price_change = 0
            
            if prev_price > 0:
                price_change = (current_price - prev_price) / prev_price
                
            high_price = float(stock_data.get('stck_hgpr', '0').replace(',', ''))
            low_price = float(stock_data.get('stck_lwpr', '0').replace(',', ''))
            
            # 거래량 데이터
            volume = int(stock_data.get('acml_vol', '0').replace(',', ''))
            volume_ratio = float(stock_data.get('prdy_vrss_vol_rate', '0').replace(',', '')) / 100  # % -> 소수점
            
            # 일일 가격 변동 폭 - 변동성 계산
            daily_range = (high_price - low_price) / low_price if low_price > 0 else 0
            
            result['details']['price'] = current_price
            result['details']['price_change'] = price_change
            result['details']['volume_ratio'] = volume_ratio
            result['details']['daily_range'] = daily_range
            
            # 추가 데이터 (있는 경우만)
            if 'prdt_name' in stock_data:
                result['details']['stock_name'] = stock_data['prdt_name']
            
            # 52주 고가/저가 처리
            if 'w52_hgpr' in stock_data and 'w52_lwpr' in stock_data:
                w52_high = float(stock_data.get('w52_hgpr', '0').replace(',', ''))
                w52_low = float(stock_data.get('w52_lwpr', '0').replace(',', ''))
                
                # 52주 고가와 저가 대비 상대적 위치 (0~1)
                if w52_high > w52_low:
                    rel_position = (current_price - w52_low) / (w52_high - w52_low)
                    result['details']['w52_position'] = rel_position
                    
                    # 52주 고가와 저가 근처 확인
                    if current_price >= w52_high * 0.95:
                        result['reasons'].append('52주 고가 근처 거래')
                    elif current_price <= w52_low * 1.05:
                        result['reasons'].append('52주 저가 근처 거래')
        except Exception as e:
            logger.error(f"{stock_code if stock_code else ''} 가격 데이터 추출 오류: {str(e)}")
            # 기본값 설정
            current_price = 0
            price_change = 0
            volume_ratio = 0
            daily_range = 0
        
        # 2. 기술적 지표 확인
        rsi = None
        macd = None
        bb_upper = None
        bb_lower = None
        ma_5 = None
        ma_20 = None
        ma_60 = None
        adx = None
        cci = None
        
        if technical_indicators:
            # 기본 기술적 지표
            rsi = technical_indicators.get('rsi')
            macd = technical_indicators.get('macd')
            bb_upper = technical_indicators.get('bb_upper')
            bb_lower = technical_indicators.get('bb_lower')
            ma_5 = technical_indicators.get('ma_5')
            ma_20 = technical_indicators.get('ma_20')
            ma_60 = technical_indicators.get('ma_60')
            
            # 고급 기술적 지표 - 추가
            adx = technical_indicators.get('adx')  # 추세 강도 지표
            cci = technical_indicators.get('cci')  # 상품 채널 지수
            obv = technical_indicators.get('obv')  # On-Balance Volume
            
            # 지표 세부 정보 저장
            result['details']['rsi'] = rsi
            result['details']['macd'] = macd
            result['details']['bb_upper'] = bb_upper
            result['details']['bb_lower'] = bb_lower
            result['details']['ma_5'] = ma_5
            result['details']['ma_20'] = ma_20
            result['details']['ma_60'] = ma_60
            result['details']['adx'] = adx
            result['details']['cci'] = cci
            result['details']['obv'] = obv
        
        # 3. 점수 계산
        score = 0
        confidence = 0.5  # 기본 신뢰도 50%
        
        # 가격 변동 점수 (최대 30점)
        if price_change > 0.03:  # 3% 이상 상승
            score += 30
            result['reasons'].append('가격 급등 (3% 이상)')
            confidence += 0.1  # 신뢰도 증가
        elif price_change > 0.02:  # 2% 이상 상승
            score += 20
            result['reasons'].append('가격 상승 (2% 이상)')
            confidence += 0.05
        elif price_change > 0.01:  # 1% 이상 상승
            score += 10
            result['reasons'].append('가격 상승 (1% 이상)')
        elif price_change > 0:  # 상승
            score += 5
            result['reasons'].append('가격 상승')
        elif price_change < -0.03:  # 3% 이상 하락
            score -= 30
            result['reasons'].append('가격 급락 (3% 이상)')
            confidence += 0.1  # 하락도 확실해지면 신뢰도 증가
        elif price_change < -0.02:  # 2% 이상 하락
            score -= 20
            result['reasons'].append('가격 하락 (2% 이상)')
            confidence += 0.05
        elif price_change < -0.01:  # 1% 이상 하락
            score -= 10
            result['reasons'].append('가격 하락 (1% 이상)')
        elif price_change < 0:  # 하락
            score -= 5
            result['reasons'].append('가격 하락')
        
        # 거래량 점수 (최대 20점)
        if volume_ratio > 3:  # 평균 대비 300%
            score += 20
            result['reasons'].append('거래량 급증 (평균 대비 3배)')
            confidence += 0.1  # 거래량 급증은 신뢰도 증가
        elif volume_ratio > 2:  # 평균 대비 200%
            score += 15
            result['reasons'].append('거래량 증가 (평균 대비 2배)')
            confidence += 0.05
        elif volume_ratio > 1.5:  # 평균 대비 150%
            score += 10
            result['reasons'].append('거래량 증가 (평균 대비 1.5배)')
            confidence += 0.02
        elif volume_ratio > 1:  # 평균 이상
            score += 5
            result['reasons'].append('거래량 증가')
        
        # RSI 점수 (최대 20점)
        if rsi is not None:
            if rsi < config['rsi_oversold']:  # 과매도
                score += 20
                result['reasons'].append(f'RSI 과매도 ({rsi:.1f})')
                confidence += 0.1  # RSI 과매도는 중요한 신호
            elif rsi < 40:  # 저점 반등 가능성
                score += 10
                result['reasons'].append(f'RSI 저점 ({rsi:.1f})')
            elif rsi > config['rsi_overbought']:  # 과매수
                score -= 20
                result['reasons'].append(f'RSI 과매수 ({rsi:.1f})')
                confidence += 0.1  # RSI 과매수도 중요한 신호
            elif rsi > 60:  # 고점 하락 가능성
                score -= 10
                result['reasons'].append(f'RSI 고점 ({rsi:.1f})')
        
        # 볼린저 밴드 점수 (최대 15점)
        if bb_lower is not None and bb_upper is not None:
            if current_price < bb_lower:  # 하단 돌파 (강한 매수 신호)
                score += 15
                result['reasons'].append('볼린저 밴드 하단 돌파')
                confidence += 0.08
            elif current_price < bb_lower * 1.02:  # 하단 근처
                score += 10
                result['reasons'].append('볼린저 밴드 하단 근처')
                confidence += 0.05
            elif current_price > bb_upper:  # 상단 돌파 (강한 매도 신호)
                score -= 15
                result['reasons'].append('볼린저 밴드 상단 돌파')
                confidence += 0.08
            elif current_price > bb_upper * 0.98:  # 상단 근처
                score -= 10
                result['reasons'].append('볼린저 밴드 상단 근처')
                confidence += 0.05
        
        # 이동평균선 점수 (최대 15점)
        if ma_5 is not None and ma_20 is not None:
            if ma_5 > ma_20 and price_change > 0:  # 골든크로스 상태에서 상승
                score += 15
                result['reasons'].append('골든크로스 상태에서 상승')
                confidence += 0.08
            elif ma_5 > ma_20:  # 골든크로스 상태
                score += 10
                result['reasons'].append('골든크로스 상태')
                confidence += 0.05
            elif ma_5 < ma_20 and price_change < 0:  # 데드크로스 상태에서 하락
                score -= 15
                result['reasons'].append('데드크로스 상태에서 하락')
                confidence += 0.08
            elif ma_5 < ma_20:  # 데드크로스 상태
                score -= 10
                result['reasons'].append('데드크로스 상태')
                confidence += 0.05
        
        # 장기 이동평균 점검 (최대 10점)
        if ma_60 is not None and ma_20 is not None:
            if current_price > ma_60 and ma_20 > ma_60:  # 장기적 상승 추세
                score += 10
                result['reasons'].append('장기 상승 추세')
                confidence += 0.05
            elif current_price < ma_60 and ma_20 < ma_60:  # 장기적 하락 추세
                score -= 10
                result['reasons'].append('장기 하락 추세')
                confidence += 0.05
        
        # MACD 점수 (최대 15점)
        if macd is not None:
            macd_signal = macd.get('signal')
            macd_value = macd.get('value')
            macd_histogram = macd.get('histogram')
            
            if macd_histogram is not None:
                if macd_histogram > 0 and macd_value > 0:  # 강한 상승 신호
                    score += 15
                    result['reasons'].append('MACD 강한 상승 신호')
                    confidence += 0.08
                elif macd_histogram > 0:  # 상승 신호
                    score += 10
                    result['reasons'].append('MACD 상승 신호')
                    confidence += 0.05
                elif macd_histogram < 0 and macd_value < 0:  # 강한 하락 신호
                    score -= 15
                    result['reasons'].append('MACD 강한 하락 신호')
                    confidence += 0.08
                elif macd_histogram < 0:  # 하락 신호
                    score -= 10
                    result['reasons'].append('MACD 하락 신호')
                    confidence += 0.05
        
        # 추가: ADX 지표 추세 강도 점검 (최대 10점)
        if adx is not None:
            if adx > 30:  # 강한 추세
                # 상승 추세인지 하락 추세인지 확인
                if price_change > 0:
                    score += 10
                    result['reasons'].append(f'ADX 강한 상승 추세 ({adx:.1f})')
                    confidence += 0.07
                elif price_change < 0:
                    score -= 10
                    result['reasons'].append(f'ADX 강한 하락 추세 ({adx:.1f})')
                    confidence += 0.07
            elif adx > 20:  # 중간 축 추세
                if price_change > 0:
                    score += 5
                    result['reasons'].append(f'ADX 중간 상승 추세 ({adx:.1f})')
                    confidence += 0.03
                elif price_change < 0:
                    score -= 5
                    result['reasons'].append(f'ADX 중간 하락 추세 ({adx:.1f})')
                    confidence += 0.03
        
        # 추가: CCI 지표 점검 (최대 10점)
        if cci is not None:
            if cci > 100:  # 과매수 지역
                score -= 10
                result['reasons'].append(f'CCI 과매수 지역 ({cci:.1f})')
                confidence += 0.05
            elif cci < -100:  # 과매도 지역
                score += 10
                result['reasons'].append(f'CCI 과매도 지역 ({cci:.1f})')
                confidence += 0.05
        
        # 5. 최종 점수 및 신호 결정
        # 점수 범위 조정 (-100 ~ 100)
        score = max(-100, min(100, score))
        # 신뢰도 범위 조정 (0 ~ 1)
        confidence = max(0.0, min(1.0, confidence))
        
        result['score'] = score
        result['confidence'] = confidence
        
        # 신호 결정
        score_threshold = config['score_threshold']
        
        # 신뢰도가 높으면 점수 기준 느심하게 조정
        if confidence > config['confidence_threshold']:
            # 신뢰도가 높으면 더 적은 점수로도 신호 가능
            score_threshold *= 0.8
            # 신뢰도 상황 추가
            if confidence > 0.8:
                result['reasons'].append('매우 높은 신뢰도의 신호')
            else:
                result['reasons'].append('높은 신뢰도의 신호')
        
        if score >= score_threshold:
            result['signal'] = 'buy'
        elif score <= -score_threshold:
            result['signal'] = 'sell'
        else:
            result['signal'] = 'neutral'
        
        return result
        
    except Exception as e:
        logger.error(f"{stock_code if stock_code else ''} 신호 계산 중 오류: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return result

def calculate_technical_indicators(daily_data):
    """일봉 데이터로부터 기술적 지표 계산
    
    Args:
        daily_data (pd.DataFrame): 일봉 데이터
        
    Returns:
        dict: 계산된 기술적 지표
    """
    if daily_data is None or daily_data.empty:
        return {}
    
    result = {}
    
    try:
        # 기본 데이터 준비
        df = daily_data.copy()
        
        # 종가 컬럼 확인
        price_col = 'stck_clpr'
        if price_col not in df.columns:
            # 컬럼명 체크
            potential_cols = [col for col in df.columns if 'pr' in col.lower() or 'close' in col.lower()]
            if potential_cols:
                price_col = potential_cols[0]
            else:
                logger.error("종가 컬럼을 찾을 수 없습니다.")
                return {}
        
        # 1. 이동평균선 계산
        result['ma_5'] = df[price_col].rolling(window=5).mean().iloc[0] if len(df) >= 5 else None
        result['ma_10'] = df[price_col].rolling(window=10).mean().iloc[0] if len(df) >= 10 else None
        result['ma_20'] = df[price_col].rolling(window=20).mean().iloc[0] if len(df) >= 20 else None
        result['ma_60'] = df[price_col].rolling(window=60).mean().iloc[0] if len(df) >= 60 else None
        
        # 2. RSI 계산
        if len(df) >= 14:
            delta = df[price_col].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            result['rsi'] = 100 - (100 / (1 + rs)).iloc[0]
        
        # 3. 볼린저 밴드 계산
        if len(df) >= 20:
            ma_20 = df[price_col].rolling(window=20).mean()
            std_20 = df[price_col].rolling(window=20).std()
            result['bb_upper'] = (ma_20 + 2 * std_20).iloc[0]
            result['bb_lower'] = (ma_20 - 2 * std_20).iloc[0]
            result['bb_ma'] = ma_20.iloc[0]
        
        # 4. MACD 계산
        if len(df) >= 26:
            ema_12 = df[price_col].ewm(span=12, adjust=False).mean()
            ema_26 = df[price_col].ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line
            
            result['macd'] = {
                'value': macd_line.iloc[0],
                'signal': signal_line.iloc[0],
                'histogram': histogram.iloc[0]
            }
        
        return result
        
    except Exception as e:
        logger.error(f"기술적 지표 계산 중 오류: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {}

def get_trading_signal(market_data, stock_code, config=None):
    """종목 코드로부터 매매 신호 계산 (통합 인터페이스)
    
    Args:
        market_data: MarketData 객체
        stock_code (str): 종목 코드
        config (dict, optional): 설정
        
    Returns:
        dict: 매매 신호 정보
    """
    try:
        # 1. 현재가 데이터 조회
        current_data = market_data.get_stock_current_price(stock_code)
        if not current_data:
            logger.warning(f"{stock_code} 현재가 데이터 조회 실패")
            return {'signal': 'neutral', 'score': 0, 'reasons': ['데이터 없음']}
        
        # 2. 일봉 데이터 조회
        daily_data = market_data.get_stock_daily_price(stock_code, period=60)
        
        # 3. 기술적 지표 계산
        indicators = calculate_technical_indicators(daily_data)
        
        # 4. 매매 신호 계산
        signal = calculate_trading_signal(current_data, indicators, config)
        
        # 추가 메타데이터
        signal['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        signal['stock_code'] = stock_code
        
        return signal
        
    except Exception as e:
        logger.error(f"{stock_code} 매매 신호 계산 중 오류: {str(e)}")
        return {'signal': 'error', 'score': 0, 'reasons': [f'오류: {str(e)}']}