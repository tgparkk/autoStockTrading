import pandas as pd
import numpy as np
import talib
from talib import abstract
import logging

logger = logging.getLogger(__name__)

def ensure_numeric(df):
    """데이터프레임의 모든 열을 숫자형으로 변환"""
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def add_advanced_indicators(df, ohlcv_columns=None):
    """
    고급 기술적 지표 추가
    
    Args:
        df (pd.DataFrame): 주가 데이터프레임
        ohlcv_columns (dict, optional): OHLCV 컬럼명 매핑. 기본값은 None.
            예: {'open': 'stck_oprc', 'high': 'stck_hgpr', 'low': 'stck_lwpr', 'close': 'stck_clpr', 'volume': 'acml_vol'}
    
    Returns:
        pd.DataFrame: 기술적 지표가 추가된 데이터프레임
    """
    # 기본 컬럼명 설정
    if ohlcv_columns is None:
        ohlcv_columns = {
            'open': 'stck_oprc',
            'high': 'stck_hgpr',
            'low': 'stck_lwpr',
            'close': 'stck_clpr',
            'volume': 'acml_vol'
        }
    
    # 복사본 생성
    result = df.copy()
    
    # 숫자형으로 변환
    result = ensure_numeric(result)
    
    # 필요한 컬럼이 있는지 확인
    required_columns = list(ohlcv_columns.values())
    if not all(col in result.columns for col in required_columns):
        missing = [col for col in required_columns if col not in result.columns]
        logger.warning(f"데이터프레임에 필요한 컬럼이 없습니다: {missing}")
        return result
    
    # TA-Lib에서 사용하는 이름 매핑
    talib_inputs = {
        'open': result[ohlcv_columns['open']].values,
        'high': result[ohlcv_columns['high']].values,
        'low': result[ohlcv_columns['low']].values,
        'close': result[ohlcv_columns['close']].values,
        'volume': result[ohlcv_columns['volume']].values
    }
    
    try:
        # 1. 추세 지표
        
        # ADX - Average Directional Index (추세 강도)
        result['adx'] = talib.ADX(
            talib_inputs['high'], 
            talib_inputs['low'], 
            talib_inputs['close'], 
            timeperiod=14
        )
        
        # AROON - Aroon Indicator (추세 방향과 강도)
        aroon_down, aroon_up = talib.AROON(
            talib_inputs['high'], 
            talib_inputs['low'], 
            timeperiod=14
        )
        result['aroon_down'] = aroon_down
        result['aroon_up'] = aroon_up
        result['aroon_osc'] = aroon_up - aroon_down
        
        # 2. 모멘텀 지표
        
        # MACD - Moving Average Convergence/Divergence
        macd, macd_signal, macd_hist = talib.MACD(
            talib_inputs['close'], 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        result['macd'] = macd
        result['macd_signal'] = macd_signal
        result['macd_hist'] = macd_hist
        
        # RSI - Relative Strength Index
        result['rsi_14'] = talib.RSI(talib_inputs['close'], timeperiod=14)
        
        # Stochastic
        slowk, slowd = talib.STOCH(
            talib_inputs['high'], 
            talib_inputs['low'], 
            talib_inputs['close'], 
            fastk_period=5, 
            slowk_period=3, 
            slowk_matype=0, 
            slowd_period=3, 
            slowd_matype=0
        )
        result['stoch_k'] = slowk
        result['stoch_d'] = slowd
        
        # 3. 변동성 지표
        
        # ATR - Average True Range (변동성)
        result['atr'] = talib.ATR(
            talib_inputs['high'], 
            talib_inputs['low'], 
            talib_inputs['close'], 
            timeperiod=14
        )
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            talib_inputs['close'], 
            timeperiod=20, 
            nbdevup=2, 
            nbdevdn=2, 
            matype=0
        )
        result['bb_upper'] = upper
        result['bb_middle'] = middle
        result['bb_lower'] = lower
        
        # BB Width & %B
        result['bb_width'] = (upper - lower) / middle
        result['bb_pct_b'] = (talib_inputs['close'] - lower) / (upper - lower)
        
        # 4. 거래량 지표
        
        # OBV - On Balance Volume
        result['obv'] = talib.OBV(talib_inputs['close'], talib_inputs['volume'])
        
        # CMF - Chaikin Money Flow
        adl = talib.AD(
            talib_inputs['high'], 
            talib_inputs['low'], 
            talib_inputs['close'], 
            talib_inputs['volume']
        )
        result['cmf'] = talib.SMA(adl, timeperiod=20) / talib.SMA(talib_inputs['volume'], timeperiod=20)
        
        # 5. 캔들 패턴
        
        # Hammer (망치형)
        result['hammer'] = talib.CDLHAMMER(
            talib_inputs['open'], 
            talib_inputs['high'], 
            talib_inputs['low'], 
            talib_inputs['close']
        )
        
        # Evening/Morning Star
        result['evening_star'] = talib.CDLEVENINGSTAR(
            talib_inputs['open'], 
            talib_inputs['high'], 
            talib_inputs['low'], 
            talib_inputs['close']
        )
        
        result['morning_star'] = talib.CDLMORNINGSTAR(
            talib_inputs['open'], 
            talib_inputs['high'], 
            talib_inputs['low'], 
            talib_inputs['close']
        )
        
        # 6. 추가 지표
        
        # Ichimoku Cloud
        tenkan_sen = (talib.MAX(talib_inputs['high'], timeperiod=9) + 
                      talib.MIN(talib_inputs['low'], timeperiod=9)) / 2
        kijun_sen = (talib.MAX(talib_inputs['high'], timeperiod=26) + 
                     talib.MIN(talib_inputs['low'], timeperiod=26)) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        
        # Senkou Span B (Leading Span B)
        senkou_span_b = (talib.MAX(talib_inputs['high'], timeperiod=52) + 
                         talib.MIN(talib_inputs['low'], timeperiod=52)) / 2
        
        result['ichimoku_tenkan'] = tenkan_sen
        result['ichimoku_kijun'] = kijun_sen
        result['ichimoku_senkou_a'] = senkou_span_a
        result['ichimoku_senkou_b'] = senkou_span_b
        
        # 7. 추가 이동평균선
        result['ema_5'] = talib.EMA(talib_inputs['close'], timeperiod=5)
        result['ema_10'] = talib.EMA(talib_inputs['close'], timeperiod=10)
        result['ema_20'] = talib.EMA(talib_inputs['close'], timeperiod=20)
        result['ema_50'] = talib.EMA(talib_inputs['close'], timeperiod=50)
        
        # 8. 피봇 포인트 (Pivot Points)
        # 일 단위 데이터일 경우에만 계산
        if len(result) > 1:
            result['pivot'] = (talib_inputs['high'] + talib_inputs['low'] + talib_inputs['close']) / 3
            result['pivot_r1'] = 2 * result['pivot'] - talib_inputs['low']
            result['pivot_s1'] = 2 * result['pivot'] - talib_inputs['high']
            
        # 9. 가격 변동성 및 모멘텀 파생 지표
        result['price_change'] = talib_inputs['close'].pct_change()
        result['volume_change'] = talib_inputs['volume'].pct_change()
        
        # 10일 평균 대비 거래량 비율
        vol_sma = talib.SMA(talib_inputs['volume'], timeperiod=10)
        result['volume_sma_ratio'] = talib_inputs['volume'] / vol_sma
        
        # 10. 가격 패턴 분석
        result['higher_high'] = np.nan
        result['lower_low'] = np.nan
        
        # 5일 고점/저점 분석
        for i in range(5, len(result)):
            result.loc[result.index[i], 'higher_high'] = 1 if talib_inputs['high'][i] > max(talib_inputs['high'][i-5:i]) else 0
            result.loc[result.index[i], 'lower_low'] = 1 if talib_inputs['low'][i] < min(talib_inputs['low'][i-5:i]) else 0
            
    except Exception as e:
        logger.error(f"기술적 지표 계산 중 오류: {str(e)}")
    
    return result


def add_fundamental_features(df, fundamental_data):
    """
    펀더멘털 지표 추가
    
    Args:
        df (pd.DataFrame): 주가 데이터프레임
        fundamental_data (dict): 펀더멘털 데이터 딕셔너리
            예상 키: 'per', 'pbr', 'eps', 'bps', 'roe', 'roa' 등
    
    Returns:
        pd.DataFrame: 펀더멘털 지표가 추가된 데이터프레임
    """
    result = df.copy()
    
    # 각 지표 추가
    for indicator, value in fundamental_data.items():
        try:
            result[f'fund_{indicator}'] = value
        except Exception as e:
            logger.error(f"펀더멘털 지표 '{indicator}' 추가 중 오류: {str(e)}")
    
    return result


def create_advanced_features(df, window_sizes=[5, 10, 20, 60], ohlcv_columns=None, fundamental_data=None):
    """
    고급 특성 생성 (기술적 지표 + 파생 특성)
    
    Args:
        df (pd.DataFrame): 주가 데이터프레임
        window_sizes (list): 이동평균 윈도우 크기 리스트
        ohlcv_columns (dict, optional): OHLCV 컬럼명 매핑
        fundamental_data (dict, optional): 펀더멘털 데이터
    
    Returns:
        pd.DataFrame: 고급 특성이 추가된 데이터프레임
    """
    # 기본 컬럼명 설정
    if ohlcv_columns is None:
        ohlcv_columns = {
            'open': 'stck_oprc',
            'high': 'stck_hgpr',
            'low': 'stck_lwpr',
            'close': 'stck_clpr',
            'volume': 'acml_vol'
        }
    
    # 1. 고급 기술적 지표 추가
    features = add_advanced_indicators(df, ohlcv_columns)
    
    # 2. 이동평균선 추가 (기존 기능 활용)
    from src.utils.data_utils import calculate_moving_average
    features = calculate_moving_average(features, column=ohlcv_columns['close'], windows=window_sizes)
    
    # 3. 골든 크로스/데드 크로스 특성 추가
    for i, short_window in enumerate(window_sizes[:-1]):
        for long_window in window_sizes[i+1:]:
            # 두 이동평균선 컬럼명
            short_ma = f'ma_{short_window}'
            long_ma = f'ma_{long_window}'
            
            # 골든 크로스/데드 크로스 계산
            if short_ma in features.columns and long_ma in features.columns:
                # 교차 방향 계산 (1: 상향 교차(골든), -1: 하향 교차(데드), 0: 교차 없음)
                features[f'cross_{short_window}_{long_window}'] = 0
                
                for j in range(1, len(features)):
                    if features[short_ma].iloc[j-1] <= features[long_ma].iloc[j-1] and \
                       features[short_ma].iloc[j] > features[long_ma].iloc[j]:
                        # 골든 크로스
                        features.loc[features.index[j], f'cross_{short_window}_{long_window}'] = 1
                    elif features[short_ma].iloc[j-1] >= features[long_ma].iloc[j-1] and \
                         features[short_ma].iloc[j] < features[long_ma].iloc[j]:
                        # 데드 크로스
                        features.loc[features.index[j], f'cross_{short_window}_{long_window}'] = -1
    
    # 4. 추가 파생 특성
    try:
        # 가격 변동성 계산
        close_col = ohlcv_columns['close']
        if close_col in features.columns:
            # 변동성 (표준편차)
            for window in window_sizes:
                features[f'volatility_{window}'] = features[close_col].rolling(window=window).std()
            
            # 추세 강도
            for window in window_sizes:
                # 선형 회귀 기울기로 추세 강도 계산
                x = np.arange(window)
                for i in range(window, len(features)):
                    y = features[close_col].iloc[i-window:i].values
                    slope, _ = np.polyfit(x, y, 1)
                    features.loc[features.index[i], f'trend_strength_{window}'] = slope
            
            # 가격 모멘텀
            for window in window_sizes:
                features[f'momentum_{window}'] = features[close_col].pct_change(periods=window)
    except Exception as e:
        logger.error(f"파생 특성 계산 중 오류: {str(e)}")
    
    # 5. 펀더멘털 데이터 추가 (제공된 경우)
    if fundamental_data:
        features = add_fundamental_features(features, fundamental_data)
    
    # 6. 결측치 처리
    features = features.fillna(method='ffill').fillna(0)
    
    return features
