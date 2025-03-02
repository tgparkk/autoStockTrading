import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def calculate_moving_average(df, column='stck_clpr', windows=[5, 20, 60, 120]):
    """주가 데이터에 이동평균선 추가
    
    Args:
        df (pd.DataFrame): 주가 데이터프레임
        column (str): 이동평균을 계산할 컬럼명
        windows (list): 이동평균 기간 리스트
        
    Returns:
        pd.DataFrame: 이동평균이 추가된 데이터프레임
    """
    result = df.copy()
    
    for window in windows:
        result[f'ma_{window}'] = result[column].rolling(window=window).mean()
    
    return result

def calculate_rsi(df, column='stck_clpr', period=14):
    """RSI(Relative Strength Index) 계산
    
    Args:
        df (pd.DataFrame): 주가 데이터프레임
        column (str): RSI를 계산할 컬럼명
        period (int): RSI 계산 기간
        
    Returns:
        pd.DataFrame: RSI가 추가된 데이터프레임
    """
    result = df.copy()
    
    # 가격 변화 계산
    delta = result[column].diff()
    
    # 상승/하락 분리
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # 평균 상승/하락 계산
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # RS 계산
    rs = avg_gain / avg_loss
    
    # RSI 계산
    result['rsi'] = 100 - (100 / (1 + rs))
    
    return result

def calculate_bollinger_bands(df, column='stck_clpr', window=20, num_std=2):
    """볼린저 밴드 계산
    
    Args:
        df (pd.DataFrame): 주가 데이터프레임
        column (str): 볼린저 밴드를 계산할 컬럼명
        window (int): 이동평균 기간
        num_std (int): 표준편차 승수
        
    Returns:
        pd.DataFrame: 볼린저 밴드가 추가된 데이터프레임
    """
    result = df.copy()
    
    # 이동평균 계산
    result['bb_ma'] = result[column].rolling(window=window).mean()
    
    # 표준편차 계산
    result['bb_std'] = result[column].rolling(window=window).std()
    
    # 상단 및 하단 밴드 계산
    result['bb_upper'] = result['bb_ma'] + (result['bb_std'] * num_std)
    result['bb_lower'] = result['bb_ma'] - (result['bb_std'] * num_std)
    
    return result

def filter_trading_hours(df, date_column='stck_bsop_date'):
    """장 시간 데이터만 필터링
    
    Args:
        df (pd.DataFrame): 데이터프레임
        date_column (str): 날짜 컬럼명
        
    Returns:
        pd.DataFrame: 장 시간 데이터만 포함된 데이터프레임
    """
    # 장 시간 필터 (9:00 - 15:30)
    df['hour'] = pd.to_datetime(df[date_column]).dt.hour
    df['minute'] = pd.to_datetime(df[date_column]).dt.minute
    
    # 9:00 ~ 15:30 사이 데이터만 필터링
    mask = ((df['hour'] > 9) | ((df['hour'] == 9) & (df['minute'] >= 0))) & \
           ((df['hour'] < 15) | ((df['hour'] == 15) & (df['minute'] <= 30)))
    
    filtered_df = df[mask].copy()
    
    # 임시 컬럼 제거
    filtered_df.drop(['hour', 'minute'], axis=1, inplace=True)
    
    return filtered_df

def calculate_volatility(df, column='stck_clpr', window=20):
    """변동성 계산
    
    Args:
        df (pd.DataFrame): 주가 데이터프레임
        column (str): 변동성을 계산할 컬럼명
        window (int): 변동성 계산 기간
        
    Returns:
        pd.DataFrame: 변동성이 추가된 데이터프레임
    """
    result = df.copy()
    
    # 일별 수익률 계산
    result['daily_return'] = result[column].pct_change()
    
    # 변동성 계산 (표준편차 * sqrt(거래일))
    result['volatility'] = result['daily_return'].rolling(window=window).std() * np.sqrt(window)
    
    # 임시 컬럼 제거
    result.drop(['daily_return'], axis=1, inplace=True)
    
    return result