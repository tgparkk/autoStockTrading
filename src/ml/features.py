import pandas as pd
import numpy as np
from src.utils.data_utils import calculate_moving_average, calculate_rsi, calculate_bollinger_bands

def create_features(df, window_sizes=[5, 10, 20]):
    """시계열 데이터로부터 ML 특성 생성"""
    features = df.copy()
    
    # 1. 기술적 지표 추가
    for size in window_sizes:
        features = calculate_moving_average(features, windows=[size])
    
    features = calculate_rsi(features)
    features = calculate_bollinger_bands(features)
    
    # 2. 가격 변화율
    features['price_change'] = features['stck_clpr'].pct_change()
    
    # 3. 추가 특성
    # 거래량 특성
    features['volume_change'] = features['acml_vol'].pct_change()
    features['volume_ma'] = features['acml_vol'].rolling(window=10).mean()
    
    # 변동성 특성
    features['volatility'] = features['stck_clpr'].rolling(window=20).std()
    
    # 4. 추세 레이블 (상승:1, 하락:-1, 횡보:0)
    features['target'] = np.where(features['price_change'] > 0.01, 1, 
                        np.where(features['price_change'] < -0.01, -1, 0))
    
    # 결측치 제거
    features = features.dropna()
    
    return features