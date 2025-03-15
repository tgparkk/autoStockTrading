import pandas as pd
import logging
from src.ml.features import create_features
from src.ml.model import StockPredictionModel

logger = logging.getLogger(__name__)

def train_model(market_data, stock_codes, days=300):
    """다수 종목 데이터로 모델 학습"""
    all_features = []
    
    for stock_code in stock_codes:
        try:
            # 일별 주가 데이터 조회
            df = market_data.get_stock_daily_price(stock_code, period=days)
            
            if not df.empty:
                # 특성 생성
                features = create_features(df)
                features['stock_code'] = stock_code  # 종목 코드 추가
                all_features.append(features)
                
        except Exception as e:
            logger.error(f"{stock_code} 데이터 처리 중 오류: {str(e)}")
    
    if not all_features:
        logger.error("학습할 데이터가 없습니다.")
        return None
    
    # 모든 종목 데이터 결합
    combined_features = pd.concat(all_features)
    
    # 특성과 타깃 분리
    X = combined_features.drop(['target', 'stock_code', 'stck_bsop_date'], axis=1, errors='ignore')
    y = combined_features['target']
    
    # 학습
    model = StockPredictionModel()
    model.train(X, y)
    saved_path = model.save()
    
    logger.info(f"모델이 학습되었습니다: {saved_path}")
    return model