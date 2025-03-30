import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
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
    columns_to_drop = ['target', 'stock_code', 'stck_bsop_date']
    X_columns = [col for col in combined_features.columns if col not in columns_to_drop]
    X = combined_features[X_columns]
    y = combined_features['target']
    
    # 훈련/테스트 세트 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # 모델 학습
    model = StockPredictionModel()
    model.train(X_train, y_train, feature_names=X_columns)
    
    # 모델 평가
    metrics = model.evaluate(X_test, y_test)
    logger.info(f"모델 성능: 정확도={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    # 모델 저장
    saved_path = model.save()
    logger.info(f"모델이 저장되었습니다: {saved_path}")
    
    return model