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
    combined_features = pd.concat(all_features, ignore_index=True)
    
    # 결측치 처리
    combined_features = combined_features.fillna(method='ffill').fillna(0)
    
    # 특성과 타깃 분리
    columns_to_drop = ['target', 'stock_code', 'stck_bsop_date']
    X_columns = [col for col in combined_features.columns if col not in columns_to_drop]
    X = combined_features[X_columns]
    y = combined_features['target']
    
    # 훈련/테스트 세트 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # 현재 경로에 models 디렉토리 생성
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # 모델 유형에 따른 학습
    if model_type == 'random_forest':
        # 랜덤 포레스트 모델 학습
        model = StockPredictionModel(model_path=models_dir)
        model.train(X_train, y_train, feature_names=X_columns)
        
        # 모델 평가
        metrics = model.evaluate(X_test, y_test)
        logger.info(f"랜덤 포레스트 모델 성능: 정확도={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
        
        # 모델 저장
        saved_path = model.save()
        logger.info(f"모델이 저장되었습니다: {saved_path}")
        
        # 모델 설명 생성
        explainer = ModelExplainer(model_path=os.path.join(models_dir, 'explainer'))
        explainer.setup_shap_explainer(model.model, X_train, feature_names=X_columns)
        explainer_path = explainer.save_explainer()
        logger.info(f"모델 설명자가 저장되었습니다: {explainer_path}")
        
        return model
        
    elif model_type == 'lstm':
        # LSTM 모델 학습
        deep_model_dir = os.path.join(models_dir, 'deep_learning')
        os.makedirs(deep_model_dir, exist_ok=True)
        
        deep_model = LSTMModel(model_path=deep_model_dir)
        window_size = 10  # LSTM 시퀀스 길이
        
        # 모델 구축
        deep_model.build_model(input_shape=(window_size, X.shape[1]), num_classes=len(np.unique(y)))
        
        # 모델 학습
        history = deep_model.train(X, y, window_size=window_size, epochs=50, batch_size=32)
        
        # 모델 저장
        saved_path = deep_model.save()
        logger.info(f"LSTM 모델이 저장되었습니다: {saved_path}")
        
        # 성능 기록
        metrics = {
            'accuracy': float(history.history['accuracy'][-1]),
            'val_accuracy': float(history.history['val_accuracy'][-1])
        }
        logger.info(f"LSTM 모델 성능: 정확도={metrics['accuracy']:.4f}, 검증 정확도={metrics['val_accuracy']:.4f}")
        
        return deep_model
        
    elif model_type == 'transformer':
        # Transformer 모델 학습
        deep_model_dir = os.path.join(models_dir, 'deep_learning')
        os.makedirs(deep_model_dir, exist_ok=True)
        
        deep_model = TransformerModel(model_path=deep_model_dir)
        window_size = 20  # Transformer 시퀀스 길이
        
        # 모델 구축
        deep_model.build_model(input_shape=(window_size, X.shape[1]), num_classes=len(np.unique(y)))
        
        # 모델 학습
        history = deep_model.train(X, y, window_size=window_size, epochs=50, batch_size=32)
        
        # 모델 저장
        saved_path = deep_model.save()
        logger.info(f"Transformer 모델이 저장되었습니다: {saved_path}")
        
        # 성능 기록
        metrics = {
            'accuracy': float(history.history['accuracy'][-1]),
            'val_accuracy': float(history.history['val_accuracy'][-1])
        }
        logger.info(f"Transformer 모델 성능: 정확도={metrics['accuracy']:.4f}, 검증 정확도={metrics['val_accuracy']:.4f}")
        
        return deep_model
    
    else:
        logger.error(f"지원되지 않는 모델 유형: {model_type}")
        return None