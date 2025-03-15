import joblib
import os
from sklearn.ensemble import RandomForestClassifier

class StockPredictionModel:
    def __init__(self, model_path="models"):
        self.model_path = model_path
        self.model = None
        self._ensure_dir()
    
    def _ensure_dir(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
    
    def train(self, X, y):
        """모델 학습"""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        self.model = model
        return model
    
    def predict(self, X):
        """예측 수행"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """확률 예측"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        return self.model.predict_proba(X)
    
    def save(self, filename=None):
        """모델 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        if filename is None:
            # 현재 날짜로 파일명 생성
            from datetime import datetime
            filename = f"stock_model_{datetime.now().strftime('%Y%m%d')}.pkl"
        
        path = os.path.join(self.model_path, filename)
        joblib.dump(self.model, path)
        return path
    
    def load(self, filename):
        """모델 로드"""
        path = os.path.join(self.model_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
        
        self.model = joblib.load(path)
        return self.model