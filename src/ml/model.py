import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import numpy as np

class StockPredictionModel:
    def __init__(self, model_path="models"):
        self.model_path = model_path
        self.model = None
        self._ensure_dir()
        
        # 모델 메타데이터 추가
        self.training_date = None
        self.accuracy = None
        self.f1_score = None
        self.feature_names = None
        self.feature_importances = None
        self.training_metrics_history = []
        
    def _ensure_dir(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
    
    def train(self, X, y, feature_names=None):
        """모델 학습"""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        self.model = model
        
        # 학습 날짜 기록
        self.training_date = datetime.now()
        
        # 특성 이름 저장
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            # 특성 이름이 제공되지 않은 경우 임시 이름 생성
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # 특성 중요도 저장
        if hasattr(model, 'feature_importances_'):
            self.feature_importances = model.feature_importances_
        
        return model
    
    def evaluate(self, X_test, y_test):
        """모델 성능 평가"""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
            
        # 예측 수행
        y_pred = self.model.predict(X_test)
        
        # 각종 성능 지표 계산
        self.accuracy = accuracy_score(y_test, y_pred)
        
        # 다중 클래스인 경우 weighted 평균 사용
        self.f1 = f1_score(y_test, y_pred, average='weighted')
        self.precision = precision_score(y_test, y_pred, average='weighted')
        self.recall = recall_score(y_test, y_pred, average='weighted')
        
        # 성능 지표 기록
        metrics = {
            'date': self.training_date.strftime("%Y-%m-%d %H:%M"),
            'accuracy': self.accuracy,
            'f1': self.f1,
            'precision': self.precision,
            'recall': self.recall
        }
        
        # 성능 이력에 추가
        self.training_metrics_history.append(metrics)
        
        return metrics
    
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
            filename = f"stock_model_{datetime.now().strftime('%Y%m%d')}.pkl"
        
        path = os.path.join(self.model_path, filename)
        
        # 모델과 메타데이터를 딕셔너리로 묶어 저장
        model_data = {
            'model': self.model,
            'training_date': self.training_date,
            'accuracy': self.accuracy,
            'f1_score': self.f1,
            'precision': self.precision,
            'recall': self.recall,
            'feature_names': self.feature_names,
            'feature_importances': self.feature_importances,
            'training_metrics_history': self.training_metrics_history
        }
        
        joblib.dump(model_data, path)
        return path
    
    def load(self, filename):
        """모델 로드"""
        path = os.path.join(self.model_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
        
        # 모델 데이터 로드
        model_data = joblib.load(path)
        
        # 모델 객체 로드
        if isinstance(model_data, dict) and 'model' in model_data:
            # 신규 형식 (메타데이터 포함)
            self.model = model_data['model']
            self.training_date = model_data.get('training_date')
            self.accuracy = model_data.get('accuracy')
            self.f1 = model_data.get('f1_score')
            self.precision = model_data.get('precision')
            self.recall = model_data.get('recall')
            self.feature_names = model_data.get('feature_names')
            self.feature_importances = model_data.get('feature_importances')
            self.training_metrics_history = model_data.get('training_metrics_history', [])
        else:
            # 기존 형식 (모델만 저장)
            self.model = model_data
            
        return self.model
    
    def get_model_info(self):
        """모델 정보 반환"""
        if self.model is None:
            return {
                'model_type': 'None',
                'last_training': 'Never',
                'accuracy': 0.0,
                'f1_score': 0.0,
                'feature_importance': None,
                'performance_history': []
            }
        
        model_type = type(self.model).__name__
        last_training = self.training_date.strftime('%Y-%m-%d %H:%M') if self.training_date else 'Unknown'
        
        # 특성 중요도 계산
        feature_importance = None
        if self.feature_names and self.feature_importances is not None:
            # 중요도 기준 정렬
            indices = np.argsort(self.feature_importances)[::-1]
            feature_importance = {
                'labels': [self.feature_names[i] for i in indices],
                'values': [float(self.feature_importances[i]) for i in indices]
            }
        
        # 성능 이력 처리
        performance_history = {
            'dates': [],
            'accuracy': [],
            'f1_score': []
        }
        
        for record in self.training_metrics_history[-30:]:  # 최근 30개 기록만
            performance_history['dates'].append(record.get('date', ''))
            performance_history['accuracy'].append(record.get('accuracy', 0.0))
            performance_history['f1_score'].append(record.get('f1', 0.0))
        
        return {
            'model_type': model_type,
            'last_training': last_training,
            'accuracy': self.accuracy if self.accuracy is not None else 0.0,
            'f1_score': self.f1 if self.f1 is not None else 0.0,
            'feature_importance': feature_importance,
            'performance_history': performance_history
        }