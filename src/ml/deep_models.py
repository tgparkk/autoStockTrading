import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Embedding, MultiHeadAttention
from tensorflow.keras.layers import LayerNormalization, GlobalAveragePooling1D, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DeepLearningModel:
    """딥러닝 기반 주가 예측 모델의 기본 클래스"""
    
    def __init__(self, model_dir="models/deep_learning"):
        """
        Args:
            model_dir (str): 모델 저장 디렉토리
        """
        self.model_dir = model_dir
        self.model = None
        self.scalers = {}
        self.feature_names = None
        self.training_date = None
        self.metrics_history = []
        self._ensure_dir()
    
    def _ensure_dir(self):
        """모델 저장 디렉토리 확인 및 생성"""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    def preprocess_data(self, X, y=None, is_training=False):
        """데이터 전처리
        
        Args:
            X (pd.DataFrame): 입력 데이터
            y (pd.Series, optional): 타겟 데이터
            is_training (bool): 학습 모드 여부
            
        Returns:
            tuple: 전처리된 X, y
        """
        if is_training:
            # 학습 모드에서는 새로운 scaler 생성
            X_copy = X.copy()
            
            # 특성별 스케일러 생성 및 적용
            for column in X_copy.columns:
                self.scalers[column] = MinMaxScaler()
                X_copy[column] = self.scalers[column].fit_transform(X_copy[[column]])
            
            # 특성 이름 저장
            self.feature_names = X.columns.tolist()
            
            if y is not None:
                return X_copy, y
            return X_copy
        else:
            # 평가/예측 모드에서는 기존 scaler 사용
            if not self.scalers:
                raise ValueError("먼저 모델을 학습시키거나 저장된 모델을 로드해야 합니다.")
            
            X_copy = X.copy()
            
            # 모든 열 스케일링
            for column in X_copy.columns:
                if column in self.scalers:
                    X_copy[column] = self.scalers[column].transform(X_copy[[column]])
                else:
                    logger.warning(f"'{column}' 특성에 대한 스케일러가 없습니다. 0으로 대체합니다.")
                    X_copy[column] = 0  # 없는 특성은 0으로 처리
            
            if y is not None:
                return X_copy, y
            return X_copy
    
    def create_sequences(self, X, y=None, window_size=10):
        """시계열 데이터를 시퀀스로 변환
        
        Args:
            X (pd.DataFrame): 입력 데이터
            y (pd.Series, optional): 타겟 데이터
            window_size (int): 시퀀스 윈도우 크기
            
        Returns:
            tuple: (X_seq, y_seq) 시퀀스 데이터
        """
        X_values = X.values
        X_seq = []
        y_seq = []
        
        for i in range(len(X_values) - window_size):
            X_seq.append(X_values[i:i+window_size])
            if y is not None:
                y_seq.append(y.iloc[i+window_size])
        
        X_seq = np.array(X_seq)
        
        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        return X_seq
    
    def save(self, filename=None):
        """모델 저장
        
        Args:
            filename (str, optional): 저장할 파일명 (기본값: 날짜_모델타입.h5)
            
        Returns:
            str: 저장된 파일 경로
        """
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        # 현재 날짜로 파일명 생성
        if filename is None:
            model_type = self.__class__.__name__
            today = datetime.now().strftime('%Y%m%d')
            filename = f"{model_type}_{today}.h5"
        
        file_path = os.path.join(self.model_dir, filename)
        
        # 모델 저장
        self.model.save(file_path)
        
        # 스케일러와 메타데이터 저장
        meta_file = os.path.splitext(file_path)[0] + '_meta.joblib'
        metadata = {
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'training_date': self.training_date,
            'metrics_history': self.metrics_history
        }
        joblib.dump(metadata, meta_file)
        
        logger.info(f"모델이 저장되었습니다: {file_path}")
        return file_path
    
    def load(self, filepath):
        """모델 로드
        
        Args:
            filepath (str): 모델 파일 경로
            
        Returns:
            Model: 로드된 모델
        """
        # 모델 파일 경로 확인
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {filepath}")
        
        # 모델 로드
        self.model = load_model(filepath)
        
        # 메타데이터 파일 경로
        meta_file = os.path.splitext(filepath)[0] + '_meta.joblib'
        
        # 메타데이터 로드
        if os.path.exists(meta_file):
            metadata = joblib.load(meta_file)
            self.scalers = metadata.get('scalers', {})
            self.feature_names = metadata.get('feature_names')
            self.training_date = metadata.get('training_date')
            self.metrics_history = metadata.get('metrics_history', [])
            
            logger.info(f"모델과 메타데이터가 로드되었습니다: {filepath}")
        else:
            logger.warning(f"메타데이터 파일이 없습니다: {meta_file}")
        
        return self.model
    
    def get_model_info(self):
        """모델 정보 반환
        
        Returns:
            dict: 모델 정보
        """
        if self.model is None:
            return {
                'model_type': 'None',
                'last_training': 'Never',
                'metrics': {},
                'feature_names': [],
                'performance_history': []
            }
        
        # 모델 타입
        model_type = self.__class__.__name__
        
        # 학습 날짜
        last_training = self.training_date.strftime('%Y-%m-%d %H:%M') if self.training_date else 'Unknown'
        
        # 성능 지표
        metrics = {}
        if self.metrics_history:
            last_metrics = self.metrics_history[-1]
            metrics = last_metrics.get('metrics', {})
        
        # 성능 이력
        performance_history = {
            'dates': [],
            'accuracy': [],
            'loss': []
        }
        
        for record in self.metrics_history:
            if 'date' in record:
                performance_history['dates'].append(record['date'])
                performance_history['accuracy'].append(record.get('metrics', {}).get('accuracy', 0))
                performance_history['loss'].append(record.get('metrics', {}).get('loss', 0))
        
        return {
            'model_type': model_type,
            'last_training': last_training,
            'metrics': metrics,
            'feature_names': self.feature_names,
            'performance_history': performance_history
        }


class LSTMModel(DeepLearningModel):
    """LSTM 기반 주가 예측 모델"""
    
    def build_model(self, input_shape, num_classes=3):
        """LSTM 모델 구축
        
        Args:
            input_shape (tuple): 입력 형태 (시퀀스 길이, 특성 수)
            num_classes (int): 출력 클래스 수 (기본: 3 - 상승/유지/하락)
            
        Returns:
            Model: 구축된 LSTM 모델
        """
        model = Sequential()
        
        # LSTM 레이어 추가
        model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.2))
        
        # 출력 레이어
        if num_classes == 2:
            model.add(Dense(1, activation='sigmoid'))  # 이진 분류 (상승/하락)
        else:
            model.add(Dense(num_classes, activation='softmax'))  # 다중 분류
        
        # 모델 컴파일
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X, y, validation_split=0.2, window_size=10, epochs=50, batch_size=32):
        """LSTM 모델 학습
        
        Args:
            X (pd.DataFrame): 입력 데이터
            y (pd.Series): 타겟 데이터
            validation_split (float): 검증 데이터 비율
            window_size (int): 시퀀스 윈도우 크기
            epochs (int): 학습 에폭 수
            batch_size (int): 배치 크기
            
        Returns:
            History: 학습 이력
        """
        # 데이터 전처리
        X_scaled = self.preprocess_data(X, y, is_training=True)
        
        # 시퀀스 데이터 생성
        X_seq, y_seq = self.create_sequences(X_scaled, y, window_size=window_size)
        
        # 모델이 없으면 생성
        if self.model is None:
            self.build_model(input_shape=(window_size, X.shape[1]), num_classes=len(np.unique(y)))
        
        # 콜백 설정
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(self.model_dir, 'lstm_checkpoint.h5'),
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # 모델 학습
        history = self.model.fit(
            X_seq, y_seq,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # 학습 날짜 기록
        self.training_date = datetime.now()
        
        # 성능 지표 기록
        metrics = {
            'loss': float(history.history['loss'][-1]),
            'accuracy': float(history.history['accuracy'][-1]),
            'val_loss': float(history.history['val_loss'][-1]),
            'val_accuracy': float(history.history['val_accuracy'][-1])
        }
        
        self.metrics_history.append({
            'date': self.training_date.strftime('%Y-%m-%d %H:%M'),
            'metrics': metrics
        })
        
        return history
    
    def predict(self, X, window_size=10):
        """예측 수행
        
        Args:
            X (pd.DataFrame): 입력 데이터
            window_size (int): 시퀀스 윈도우 크기
            
        Returns:
            np.ndarray: 예측 결과
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        # 데이터 전처리
        X_scaled = self.preprocess_data(X, is_training=False)
        
        # 시퀀스 데이터 생성
        X_seq = self.create_sequences(X_scaled, window_size=window_size)
        
        # 예측
        predictions = self.model.predict(X_seq)
        
        # 다중 분류인 경우 클래스 확률에서 클래스 인덱스로 변환
        if predictions.shape[1] > 1:
            return np.argmax(predictions, axis=1)
        else:
            return (predictions > 0.5).astype(int).flatten()  # 이진 분류
    
    def predict_proba(self, X, window_size=10):
        """확률 예측 수행
        
        Args:
            X (pd.DataFrame): 입력 데이터
            window_size (int): 시퀀스 윈도우 크기
            
        Returns:
            np.ndarray: 클래스별 확률
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        # 데이터 전처리
        X_scaled = self.preprocess_data(X, is_training=False)
        
        # 시퀀스 데이터 생성
        X_seq = self.create_sequences(X_scaled, window_size=window_size)
        
        # 확률 예측
        return self.model.predict(X_seq)


class TransformerModel(DeepLearningModel):
    """Transformer 기반 주가 예측 모델"""
    
    def build_model(self, input_shape, num_classes=3):
        """Transformer 모델 구축
        
        Args:
            input_shape (tuple): 입력 형태 (시퀀스 길이, 특성 수)
            num_classes (int): 출력 클래스 수 (기본: 3 - 상승/유지/하락)
            
        Returns:
            Model: 구축된 Transformer 모델
        """
        # 하이퍼파라미터
        head_size = 256
        num_heads = 4
        ff_dim = 4
        dropout = 0.2
        
        # 입력 레이어
        inputs = Input(shape=input_shape)
        
        # Transformer 인코더 블록
        x = inputs
        
        # 첫 번째 블록
        attention_output = MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = LayerNormalization(epsilon=1e-6)(attention_output + x)
        x = Conv1D(filters=head_size, kernel_size=1, activation="relu")(x)
        x = Dropout(dropout)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # 두 번째 블록
        attention_output = MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = LayerNormalization(epsilon=1e-6)(attention_output + x)
        x = Conv1D(filters=head_size, kernel_size=1, activation="relu")(x)
        x = Dropout(dropout)(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # 출력 처리
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.2)(x)
        
        # 출력 레이어
        if num_classes == 2:
            outputs = Dense(1, activation="sigmoid")(x)  # 이진 분류
        else:
            outputs = Dense(num_classes, activation="softmax")(x)  # 다중 분류
        
        # 모델 생성
        model = Model(inputs=inputs, outputs=outputs)
        
        # 모델 컴파일
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X, y, validation_split=0.2, window_size=20, epochs=50, batch_size=32):
        """Transformer 모델 학습
        
        Args:
            X (pd.DataFrame): 입력 데이터
            y (pd.Series): 타겟 데이터
            validation_split (float): 검증 데이터 비율
            window_size (int): 시퀀스 윈도우 크기
            epochs (int): 학습 에폭 수
            batch_size (int): 배치 크기
            
        Returns:
            History: 학습 이력
        """
        # 데이터 전처리
        X_scaled = self.preprocess_data(X, y, is_training=True)
        
        # 시퀀스 데이터 생성
        X_seq, y_seq = self.create_sequences(X_scaled, y, window_size=window_size)
        
        # 모델이 없으면 생성
        if self.model is None:
            self.build_model(input_shape=(window_size, X.shape[1]), num_classes=len(np.unique(y)))
        
        # 콜백 설정
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(self.model_dir, 'transformer_checkpoint.h5'),
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # 모델 학습
        history = self.model.fit(
            X_seq, y_seq,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # 학습 날짜 기록
        self.training_date = datetime.now()
        
        # 성능 지표 기록
        metrics = {
            'loss': float(history.history['loss'][-1]),
            'accuracy': float(history.history['accuracy'][-1]),
            'val_loss': float(history.history['val_loss'][-1]),
            'val_accuracy': float(history.history['val_accuracy'][-1])
        }
        
        self.metrics_history.append({
            'date': self.training_date.strftime('%Y-%m-%d %H:%M'),
            'metrics': metrics
        })
        
        return history
    
    def predict(self, X, window_size=20):
        """예측 수행
        
        Args:
            X (pd.DataFrame): 입력 데이터
            window_size (int): 시퀀스 윈도우 크기
            
        Returns:
            np.ndarray: 예측 결과
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        # 데이터 전처리
        X_scaled = self.preprocess_data(X, is_training=False)
        
        # 시퀀스 데이터 생성
        X_seq = self.create_sequences(X_scaled, window_size=window_size)
        
        # 예측
        predictions = self.model.predict(X_seq)
        
        # 다중 분류인 경우 클래스 확률에서 클래스 인덱스로 변환
        if predictions.shape[1] > 1:
            return np.argmax(predictions, axis=1)
        else:
            return (predictions > 0.5).astype(int).flatten()  # 이진 분류
    
    def predict_proba(self, X, window_size=20):
        """확률 예측 수행
        
        Args:
            X (pd.DataFrame): 입력 데이터
            window_size (int): 시퀀스 윈도우 크기
            
        Returns:
            np.ndarray: 클래스별 확률
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        # 데이터 전처리
        X_scaled = self.preprocess_data(X, is_training=False)
        
        # 시퀀스 데이터 생성
        X_seq = self.create_sequences(X_scaled, window_size=window_size)
        
        # 확률 예측
        return self.model.predict(X_seq)
