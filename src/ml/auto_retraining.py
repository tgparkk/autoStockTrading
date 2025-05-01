import logging
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import schedule
import time
import threading
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.ml.model import StockPredictionModel
from src.ml.deep_models import LSTMModel, TransformerModel
from src.ml.features import create_features
from src.ml.advanced_features import create_advanced_features
from src.ml.model_explainer import ModelExplainer

logger = logging.getLogger(__name__)

class AutoRetrainingSystem:
    """
    머신러닝 모델 자동 재학습 시스템
    
    주기적으로 모델 성능을 평가하고, 성능이 저하되면 자동으로 재학습을 수행합니다.
    또한 새로운 데이터가 추가될 때마다 점진적인 학습을 수행할 수 있습니다.
    """
    
    def __init__(self, market_data, model_dir="models", 
                 performance_threshold=0.65, check_interval_days=7,
                 model_type="random_forest"):
        """
        Args:
            market_data (MarketData): 시장 데이터 객체
            model_dir (str): 모델 저장 디렉토리
            performance_threshold (float): 성능 임계값 (이 값 미만이면 재학습)
            check_interval_days (int): 성능 체크 간격(일)
            model_type (str): 모델 유형 ('random_forest', 'lstm', 'transformer')
        """
        self.market_data = market_data
        self.model_dir = model_dir
        self.performance_threshold = performance_threshold
        self.check_interval_days = check_interval_days
        self.model_type = model_type
        
        # 모델 객체
        self.model = None
        self.deep_model = None
        
        # 설명 객체
        self.explainer = None
        
        # 성능 모니터링
        self.performance_history = []
        self.last_training_date = None
        self.last_check_date = None
        
        # 스케줄링 스레드
        self.scheduler_thread = None
        self.is_running = False
        
        # 디렉토리 확인
        self._ensure_dir()
        
        # 모델 로드
        self._load_latest_model()
    
    def _ensure_dir(self):
        """모델 저장 디렉토리 확인 및 생성"""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    def _load_latest_model(self):
        """최신 모델 로드"""
        try:
            # 모델 객체 생성
            if self.model_type == 'random_forest':
                self.model = StockPredictionModel(model_path=self.model_dir)
            elif self.model_type == 'lstm':
                self.deep_model = LSTMModel(model_path=os.path.join(self.model_dir, 'deep_learning'))
            elif self.model_type == 'transformer':
                self.deep_model = TransformerModel(model_path=os.path.join(self.model_dir, 'deep_learning'))
            
            # 모델 파일 찾기
            if self.model_type == 'random_forest':
                model_files = [f for f in os.listdir(self.model_dir) if f.startswith('stock_model_') and f.endswith('.pkl')]
                if model_files:
                    # 날짜 기준 정렬
                    model_files.sort(reverse=True)
                    latest_model = model_files[0]
                    # 모델 로드
                    self.model.load(latest_model)
                    self.last_training_date = datetime.strptime(latest_model.split('_')[2].split('.')[0], '%Y%m%d')
                    logger.info(f"최신 모델 로드됨: {latest_model}, 학습일: {self.last_training_date}")
            else:
                deep_model_path = os.path.join(self.model_dir, 'deep_learning')
                if os.path.exists(deep_model_path):
                    model_files = [f for f in os.listdir(deep_model_path) if f.endswith('.h5')]
                    if model_files:
                        model_files.sort(reverse=True)
                        latest_model = model_files[0]
                        # 딥러닝 모델 로드
                        self.deep_model.load(os.path.join(deep_model_path, latest_model))
                        self.last_training_date = datetime.strptime(latest_model.split('_')[1].split('.')[0], '%Y%m%d')
                        logger.info(f"최신 딥러닝 모델 로드됨: {latest_model}, 학습일: {self.last_training_date}")
            
            # 설명자 로드
            explainer_path = os.path.join(self.model_dir, 'explainer')
            if os.path.exists(explainer_path):
                explainer_files = [f for f in os.listdir(explainer_path) if f.endswith('.joblib')]
                if explainer_files:
                    explainer_files.sort(reverse=True)
                    latest_explainer = explainer_files[0]
                    # 설명자 로드
                    self.explainer = ModelExplainer(model_path=explainer_path)
                    self.explainer.load_explainer(os.path.join(explainer_path, latest_explainer))
                    logger.info(f"최신 설명자 로드됨: {latest_explainer}")
            
            # 성능 히스토리 로드
            history_path = os.path.join(self.model_dir, 'performance_history.joblib')
            if os.path.exists(history_path):
                self.performance_history = joblib.load(history_path)
                logger.info(f"성능 히스토리 로드됨: {len(self.performance_history)}개 기록")
        
        except Exception as e:
            logger.error(f"모델 로드 중 오류: {str(e)}")
    
    def fetch_training_data(self, stock_codes, days=300, use_advanced_features=True):
        """
        학습용 데이터 수집
        
        Args:
            stock_codes (list): 종목 코드 리스트
            days (int): 데이터 수집 기간(일)
            use_advanced_features (bool): 고급 특성 사용 여부
            
        Returns:
            tuple: X, y 데이터셋
        """
        all_features = []
        
        for stock_code in stock_codes:
            try:
                # 일별 주가 데이터 조회
                df = self.market_data.get_stock_daily_price(stock_code, period=days)
                
                if not df.empty:
                    # 기본/고급 특성 생성
                    if use_advanced_features:
                        features = create_advanced_features(df)
                    else:
                        features = create_features(df)
                    
                    # 종목 코드 추가
                    features['stock_code'] = stock_code
                    all_features.append(features)
                    
            except Exception as e:
                logger.error(f"{stock_code} 데이터 처리 중 오류: {str(e)}")
        
        if not all_features:
            logger.error("학습할 데이터가 없습니다.")
            return None, None
        
        # 모든 종목 데이터 결합
        combined_features = pd.concat(all_features, ignore_index=True)
        
        # 결측치 처리
        combined_features = combined_features.fillna(method='ffill').fillna(0)
        
        # 특성과 타깃 분리
        if 'target' in combined_features.columns:
            columns_to_drop = ['target', 'stock_code', 'stck_bsop_date']
            y = combined_features['target']
        else:
            # 타깃이 없는 경우, 가격 변화를 타깃으로 사용
            combined_features['price_change'] = combined_features['stck_clpr'].pct_change()
            combined_features['target'] = np.where(combined_features['price_change'] > 0.01, 1, 
                                                np.where(combined_features['price_change'] < -0.01, -1, 0))
            columns_to_drop = ['price_change', 'target', 'stock_code', 'stck_bsop_date']
            y = combined_features['target']
        
        # 특성 추출
        X_columns = [col for col in combined_features.columns if col not in columns_to_drop]
        X = combined_features[X_columns]
        
        logger.info(f"학습 데이터 수집 완료: {len(X)} 샘플, {len(X.columns)} 특성")
        return X, y
    
    def train_model(self, X, y, test_size=0.2, save_model=True):
        """
        모델 학습
        
        Args:
            X (pd.DataFrame): 특성 데이터
            y (pd.Series): 타깃 데이터
            test_size (float): 테스트 데이터 비율
            save_model (bool): 모델 저장 여부
            
        Returns:
            dict: 학습 결과 및 성능 지표
        """
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )
        
        # 모델 유형에 따라 학습
        if self.model_type == 'random_forest':
            if self.model is None:
                self.model = StockPredictionModel(model_path=self.model_dir)
            
            # 모델 학습
            self.model.train(X_train, y_train, feature_names=X.columns.tolist())
            
            # 모델 평가
            metrics = self.model.evaluate(X_test, y_test)
            
            # 모델 저장
            if save_model:
                saved_path = self.model.save()
                logger.info(f"모델 저장 완료: {saved_path}")
                
                # 설명자 생성 및 저장
                self.explainer = ModelExplainer(model_path=os.path.join(self.model_dir, 'explainer'))
                self.explainer.setup_shap_explainer(self.model.model, X_train, feature_names=X.columns.tolist())
                explainer_path = self.explainer.save_explainer()
                logger.info(f"설명자 저장 완료: {explainer_path}")
        
        else:  # 딥러닝 모델
            # 딥러닝 모델 학습을 위한 데이터 준비
            window_size = 10 if self.model_type == 'lstm' else 20
            
            if self.model_type == 'lstm':
                if self.deep_model is None:
                    self.deep_model = LSTMModel(model_path=os.path.join(self.model_dir, 'deep_learning'))
                
                # 모델 구축
                self.deep_model.build_model(input_shape=(window_size, X.shape[1]), num_classes=len(np.unique(y)))
                
                # 모델 학습
                history = self.deep_model.train(X_train, y_train, window_size=window_size, epochs=50, batch_size=32)
                
                # 모델 저장
                if save_model:
                    saved_path = self.deep_model.save()
                    logger.info(f"LSTM 모델 저장 완료: {saved_path}")
                
                # 성능 지표
                metrics = {
                    'accuracy': float(history.history['accuracy'][-1]),
                    'val_accuracy': float(history.history['val_accuracy'][-1]),
                    'loss': float(history.history['loss'][-1]),
                    'val_loss': float(history.history['val_loss'][-1])
                }
            
            elif self.model_type == 'transformer':
                if self.deep_model is None:
                    self.deep_model = TransformerModel(model_path=os.path.join(self.model_dir, 'deep_learning'))
                
                # 모델 구축
                self.deep_model.build_model(input_shape=(window_size, X.shape[1]), num_classes=len(np.unique(y)))
                
                # 모델 학습
                history = self.deep_model.train(X_train, y_train, window_size=window_size, epochs=50, batch_size=32)
                
                # 모델 저장
                if save_model:
                    saved_path = self.deep_model.save()
                    logger.info(f"Transformer 모델 저장 완료: {saved_path}")
                
                # 성능 지표
                metrics = {
                    'accuracy': float(history.history['accuracy'][-1]),
                    'val_accuracy': float(history.history['val_accuracy'][-1]),
                    'loss': float(history.history['loss'][-1]),
                    'val_loss': float(history.history['val_loss'][-1])
                }
        
        # 학습 완료 시간 기록
        self.last_training_date = datetime.now()
        
        # 성능 히스토리 업데이트
        self.performance_history.append({
            'date': self.last_training_date.strftime('%Y-%m-%d %H:%M'),
            'model_type': self.model_type,
            'metrics': metrics,
            'feature_count': X.shape[1],
            'sample_count': X.shape[0]
        })
        
        # 성능 히스토리 저장
        history_path = os.path.join(self.model_dir, 'performance_history.joblib')
        joblib.dump(self.performance_history, history_path)
        logger.info(f"성능 히스토리 저장 완료: {history_path}")
        
        # 결과 반환
        training_result = {
            'training_date': self.last_training_date,
            'metrics': metrics,
            'feature_names': X.columns.tolist(),
            'feature_count': X.shape[1],
            'sample_count': X.shape[0]
        }
        
        return training_result
    
    def evaluate_model(self, X_test, y_test):
        """
        모델 평가
        
        Args:
            X_test (pd.DataFrame): 테스트 특성 데이터
            y_test (pd.Series): 테스트 타깃 데이터
            
        Returns:
            dict: 평가 결과
        """
        if self.model_type == 'random_forest':
            if self.model is None:
                logger.error("평가할 모델이 없습니다.")
                return None
            
            # 모델 평가
            metrics = self.model.evaluate(X_test, y_test)
            
        else:  # 딥러닝 모델
            if self.deep_model is None:
                logger.error("평가할 딥러닝 모델이 없습니다.")
                return None
            
            # 예측 수행
            window_size = 10 if self.model_type == 'lstm' else 20
            y_pred = self.deep_model.predict(X_test, window_size=window_size)
            
            # 성능 지표 계산
            metrics = {
                'accuracy': accuracy_score(y_test.iloc[window_size:], y_pred),
                'f1': f1_score(y_test.iloc[window_size:], y_pred, average='weighted'),
                'precision': precision_score(y_test.iloc[window_size:], y_pred, average='weighted'),
                'recall': recall_score(y_test.iloc[window_size:], y_pred, average='weighted')
            }
        
        # 평가 결과
        evaluation_result = {
            'evaluation_date': datetime.now(),
            'metrics': metrics
        }
        
        logger.info(f"모델 평가 결과: 정확도={metrics['accuracy']:.4f}, F1={metrics.get('f1', 0):.4f}")
        return evaluation_result
    
    def check_performance_decay(self):
        """
        모델 성능 저하 검사
        
        Returns:
            bool: 성능이 저하되었는지 여부
        """
        if not self.performance_history:
            logger.info("성능 히스토리가 없어 성능 저하를 확인할 수 없습니다.")
            return False
        
        # 가장 최근 성능
        latest_performance = self.performance_history[-1]
        
        # 성능 저하 확인 기준: 정확도가 임계값보다 낮거나, F1 점수가 20% 이상 감소
        if latest_performance['metrics'].get('accuracy', 0) < self.performance_threshold:
            logger.info(f"현재 정확도({latest_performance['metrics'].get('accuracy', 0):.4f})가 임계값({self.performance_threshold:.4f})보다 낮습니다.")
            return True
        
        # 이전 성능이 두 개 이상 있는 경우 비교
        if len(self.performance_history) >= 2:
            previous_performance = self.performance_history[-2]
            
            # 정확도 비교
            current_accuracy = latest_performance['metrics'].get('accuracy', 0)
            previous_accuracy = previous_performance['metrics'].get('accuracy', 0)
            
            accuracy_change = (current_accuracy - previous_accuracy) / previous_accuracy
            
            # F1 점수 비교
            current_f1 = latest_performance['metrics'].get('f1', 0)
            previous_f1 = previous_performance['metrics'].get('f1', 0)
            
            f1_change = (current_f1 - previous_f1) / max(0.001, previous_f1)
            
            # 성능 저하 기준
            if accuracy_change < -0.1 or f1_change < -0.2:
                logger.info(f"성능 저하 감지: 정확도 변화={accuracy_change:.2%}, F1 변화={f1_change:.2%}")
                return True
        
        return False
    
    def auto_retraining_check(self, stock_codes):
        """
        자동 재학습 점검
        
        Args:
            stock_codes (list): 종목 코드 리스트
            
        Returns:
            bool: 재학습 수행 여부
        """
        logger.info("자동 재학습 점검 시작")
        self.last_check_date = datetime.now()
        
        # 재학습 필요성 확인
        need_retraining = False
        
        # 모델이 없거나 만료된 경우
        if (self.model is None and self.deep_model is None) or \
           (self.last_training_date is None) or \
           (datetime.now() - self.last_training_date).days > 30:  # 모델이 30일 이상 지난 경우
            logger.info("모델이 없거나 오래되어 재학습이 필요합니다.")
            need_retraining = True
        else:
            # 테스트 데이터 수집
            X, y = self.fetch_training_data(stock_codes, days=60, use_advanced_features=True)
            
            if X is not None and y is not None:
                # 모델 평가
                evaluation = self.evaluate_model(X, y)
                
                if evaluation:
                    # 성능 히스토리 업데이트
                    self.performance_history.append({
                        'date': self.last_check_date.strftime('%Y-%m-%d %H:%M'),
                        'model_type': self.model_type,
                        'metrics': evaluation['metrics'],
                        'is_evaluation': True
                    })
                    
                    # 성능 히스토리 저장
                    history_path = os.path.join(self.model_dir, 'performance_history.joblib')
                    joblib.dump(self.performance_history, history_path)
                    
                    # 성능 저하 확인
                    need_retraining = self.check_performance_decay()
        
        # 재학습 수행
        if need_retraining:
            logger.info("모델 재학습 시작")
            
            # 학습 데이터 수집
            X, y = self.fetch_training_data(stock_codes, days=300, use_advanced_features=True)
            
            if X is not None and y is not None:
                # 모델 학습
                training_result = self.train_model(X, y, test_size=0.2, save_model=True)
                
                if training_result:
                    logger.info("모델 재학습 완료")
                    return True
                else:
                    logger.error("모델 재학습 실패")
            else:
                logger.error("학습 데이터 수집 실패")
        else:
            logger.info("모델 성능이 양호하여 재학습이 필요하지 않습니다.")
        
        return need_retraining
    
    def start_scheduled_retraining(self, stock_codes, check_time="00:01"):
        """
        주기적 재학습 스케줄링 시작
        
        Args:
            stock_codes (list): 종목 코드 리스트
            check_time (str): 매일 점검 시간 (HH:MM)
        """
        if self.is_running:
            logger.warning("이미 스케줄링이 실행 중입니다.")
            return
        
        # 자동 재학습 작업 설정
        def job():
            try:
                self.auto_retraining_check(stock_codes)
            except Exception as e:
                logger.error(f"자동 재학습 작업 중 오류: {str(e)}")
        
        # 매일 지정된 시간에 실행
        schedule.every().day.at(check_time).do(job)
        
        # 스케줄링 스레드 시작
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger.info(f"주기적 재학습 스케줄링 시작 (매일 {check_time})")
    
    def _run_scheduler(self):
        """스케줄러 실행 루프"""
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)
    
    def stop_scheduled_retraining(self):
        """주기적 재학습 스케줄링 중지"""
        if not self.is_running:
            logger.warning("스케줄링이 실행 중이지 않습니다.")
            return
        
        # 스케줄링 중지
        self.is_running = False
        
        # 모든 예약된 작업 취소
        schedule.clear()
        
        # 스레드 종료 대기
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        
        logger.info("주기적 재학습 스케줄링 중지")
    
    def get_model_info(self):
        """
        현재 모델 정보 반환
        
        Returns:
            dict: 모델 정보
        """
        if self.model_type == 'random_forest' and self.model:
            model_info = self.model.get_model_info()
        elif self.deep_model:
            model_info = self.deep_model.get_model_info()
        else:
            model_info = {
                'model_type': 'None',
                'last_training': 'Never',
                'metrics': {},
                'feature_names': []
            }
        
        # 추가 정보
        model_info['last_check_date'] = self.last_check_date.strftime('%Y-%m-%d %H:%M') if self.last_check_date else 'Never'
        model_info['performance_history'] = self.performance_history[-10:] if len(self.performance_history) > 10 else self.performance_history
        
        return model_info
