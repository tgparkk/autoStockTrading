import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import shap
import lime
import lime.lime_tabular
import joblib
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelExplainer:
    """머신러닝 모델 설명 클래스 (SHAP, LIME 활용)"""
    
    def __init__(self, model_path="models/explainer"):
        """
        Args:
            model_path (str): 설명자 모델 저장 경로
        """
        self.model_path = model_path
        self.explainer = None
        self.model = None
        self.feature_names = None
        self._ensure_dir()
    
    def _ensure_dir(self):
        """디렉토리 생성"""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
    
    def setup_shap_explainer(self, model, X_train, feature_names=None):
        """
        SHAP 설명자 설정
        
        Args:
            model: 설명할 모델 객체
            X_train (pd.DataFrame): 학습 데이터
            feature_names (list, optional): 특성 이름 목록
            
        Returns:
            shap.Explainer: SHAP 설명자
        """
        try:
            # 랜덤 포레스트, XGBoost 등 트리 기반 모델인 경우
            if hasattr(model, 'estimators_') or hasattr(model, 'booster'):
                explainer = shap.TreeExplainer(model)
            # 일반적인 ML 모델인 경우
            else:
                # KernelExplainer 사용 (가장 범용적)
                explainer = shap.KernelExplainer(model.predict, X_train)
            
            self.explainer = explainer
            self.model = model
            self.feature_names = feature_names if feature_names else X_train.columns.tolist()
            
            return explainer
        except Exception as e:
            logger.error(f"SHAP 설명자 설정 중 오류: {str(e)}")
            return None
    
    def setup_lime_explainer(self, X_train, feature_names=None, class_names=None):
        """
        LIME 설명자 설정
        
        Args:
            X_train (pd.DataFrame): 학습 데이터
            feature_names (list, optional): 특성 이름 목록
            class_names (list, optional): 클래스 이름 목록
            
        Returns:
            LimeTabularExplainer: LIME 설명자
        """
        try:
            # 설정
            feature_names = feature_names if feature_names else X_train.columns.tolist()
            
            # LIME 설명자 생성
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=feature_names,
                class_names=class_names,
                mode='classification' if class_names else 'regression'
            )
            
            self.lime_explainer = explainer
            self.feature_names = feature_names
            
            return explainer
        except Exception as e:
            logger.error(f"LIME 설명자 설정 중 오류: {str(e)}")
            return None
    
    def save_explainer(self, filename=None):
        """
        설명자 저장
        
        Args:
            filename (str, optional): 파일명 (기본값: shap_explainer_{현재날짜}.joblib)
            
        Returns:
            str: 저장된 파일 경로
        """
        if self.explainer is None:
            raise ValueError("저장할 설명자가 없습니다.")
        
        if filename is None:
            today = datetime.now().strftime('%Y%m%d')
            filename = f"shap_explainer_{today}.joblib"
        
        file_path = os.path.join(self.model_path, filename)
        
        # 설명자 저장
        joblib.dump(self.explainer, file_path)
        
        # 특성 이름 저장
        meta_file = os.path.splitext(file_path)[0] + '_meta.joblib'
        metadata = {
            'feature_names': self.feature_names,
            'model_type': type(self.model).__name__ if self.model else None
        }
        joblib.dump(metadata, meta_file)
        
        logger.info(f"설명자 모델이 저장되었습니다: {file_path}")
        return file_path
    
    def load_explainer(self, filepath):
        """
        설명자 로드
        
        Args:
            filepath (str): 설명자 파일 경로
            
        Returns:
            shap.Explainer: 로드된 설명자
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"설명자 파일을 찾을 수 없습니다: {filepath}")
        
        # 설명자 로드
        self.explainer = joblib.load(filepath)
        
        # 메타데이터 로드
        meta_file = os.path.splitext(filepath)[0] + '_meta.joblib'
        if os.path.exists(meta_file):
            metadata = joblib.load(meta_file)
            self.feature_names = metadata.get('feature_names')
        
        logger.info(f"설명자 모델이 로드되었습니다: {filepath}")
        return self.explainer
    
    def explain_prediction_with_shap(self, instance, plot_type='bar'):
        """
        SHAP를 사용한 예측 설명
        
        Args:
            instance: 설명할 인스턴스 (데이터프레임 또는 배열)
            plot_type (str): 플롯 유형 ('bar', 'waterfall', 'force')
            
        Returns:
            dict: 설명 결과 및 시각화 Figure 객체
        """
        if self.explainer is None:
            raise ValueError("설명자가 설정되지 않았습니다.")
        
        # 인스턴스 형식 확인 및 변환
        if isinstance(instance, pd.DataFrame):
            instance_values = instance.values
        else:
            instance_values = np.array(instance).reshape(1, -1)
        
        try:
            # SHAP 값 계산
            shap_values = self.explainer.shap_values(instance_values)
            
            # shap_values 형식에 따라 처리
            if isinstance(shap_values, list):
                # 여러 클래스에 대한 SHAP 값 (분류 문제)
                # 여기서는 첫 번째 클래스의 SHAP 값을 사용
                shap_values = shap_values[0] if len(shap_values) > 0 else shap_values
            
            # 특성 중요도 계산
            feature_importance = {}
            for i, feature in enumerate(self.feature_names):
                if shap_values.ndim > 1:
                    importance = abs(shap_values[0][i])
                else:
                    importance = abs(shap_values[i])
                feature_importance[feature] = float(importance)
            
            # 정렬된 특성 중요도
            sorted_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            # 시각화
            plt.figure(figsize=(10, 6))
            
            if plot_type == 'bar':
                # Bar 플롯
                if shap_values.ndim > 1:
                    shap.summary_plot(shap_values, instance, feature_names=self.feature_names, plot_type='bar', show=False)
                else:
                    shap.summary_plot(shap_values.reshape(1, -1), instance.reshape(1, -1), feature_names=self.feature_names, plot_type='bar', show=False)
            elif plot_type == 'waterfall':
                # Waterfall 플롯
                if isinstance(instance_values, np.ndarray) and instance_values.ndim > 1:
                    shap.plots.waterfall(shap.Explanation(values=shap_values[0], 
                                                         base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0, 
                                                         data=instance_values[0], 
                                                         feature_names=self.feature_names), 
                                         show=False)
                else:
                    shap.plots.waterfall(shap.Explanation(values=shap_values, 
                                                         base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0, 
                                                         data=instance_values, 
                                                         feature_names=self.feature_names), 
                                         show=False)
            elif plot_type == 'force':
                # Force 플롯
                if shap_values.ndim > 1:
                    shap.initjs()
                    shap.force_plot(
                        self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
                        shap_values[0], 
                        instance_values[0] if instance_values.ndim > 1 else instance_values,
                        feature_names=self.feature_names,
                        matplotlib=True,
                        show=False
                    )
                else:
                    shap.initjs()
                    shap.force_plot(
                        self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
                        shap_values, 
                        instance_values,
                        feature_names=self.feature_names,
                        matplotlib=True,
                        show=False
                    )
            
            # 현재 Figure 객체 가져오기
            fig = plt.gcf()
            plt.tight_layout()
            
            return {
                'feature_importance': sorted_importance,
                'shap_values': shap_values,
                'plot': fig
            }
            
        except Exception as e:
            logger.error(f"SHAP 설명 생성 중 오류: {str(e)}")
            return None
    
    def explain_prediction_with_lime(self, instance, predict_fn=None, num_features=10):
        """
        LIME을 사용한 예측 설명
        
        Args:
            instance: 설명할 인스턴스
            predict_fn: 예측 함수 (None이면 모델의 predict_proba 사용)
            num_features (int): 설명에 포함할 특성 수
            
        Returns:
            dict: 설명 결과 및 시각화 Figure 객체
        """
        if not hasattr(self, 'lime_explainer'):
            raise ValueError("LIME 설명자가 설정되지 않았습니다.")
        
        # 예측 함수 설정
        if predict_fn is None:
            if self.model is None:
                raise ValueError("모델 또는 예측 함수가 제공되어야 합니다.")
            
            if hasattr(self.model, 'predict_proba'):
                predict_fn = self.model.predict_proba
            else:
                predict_fn = self.model.predict
        
        try:
            # 인스턴스 형식 확인 및 변환
            if isinstance(instance, pd.DataFrame):
                instance_values = instance.values[0]
            elif isinstance(instance, np.ndarray) and instance.ndim > 1:
                instance_values = instance[0]
            else:
                instance_values = instance
            
            # LIME 설명 생성
            explanation = self.lime_explainer.explain_instance(
                instance_values, 
                predict_fn, 
                num_features=num_features
            )
            
            # 시각화
            fig = plt.figure(figsize=(10, 6))
            explanation.as_pyplot_figure(fig=fig)
            plt.tight_layout()
            
            # 특성별 중요도 추출
            feature_importance = {}
            for feature, importance in explanation.as_list():
                feature_importance[feature] = abs(importance)
            
            # 정렬된 특성 중요도
            sorted_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            return {
                'feature_importance': sorted_importance,
                'explanation': explanation,
                'plot': fig
            }
            
        except Exception as e:
            logger.error(f"LIME 설명 생성 중 오류: {str(e)}")
            return None
    
    def generate_global_feature_importance(self, X_sample=None):
        """
        글로벌 특성 중요도 생성
        
        Args:
            X_sample (pd.DataFrame, optional): 샘플 데이터 (None이면 모델의 특성 중요도 사용)
            
        Returns:
            dict: 특성 중요도 및 시각화 Figure 객체
        """
        try:
            feature_importance = {}
            
            # SHAP 기반 중요도 계산 (샘플이 제공된 경우)
            if X_sample is not None and self.explainer is not None:
                # 샘플 크기 제한 (처리 시간 감소)
                sample_size = min(100, len(X_sample))
                X_subset = X_sample.sample(sample_size) if isinstance(X_sample, pd.DataFrame) else X_sample[:sample_size]
                
                # SHAP 값 계산
                shap_values = self.explainer.shap_values(X_subset)
                
                # shap_values 형식에 따라 처리
                if isinstance(shap_values, list):
                    # 여러 클래스에 대한 SHAP 값 평균
                    avg_shap = np.abs(np.mean([np.abs(sv) for sv in shap_values], axis=0))
                else:
                    # 단일 출력 SHAP 값
                    avg_shap = np.abs(np.mean(np.abs(shap_values), axis=0))
                
                # 특성별 중요도 계산
                for i, feature in enumerate(self.feature_names):
                    feature_importance[feature] = float(avg_shap[i])
            
            # 모델 자체의 특성 중요도 (가능한 경우)
            elif self.model is not None and hasattr(self.model, 'feature_importances_'):
                for i, feature in enumerate(self.feature_names):
                    feature_importance[feature] = float(self.model.feature_importances_[i])
            
            # 정렬된 특성 중요도
            sorted_importance = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            # 시각화
            fig = plt.figure(figsize=(12, 6))
            
            # Bar 플롯
            if sorted_importance:
                features = list(sorted_importance.keys())
                values = list(sorted_importance.values())
                
                # 너무 많은 특성은 상위 20개만 표시
                if len(features) > 20:
                    features = features[:20]
                    values = values[:20]
                
                plt.barh(features, values, color='skyblue')
                plt.xlabel('Feature Importance')
                plt.ylabel('Features')
                plt.title('Global Feature Importance')
                plt.gca().invert_yaxis()  # 중요도 높은 순으로 표시
                plt.tight_layout()
            
            return {
                'feature_importance': sorted_importance,
                'plot': fig
            }
            
        except Exception as e:
            logger.error(f"글로벌 특성 중요도 생성 중 오류: {str(e)}")
            return None
    
    def visualize_feature_interactions(self, X_sample, feature1, feature2):
        """
        두 특성 간 상호작용 시각화
        
        Args:
            X_sample (pd.DataFrame): 샘플 데이터
            feature1 (str): 첫 번째 특성 이름
            feature2 (str): 두 번째 특성 이름
            
        Returns:
            matplotlib.figure.Figure: 상호작용 시각화 Figure 객체
        """
        if self.explainer is None:
            raise ValueError("설명자가 설정되지 않았습니다.")
        
        try:
            # 샘플 크기 제한
            sample_size = min(300, len(X_sample))
            X_subset = X_sample.sample(sample_size) if isinstance(X_sample, pd.DataFrame) else X_sample[:sample_size]
            
            # SHAP 값 계산
            shap_values = self.explainer.shap_values(X_subset)
            
            # shap_values 형식에 따라 처리
            if isinstance(shap_values, list):
                # 여러 클래스에 대한 SHAP 값 (첫 번째 클래스 사용)
                shap_values = shap_values[0]
            
            # 특성 인덱스 찾기
            idx1 = self.feature_names.index(feature1)
            idx2 = self.feature_names.index(feature2)
            
            # 의존성 플롯
            plt.figure(figsize=(10, 8))
            
            shap.dependence_plot(
                idx1,
                shap_values,
                X_subset,
                interaction_index=idx2,
                feature_names=self.feature_names,
                show=False
            )
            
            plt.title(f'Interaction between {feature1} and {feature2}')
            plt.tight_layout()
            
            # 현재 Figure 객체 가져오기
            fig = plt.gcf()
            
            return fig
            
        except Exception as e:
            logger.error(f"특성 상호작용 시각화 중 오류: {str(e)}")
            return None
    
    def generate_explanation_report(self, instance, X_sample=None):
        """
        종합 설명 리포트 생성
        
        Args:
            instance: 설명할 인스턴스
            X_sample (pd.DataFrame, optional): 글로벌 설명용 샘플 데이터
            
        Returns:
            dict: 설명 보고서
        """
        report = {
            'creation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': type(self.model).__name__ if self.model else 'Unknown',
            'local_explanation': None,
            'global_explanation': None
        }
        
        # 로컬 설명 (단일 예측 설명)
        local_exp = self.explain_prediction_with_shap(instance)
        if local_exp:
            # Figure 객체 저장 (Base64 문자열 또는 파일로 저장 가능)
            fig_path = os.path.join(self.model_path, 'local_explanation.png')
            local_exp['plot'].savefig(fig_path)
            plt.close(local_exp['plot'])
            
            report['local_explanation'] = {
                'feature_importance': local_exp['feature_importance'],
                'visualization_path': fig_path
            }
        
        # 글로벌 설명 (모델 전체 설명)
        if X_sample is not None:
            global_exp = self.generate_global_feature_importance(X_sample)
            if global_exp:
                # Figure 객체 저장
                fig_path = os.path.join(self.model_path, 'global_explanation.png')
                global_exp['plot'].savefig(fig_path)
                plt.close(global_exp['plot'])
                
                report['global_explanation'] = {
                    'feature_importance': global_exp['feature_importance'],
                    'visualization_path': fig_path
                }
        
        return report
