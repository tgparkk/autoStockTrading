import os
import pandas as pd
import numpy as np
import requests
import re
import json
from datetime import datetime, timedelta
import logging
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from textblob import TextBlob
import joblib

logger = logging.getLogger(__name__)

class NewsSentimentAnalyzer:
    """뉴스 데이터 수집 및 감성 분석 클래스"""
    
    def __init__(self, model_path="models/sentiment"):
        """
        Args:
            model_path (str): 감성 분석 모델 저장 경로
        """
        self.model_path = model_path
        self.sentiment_model = None
        self.vectorizer = None
        self.tfidf_transformer = None
        self.okt = Okt()  # 한국어 형태소 분석기
        self._ensure_dir()
        
        # 감성 사전 (한국어)
        self.pos_words = set()
        self.neg_words = set()
        self._load_sentiment_dict()
    
    def _ensure_dir(self):
        """디렉토리 생성"""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
    
    def _load_sentiment_dict(self):
        """감성 사전 로드 (미리 준비된 긍정/부정 단어)"""
        # 긍정 단어 예시
        pos_words = [
            '상승', '호황', '최고', '급등', '성장', '개선', '상향', '호재', '강세', '돌파',
            '증가', '획득', '확대', '성공', '달성', '우수', '양호', '흑자', '수익', '이익',
            '매수', '추천', '전망', '기대', '상승세', '호전', '개선세', '긍정', '진출', '계약'
        ]
        
        # 부정 단어 예시
        neg_words = [
            '하락', '약세', '부진', '하향', '손실', '적자', '부정', '악재', '감소', '침체',
            '하락세', '저조', '위기', '충격', '급락', '폭락', '붕괴', '실패', '위험', '우려',
            '매도', '손해', '적자', '하락', '폭락', '감소', '둔화', '악화', '부실', '악재'
        ]
        
        self.pos_words = set(pos_words)
        self.neg_words = set(neg_words)
    
    def collect_news(self, stock_code, days=7, max_items=50):
        """
        종목 관련 뉴스 수집
        
        Args:
            stock_code (str): 종목 코드
            days (int): 조회 기간(일)
            max_items (int): 최대 뉴스 항목 수
            
        Returns:
            pd.DataFrame: 뉴스 데이터프레임 (날짜, 제목, 내용)
        """
        try:
            # 여기서는 샘플 데이터만 생성 (실제 API 연동 필요)
            # 실제 구현에서는 네이버 금융, 다음 금융 등의 API 사용 가능
            
            # 임시 데이터
            today = datetime.now()
            news_data = []
            
            # 샘플 뉴스 제목과 내용
            sample_news = [
                {
                    'title': f'{stock_code} 주가 상승세, 실적 개선 기대감',
                    'content': '분기 실적 발표를 앞두고 시장 기대감이 높아지고 있다. 애널리스트들은 작년 대비 매출 15% 성장을 예상한다.',
                    'sentiment': 'positive'
                },
                {
                    'title': f'{stock_code} 신규 사업 진출 발표, 주가 급등',
                    'content': '신규 사업 진출 소식에 투자자들의 매수세가 이어지고 있다. 시장 점유율 확대가 예상된다.',
                    'sentiment': 'positive'
                },
                {
                    'title': f'{stock_code} 실적 부진 우려, 주가 하락',
                    'content': '글로벌 경기 침체 영향으로 실적 악화 우려가 커지고 있다. 매출 감소가 예상된다.',
                    'sentiment': 'negative'
                },
                {
                    'title': f'{stock_code} 경쟁사 신제품 출시, 시장 반응 주목',
                    'content': '경쟁사의 공격적인 시장 확대로 경쟁이 심화될 것으로 보인다. 시장 점유율 변화가 예상된다.',
                    'sentiment': 'neutral'
                },
                {
                    'title': f'{stock_code} 분기 실적 예상치 상회',
                    'content': '예상보다 높은 분기 실적을 발표했다. 주당 순이익이 전년 대비 12% 증가했다.',
                    'sentiment': 'positive'
                }
            ]
            
            # 임의 날짜로 뉴스 데이터 생성
            for i in range(min(max_items, 20)):  # 최대 20개 샘플 생성
                news_date = today - timedelta(days=i % days)
                news_idx = i % len(sample_news)
                
                news_data.append({
                    'date': news_date.strftime('%Y-%m-%d'),
                    'title': sample_news[news_idx]['title'],
                    'content': sample_news[news_idx]['content'],
                    'url': f'https://example.com/news/{stock_code}/{i}',
                    'source': 'Sample News',
                    'sentiment': sample_news[news_idx]['sentiment']
                })
            
            return pd.DataFrame(news_data)
        
        except Exception as e:
            logger.error(f"뉴스 수집 중 오류: {str(e)}")
            return pd.DataFrame(columns=['date', 'title', 'content', 'url', 'source'])
    
    def rule_based_sentiment(self, text):
        """
        규칙 기반 감성 분석
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            float: 감성 점수 (-1: 매우 부정 ~ +1: 매우 긍정)
        """
        # 형태소 분석
        tokens = self.okt.morphs(text)
        
        # 긍정/부정 단어 카운트
        pos_count = sum(1 for word in tokens if word in self.pos_words)
        neg_count = sum(1 for word in tokens if word in self.neg_words)
        
        # 총 단어 수
        total_count = max(1, pos_count + neg_count)
        
        # 감성 점수 계산
        sentiment_score = (pos_count - neg_count) / total_count
        
        return sentiment_score
    
    def analyze_sentiment(self, news_df):
        """
        뉴스 감성 분석
        
        Args:
            news_df (pd.DataFrame): 뉴스 데이터프레임
            
        Returns:
            pd.DataFrame: 감성 점수가 추가된 데이터프레임
        """
        result = news_df.copy()
        
        # 분석 결과 컬럼 추가
        result['sentiment_score'] = 0.0
        result['sentiment_label'] = 'neutral'
        
        # 각 뉴스 항목 분석
        for idx, row in result.iterrows():
            # 제목과 내용 합치기
            text = row['title'] + ' ' + row['content']
            
            # 감성 점수 계산
            score = self.rule_based_sentiment(text)
            
            # TextBlob을 이용한 영문 분석 (영문이 포함된 경우)
            # 한글과 영문 점수의 가중 평균
            english_ratio = len(re.findall(r'[a-zA-Z]', text)) / max(1, len(text))
            if english_ratio > 0.3:  # 영문 비율이 30% 이상인 경우
                blob = TextBlob(text)
                en_score = blob.sentiment.polarity
                score = score * 0.7 + en_score * 0.3
            
            # 점수 및 레이블 저장
            result.loc[idx, 'sentiment_score'] = score
            
            # 레이블 분류
            if score > 0.2:
                result.loc[idx, 'sentiment_label'] = 'positive'
            elif score < -0.2:
                result.loc[idx, 'sentiment_label'] = 'negative'
            else:
                result.loc[idx, 'sentiment_label'] = 'neutral'
        
        return result
    
    def calculate_sentiment_stats(self, news_df):
        """
        뉴스 감성 통계 계산
        
        Args:
            news_df (pd.DataFrame): 감성이 분석된 뉴스 데이터프레임
            
        Returns:
            dict: 감성 통계 (평균, 최근 변화, 긍정/부정/중립 비율 등)
        """
        if 'sentiment_score' not in news_df.columns or len(news_df) == 0:
            return {
                'avg_sentiment': 0,
                'sentiment_change': 0,
                'positive_ratio': 0,
                'negative_ratio': 0,
                'neutral_ratio': 0,
                'sentiment_volume': 0
            }
        
        # 날짜 변환
        news_df['date'] = pd.to_datetime(news_df['date'])
        
        # 날짜별 정렬
        sorted_news = news_df.sort_values('date', ascending=False)
        
        # 전체 평균 감성
        avg_sentiment = sorted_news['sentiment_score'].mean()
        
        # 최근 5개 vs 이전 5개 감성 변화
        recent_sentiment = sorted_news.head(min(5, len(sorted_news)))['sentiment_score'].mean()
        if len(sorted_news) > 5:
            older_sentiment = sorted_news.iloc[5:10]['sentiment_score'].mean()
            sentiment_change = recent_sentiment - older_sentiment
        else:
            sentiment_change = 0
        
        # 감성 레이블 비율
        total_news = len(sorted_news)
        positive_count = len(sorted_news[sorted_news['sentiment_label'] == 'positive'])
        negative_count = len(sorted_news[sorted_news['sentiment_label'] == 'negative'])
        neutral_count = len(sorted_news[sorted_news['sentiment_label'] == 'neutral'])
        
        positive_ratio = positive_count / total_news
        negative_ratio = negative_count / total_news
        neutral_ratio = neutral_count / total_news
        
        # 뉴스 볼륨에 가중치를 준 감성 지표
        sentiment_volume = avg_sentiment * np.log1p(len(sorted_news))
        
        return {
            'avg_sentiment': avg_sentiment,
            'sentiment_change': sentiment_change,
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'neutral_ratio': neutral_ratio,
            'sentiment_volume': sentiment_volume
        }
    
    def get_news_features(self, stock_code, days=7):
        """
        종목별 뉴스 특성 생성
        
        Args:
            stock_code (str): 종목 코드
            days (int): 뉴스 조회 기간
            
        Returns:
            dict: 뉴스 기반 특성
        """
        # 1. 뉴스 수집
        news_df = self.collect_news(stock_code, days=days)
        
        # 2. 감성 분석
        if len(news_df) > 0:
            news_df = self.analyze_sentiment(news_df)
            
            # 3. 감성 통계 계산
            sentiment_stats = self.calculate_sentiment_stats(news_df)
            
            # 4. 특성화
            features = {
                'news_sentiment_score': sentiment_stats['avg_sentiment'],
                'news_sentiment_change': sentiment_stats['sentiment_change'],
                'news_positive_ratio': sentiment_stats['positive_ratio'],
                'news_negative_ratio': sentiment_stats['negative_ratio'],
                'news_sentiment_volume': sentiment_stats['sentiment_volume'],
                'news_count': len(news_df)
            }
            
            return features
        else:
            # 뉴스가 없는 경우 기본값 반환
            return {
                'news_sentiment_score': 0,
                'news_sentiment_change': 0,
                'news_positive_ratio': 0,
                'news_negative_ratio': 0,
                'news_sentiment_volume': 0,
                'news_count': 0
            }
    
    def add_news_features_to_df(self, df, stock_code, date_column='stck_bsop_date'):
        """
        데이터프레임에 뉴스 특성 추가
        
        Args:
            df (pd.DataFrame): 주가 데이터프레임
            stock_code (str): 종목 코드
            date_column (str): 날짜 컬럼명
            
        Returns:
            pd.DataFrame: 뉴스 특성이 추가된 데이터프레임
        """
        result = df.copy()
        
        # 날짜를 datetime 형식으로 변환
        result[date_column] = pd.to_datetime(result[date_column])
        
        # 날짜 범위
        start_date = result[date_column].min()
        end_date = result[date_column].max()
        days = (end_date - start_date).days + 1
        
        # 뉴스 특성 가져오기
        news_features = self.get_news_features(stock_code, days=days)
        
        # 임시 방법: 모든 행에 동일한 뉴스 특성 추가
        # 실제로는 날짜별로 뉴스를 수집하고 해당 날짜에 맞는 특성을 추가해야 함
        for feature, value in news_features.items():
            result[feature] = value
        
        return result
