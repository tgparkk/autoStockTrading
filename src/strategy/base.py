from abc import ABC, abstractmethod

class Strategy(ABC):
    """전략 기본 인터페이스
    
    이 추상 클래스는 주식 매매 전략의 기본 인터페이스를 정의합니다.
    모든 전략 클래스는 이 인터페이스를 구현해야 합니다.
    """
    
    @abstractmethod
    def analyze_stock(self, stock_code, **kwargs):
        """종목 분석
        
        주어진 종목의 매매 신호를 분석합니다.
        
        Args:
            stock_code (str): 종목 코드
            **kwargs: 추가 인자
            
        Returns:
            dict: 분석 결과
        """
        pass
    
    @abstractmethod
    def should_buy(self, stock_code, **kwargs):
        """매수 결정
        
        주어진 종목의 매수 여부를 결정합니다.
        
        Args:
            stock_code (str): 종목 코드
            **kwargs: 추가 인자
            
        Returns:
            tuple: (매수 여부, 사유)
        """
        pass
    
    @abstractmethod
    def should_sell(self, stock_code, **kwargs):
        """매도 결정
        
        주어진 종목의 매도 여부를 결정합니다.
        
        Args:
            stock_code (str): 종목 코드
            **kwargs: 추가 인자
            
        Returns:
            tuple: (매도 여부, 사유)
        """
        pass
    
    @abstractmethod
    def run(self, target_stocks):
        """전략 실행
        
        대상 종목 리스트에 대해 전략을 실행합니다.
        
        Args:
            target_stocks (list): 대상 종목 코드 리스트
            
        Returns:
            dict: 실행 결과
        """
        pass