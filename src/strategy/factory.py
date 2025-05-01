import logging
import os
import importlib.util

logger = logging.getLogger(__name__)

class StrategyFactory:
    """전략 팩토리 클래스
    
    전략 유형에 따라 적절한 전략 객체를 생성하는 팩토리 클래스입니다.
    """
    
    @staticmethod
    def create_strategy(strategy_type, market_data, order_api, config=None, ml_model=None):
        """전략 생성
        
        Args:
            strategy_type (str): 전략 유형 ('basic', 'day_trading', 'high_frequency', 'ml_high_frequency')
            market_data (MarketData): 시장 데이터 객체
            order_api (OrderAPI): 주문 API 객체
            config (dict, optional): 전략 설정
            ml_model (object, optional): ML 모델 객체
            
        Returns:
            Strategy: 생성된 전략 객체
        """
        # 필요한 모듈 가져오기
        from .basic_strategy import BasicStrategy
        
        # 선택적 모듈 가져오기 - 모듈 존재 여부 확인
        day_trading_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                      "day_trading_strategy.py")
        high_frequency_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                         "high_frequency_strategy.py")
        ml_high_frequency_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                            "ml_high_frequency_strategy.py")
        
        # 동적 모듈 로드
        DayTradingStrategy = None
        HighFrequencyStrategy = None
        MLHighFrequencyStrategy = None
        
        if os.path.exists(day_trading_path):
            try:
                spec = importlib.util.spec_from_file_location("day_trading_strategy", day_trading_path)
                day_trading_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(day_trading_module)
                DayTradingStrategy = day_trading_module.DayTradingStrategy
            except Exception as e:
                logger.warning(f"일일 트레이딩 전략 모듈 로드 중 오류: {str(e)}")
        
        if os.path.exists(high_frequency_path):
            try:
                spec = importlib.util.spec_from_file_location("high_frequency_strategy", high_frequency_path)
                high_frequency_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(high_frequency_module)
                HighFrequencyStrategy = high_frequency_module.HighFrequencyStrategy
            except Exception as e:
                logger.warning(f"고빈도 전략 모듈 로드 중 오류: {str(e)}")
        
        if os.path.exists(ml_high_frequency_path):
            try:
                spec = importlib.util.spec_from_file_location("ml_high_frequency_strategy", ml_high_frequency_path)
                ml_high_frequency_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(ml_high_frequency_module)
                MLHighFrequencyStrategy = ml_high_frequency_module.MLHighFrequencyStrategy
            except Exception as e:
                logger.warning(f"ML 고빈도 전략 모듈 로드 중 오류: {str(e)}")
        
        # 전략 생성 (대소문자 구분 없이 비교)
        if strategy_type.lower() == 'basic':
            logger.info("기본 전략을 생성합니다.")
            return BasicStrategy(market_data, order_api, config)
        elif strategy_type.lower() == 'day_trading' and DayTradingStrategy:
            logger.info("일일 트레이딩 전략을 생성합니다.")
            return DayTradingStrategy(market_data, order_api, config)
        elif strategy_type.lower() == 'high_frequency' and HighFrequencyStrategy:
            logger.info("고빈도 전략을 생성합니다.")
            return HighFrequencyStrategy(market_data, order_api, config)
        elif strategy_type.lower() == 'ml_high_frequency' and MLHighFrequencyStrategy and ml_model:
            logger.info("ML 고빈도 전략을 생성합니다.")
            return MLHighFrequencyStrategy(market_data, order_api, ml_model, config)
        else:
            # 기본 전략 반환
            logger.info(f"요청한 전략 유형({strategy_type})이 유효하지 않거나 사용할 수 없습니다. 기본 전략을 사용합니다.")
            return BasicStrategy(market_data, order_api, config)