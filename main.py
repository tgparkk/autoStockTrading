import logging
import time
import sys
import os
import yaml
import argparse
from datetime import datetime, timedelta
import schedule

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.auth import KoreaInvestmentAuth
from src.api.market_data import MarketData
from src.api.order import OrderAPI
from src.strategy.basic_strategy import BasicStrategy
from src.utils.logger import setup_logger

def load_config(config_path):
    """설정 파일 로드
    
    Args:
        config_path (str): 설정 파일 경로
        
    Returns:
        dict: 설정 정보
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def load_target_stocks(stocks_path):
    """대상 종목 파일 로드
    
    Args:
        stocks_path (str): 종목 파일 경로
        
    Returns:
        list: 종목 코드 리스트
    """
    with open(stocks_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file if line.strip()]

def is_trading_time():
    """거래 시간 여부 확인
    
    Returns:
        bool: 거래 시간이면 True, 아니면 False
    """
    now = datetime.now()
    
    # 주말 체크
    if now.weekday() >= 5:  # 토요일(5), 일요일(6)
        return False
    
    # 휴장일 체크 (공휴일 등은 별도 리스트로 관리 필요)
    
    # 시간 체크 (9:00 ~ 15:30)
    market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return market_open <= now <= market_close

def trading_job(auth, market_data, order_api, strategy, target_stocks, logger):
    """거래 작업 실행
    
    Args:
        auth (KoreaInvestmentAuth): 인증 객체
        market_data (MarketData): 시장 데이터 객체
        order_api (OrderAPI): 주문 API 객체
        strategy (BasicStrategy): 전략 객체
        target_stocks (list): 대상 종목 리스트
        logger (logging.Logger): 로거 객체
    """
    logger.info("=== 거래 작업 시작 ===")
    
    # 거래 시간 체크
    if not is_trading_time():
        logger.info("현재 거래 시간이 아닙니다.")
        return
    
    # 토큰 갱신 확인
    auth.get_access_token()
    
    # 전략 실행
    results = strategy.run(target_stocks)
    
    # 결과 로깅
    logger.info(f"매수 실행: {len(results['buys'])}건")
    logger.info(f"매도 실행: {len(results['sells'])}건")
    logger.info(f"오류 발생: {len(results['errors'])}건")
    
    logger.info("=== 거래 작업 완료 ===")

def setup_scheduled_jobs(auth, market_data, order_api, strategy, target_stocks, interval_minutes, logger):
    """스케줄 작업 설정
    
    Args:
        auth (KoreaInvestmentAuth): 인증 객체
        market_data (MarketData): 시장 데이터 객체
        order_api (OrderAPI): 주문 API 객체
        strategy (BasicStrategy): 전략 객체
        target_stocks (list): 대상 종목 리스트
        interval_minutes (int): 작업 실행 간격(분)
        logger (logging.Logger): 로거 객체
    """
    logger.info(f"스케줄 작업 설정: {interval_minutes}분 간격")
    
    # 작업 함수 래핑
    def job():
        trading_job(auth, market_data, order_api, strategy, target_stocks, logger)
    
    # 정해진 간격으로 실행
    schedule.every(interval_minutes).minutes.do(job)
    
    # 장 시작 시 실행
    schedule.every().day.at("09:05").do(job)
    
    # 장 마감 전 실행
    schedule.every().day.at("15:20").do(job)

def main():
    """메인 실행 함수"""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='주식 자동매매 프로그램')
    parser.add_argument('--config', default='config/api_config.yaml', help='API 설정 파일 경로')
    parser.add_argument('--strategy', default='config/trading_config.yaml', help='전략 설정 파일 경로')
    parser.add_argument('--stocks', default='config/target_stocks.txt', help='대상 종목 파일 경로')
    parser.add_argument('--interval', type=int, default=2, help='작업 실행 간격(분)')
    parser.add_argument('--log', default='logs', help='로그 디렉토리 경로')
    parser.add_argument('--once', action='store_true', help='한 번만 실행')
    args = parser.parse_args()
    
    # 로깅 설정
    log_dir = os.path.abspath(args.log)
    os.makedirs(log_dir, exist_ok=True)
    # 날짜 형식이 포함된 로그 파일명 사용하지 않음 - setup_logger가 자동으로 추가
    log_file = os.path.join(log_dir, f'trading.log')

    logger = setup_logger(log_file)
    logger.info("===== 주식 자동매매 프로그램 시작 =====")
    
    try:
        # 설정 파일 로드
        logger.info(f"API 설정 파일 로드: {args.config}")
        api_config_path = os.path.abspath(args.config)
        
        logger.info(f"전략 설정 파일 로드: {args.strategy}")
        strategy_config_path = os.path.abspath(args.strategy)
        
        # 대상 종목 로드
        logger.info(f"대상 종목 파일 로드: {args.stocks}")
        stocks_path = os.path.abspath(args.stocks)
        target_stocks = load_target_stocks(stocks_path)
        logger.info(f"대상 종목: {len(target_stocks)}개")
        
        # 전략 설정 로드
        strategy_config = {}
        if os.path.exists(strategy_config_path):
            strategy_config = load_config(strategy_config_path).get('strategy', {})
        
        # API 객체 초기화
        auth = KoreaInvestmentAuth(api_config_path)
        market_data = MarketData(auth, api_config_path)
        order_api = OrderAPI(auth, api_config_path)
        
        # 액세스 토큰 발급
        auth.get_access_token()
        
        # 전략 객체 초기화
        strategy = BasicStrategy(market_data, order_api, strategy_config)
        
        # 한 번만 실행
        if args.once:
            logger.info("단일 실행 모드")
            trading_job(auth, market_data, order_api, strategy, target_stocks, logger)
            return
        
        # 스케줄 작업 설정
        setup_scheduled_jobs(auth, market_data, order_api, strategy, target_stocks, args.interval, logger)
        
        # 메인 루프
        logger.info("스케줄 작업 실행 중...")
        
        while True:
            schedule.run_pending()
            time.sleep(10)
            
    except KeyboardInterrupt:
        logger.info("사용자에 의해 프로그램 종료")
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {str(e)}", exc_info=True)
    finally:
        logger.info("===== 주식 자동매매 프로그램 종료 =====")

if __name__ == "__main__":
    main()