import logging
import time
import sys
import os
import argparse
from datetime import datetime
import schedule

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.api.auth import KoreaInvestmentAuth
from src.api.market_data import MarketData
from src.api.order import OrderAPI
from src.core.config import ConfigManager
from src.core.trading_system import TradingSystem
from src.utils.logger import setup_logger

def parse_args():
    """명령행 인자 파싱
    
    Returns:
        argparse.Namespace: 파싱된 인자
    """
    parser = argparse.ArgumentParser(description='주식 자동매매 프로그램')
    parser.add_argument('--config', default='config/api_config.yaml', help='API 설정 파일 경로')
    parser.add_argument('--strategy', default='config/trading_config.yaml', help='전략 설정 파일 경로')
    parser.add_argument('--stocks', default='config/target_stocks.txt', help='대상 종목 파일 경로')
    parser.add_argument('--strategy-type', default='basic', choices=['basic', 'day_trading', 'high_frequency', 'ml_high_frequency'], help='전략 유형')
    parser.add_argument('--log', default='logs', help='로그 디렉토리 경로')
    parser.add_argument('--once', action='store_true', help='한 번만 실행')
    parser.add_argument('--interval', type=int, default=None, help='작업 실행 간격(분), 설정 시 config 값을 덮어씁니다')
    
    return parser.parse_args()

def trading_job(trading_system, logger):
    """거래 작업 실행
    
    Args:
        trading_system (TradingSystem): 거래 시스템 객체
        logger (logging.Logger): 로거 객체
    """
    logger.info("=== 거래 작업 시작 ===")
    
    # 거래 시간 체크
    if trading_system._is_trading_time():
        # 전략 실행
        target_stocks = trading_system.get_target_stocks()
        trading_system._trading_loop()
    else:
        logger.info("현재 거래 시간이 아닙니다.")
    
    logger.info("=== 거래 작업 완료 ===")

def setup_scheduled_jobs(trading_system, interval_minutes, logger):
    """스케줄 작업 설정
    
    Args:
        trading_system (TradingSystem): 거래 시스템 객체
        interval_minutes (int): 작업 실행 간격(분)
        logger (logging.Logger): 로거 객체
    """
    logger.info(f"스케줄 작업 설정: {interval_minutes}분 간격")
    
    # 작업 함수 래핑
    def job():
        trading_job(trading_system, logger)
    
    # 정해진 간격으로 실행
    schedule.every(interval_minutes).minutes.do(job)
    
    # 장 시작 시 실행
    schedule.every().day.at("09:05").do(job)
    
    # 장 마감 전 실행
    schedule.every().day.at("15:20").do(job)

def main():
    """메인 실행 함수"""
    # 명령행 인자 파싱
    args = parse_args()
    
    # 로깅 설정
    log_dir = os.path.abspath(args.log)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'trading.log')
    logger = setup_logger(log_file)
    
    logger.info("===== 주식 자동매매 프로그램 시작 =====")
    
    try:
        # 설정 파일 경로
        config_dir = os.path.dirname(os.path.abspath(args.config))
        os.makedirs(config_dir, exist_ok=True)
        
        # 설정 관리자 초기화
        config_manager = ConfigManager(config_dir)
        
        # API 객체 초기화
        auth = KoreaInvestmentAuth(args.config)
        market_data = MarketData(auth, args.config)
        order_api = OrderAPI(auth, args.config)
        
        # 액세스 토큰 발급
        auth.get_access_token()
        
        # 거래 시스템 초기화
        trading_system = TradingSystem(
            auth_client=auth,
            market_data=market_data,
            order_api=order_api,
            config_manager=config_manager,
            strategy_type=args.strategy_type
        )
        
        # 인터벌 업데이트 (명령행 인자가 제공된 경우)
        if args.interval is not None:
            config_manager.update_interval(args.interval)
            logger.info(f"실행 간격을 {args.interval}분으로 설정했습니다.")
        
        # 한 번만 실행
        if args.once:
            logger.info("단일 실행 모드")
            trading_job(trading_system, logger)
            return
        
        # 스케줄 작업 설정
        interval = config_manager.get_interval()
        setup_scheduled_jobs(trading_system, interval, logger)
        
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