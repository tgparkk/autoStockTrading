#!/usr/bin/env python
"""
자동 주식 거래 시스템 시작 스크립트

사용법:
    python start.py --mode [cli|web|daemon] [옵션]

모드:
    cli: 명령행 인터페이스 (기본)
    web: 웹 인터페이스
    daemon: 백그라운드 데몬 모드

옵션:
    --config: API 설정 파일 경로
    --strategy-type: 사용할 전략 유형 (basic, day_trading, high_frequency, ml_high_frequency)
    --log-level: 로그 레벨 (debug, info, warning, error)
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.api.auth import KoreaInvestmentAuth
from src.api.market_data import MarketData
from src.api.order import OrderAPI
from src.core.config import ConfigManager
from src.core.trading_system import TradingSystem
from src.utils.logger import setup_logger

def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='자동 주식 거래 시스템')
    
    # 필수 옵션
    parser.add_argument('--mode', default='cli', choices=['cli', 'web', 'daemon'],
                        help='실행 모드 (cli, web, daemon)')
    
    # 일반 옵션
    parser.add_argument('--config', default='config/api_config.yaml',
                        help='API 설정 파일 경로')
    parser.add_argument('--strategy-type', default='basic',
                        choices=['basic', 'day_trading', 'high_frequency', 'ml_high_frequency'],
                        help='전략 유형')
    parser.add_argument('--log-level', default='info',
                        choices=['debug', 'info', 'warning', 'error'],
                        help='로그 레벨')
    
    # 웹 옵션
    parser.add_argument('--host', default='127.0.0.1',
                        help='웹 서버 호스트 (web 모드 전용)')
    parser.add_argument('--port', type=int, default=5000,
                        help='웹 서버 포트 (web 모드 전용)')
    
    return parser.parse_args()

def setup_logging(log_level='info'):
    """로깅 설정"""
    log_dir = os.path.abspath('logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 로그 레벨 매핑
    log_levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR
    }
    
    level = log_levels.get(log_level.lower(), logging.INFO)
    
    # 전역 로거 설정
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 핸들러 설정
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # 파일 핸들러
    log_file = os.path.join(log_dir, 'system.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def initialize_system(args):
    """시스템 초기화"""
    logger = logging.getLogger(__name__)
    logger.info("시스템 초기화 중...")
    
    # 설정 디렉토리 확인
    config_dir = os.path.dirname(os.path.abspath(args.config))
    os.makedirs(config_dir, exist_ok=True)
    
    # 설정 관리자 초기화
    config_manager = ConfigManager(config_dir)
    
    # API 객체 초기화
    try:
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
        
        logger.info("시스템 초기화 완료")
        return trading_system
    except Exception as e:
        logger.error(f"시스템 초기화 실패: {str(e)}")
        raise

def run_cli_mode(trading_system, args):
    """CLI 모드 실행"""
    logger = logging.getLogger(__name__)
    logger.info("CLI 모드 시작")
    
    # 거래 시스템 시작
    trading_system.start()
    
    try:
        # 사용자 명령 처리 루프
        while True:
            cmd = input("\n명령어 입력 (help, status, start, stop, quit): ").strip().lower()
            
            if cmd == 'help':
                print("\n사용 가능한 명령어:")
                print("  status - 현재 시스템 상태 확인")
                print("  start - 거래 시스템 시작")
                print("  stop - 거래 시스템 중지")
                print("  quit - 프로그램 종료")
            
            elif cmd == 'status':
                status = trading_system.get_status()
                print(f"\n현재 상태: {status}")
                
                # 보유 종목 정보
                account_info = trading_system.get_account_info()
                if account_info and 'stocks' in account_info:
                    print("\n보유 종목:")
                    for stock in account_info['stocks']:
                        stock_code = stock.get('pdno', 'N/A')
                        stock_name = stock.get('prdt_name', 'N/A')
                        quantity = stock.get('hldg_qty', 0)
                        avg_price = stock.get('pchs_avg_pric', 0)
                        current_price = stock.get('prpr', 0)
                        
                        print(f"  {stock_code} ({stock_name}): {quantity}주, 평균가: {avg_price}, 현재가: {current_price}")
            
            elif cmd == 'start':
                if trading_system.get_status() == 'stopped':
                    trading_system.start()
                    print("\n거래 시스템이 시작되었습니다.")
                else:
                    print("\n거래 시스템이 이미 실행 중입니다.")
            
            elif cmd == 'stop':
                if trading_system.get_status() != 'stopped':
                    trading_system.stop()
                    print("\n거래 시스템이 중지되었습니다.")
                else:
                    print("\n거래 시스템이 이미 중지되었습니다.")
            
            elif cmd == 'quit':
                trading_system.stop()
                print("\n프로그램을 종료합니다.")
                break
            
            else:
                print("\n알 수 없는 명령어입니다. 'help'를 입력하여 사용 가능한 명령어를 확인하세요.")
    
    except KeyboardInterrupt:
        logger.info("사용자에 의해 프로그램 종료")
    finally:
        # 종료 시 거래 시스템 중지
        trading_system.stop()
        logger.info("CLI 모드 종료")

def run_web_mode(trading_system, args):
    """웹 인터페이스 모드 실행"""
    logger = logging.getLogger(__name__)
    logger.info("웹 인터페이스 모드 시작")
    
    try:
        # 웹 서버 시작 시도
        try:
            from src.web.app import start_web_server
            # 거래 시스템 시작
            trading_system.start()
            # 웹 서버 시작
            start_web_server(trading_system, host=args.host, port=args.port)
        except ImportError:
            logger.error("웹 인터페이스 모듈을 찾을 수 없습니다.")
            logger.info("웹 모듈을 설치하세요: pip install flask")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("사용자에 의해 프로그램 종료")
    finally:
        # 종료 시 거래 시스템 중지
        trading_system.stop()
        logger.info("웹 인터페이스 모드 종료")

def run_daemon_mode(trading_system, args):
    """데몬 모드 실행"""
    logger = logging.getLogger(__name__)
    logger.info("데몬 모드 시작")
    
    try:
        # 데몬 라이브러리 로드 시도
        try:
            import daemon
            import lockfile
        except ImportError:
            logger.error("데몬 라이브러리를 찾을 수 없습니다.")
            logger.info("데몬 라이브러리를 설치하세요: pip install python-daemon")
            sys.exit(1)
        
        # PID 파일 경로
        pid_dir = os.path.abspath('logs')
        os.makedirs(pid_dir, exist_ok=True)
        pid_file = os.path.join(pid_dir, 'trading_daemon.pid')
        
        # 데몬 컨텍스트 설정
        context = daemon.DaemonContext(
            working_directory=os.path.abspath('.'),
            umask=0o002,
            pidfile=lockfile.FileLock(pid_file)
        )
        
        # 로그 파일 설정
        log_file = os.path.join('logs', 'daemon.log')
        context.stdout = open(log_file, 'a+')
        context.stderr = open(log_file, 'a+')
        
        # 데몬으로 실행
        with context:
            logger.info("데몬 모드로 거래 시스템 시작")
            
            # 거래 시스템 시작
            trading_system.start()
            
            # 무한 루프로 실행
            try:
                while True:
                    import time
                    time.sleep(60)  # 1분마다 상태 확인
            except Exception as e:
                logger.error(f"데몬 모드 오류: {str(e)}")
            finally:
                trading_system.stop()
                logger.info("데몬 모드 종료")
    
    except Exception as e:
        logger.error(f"데몬 모드 시작 실패: {str(e)}")
        sys.exit(1)

def main():
    """메인 함수"""
    # 명령행 인자 파싱
    args = parse_args()
    
    # 로깅 설정
    logger = setup_logging(args.log_level)
    logger.info(f"===== 자동 주식 거래 시스템 시작 ({args.mode} 모드) =====")
    
    try:
        # 시스템 초기화
        trading_system = initialize_system(args)
        
        # 모드에 따라 실행
        if args.mode == 'cli':
            run_cli_mode(trading_system, args)
        elif args.mode == 'web':
            run_web_mode(trading_system, args)
        elif args.mode == 'daemon':
            run_daemon_mode(trading_system, args)
        else:
            logger.error(f"알 수 없는 모드: {args.mode}")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        logger.info("===== 자동 주식 거래 시스템 종료 =====")

if __name__ == "__main__":
    main()
