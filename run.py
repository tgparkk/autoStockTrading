#!/usr/bin/env python
"""
자동 주식 거래 시스템 실행 스크립트
여러 실행 모드를 지원합니다:
- cli: 명령행 인터페이스 (기본)
- web: 웹 인터페이스
- daemon: 데몬 모드
"""

import argparse
import os
import sys
import logging
from datetime import datetime

# 모듈 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """로깅 설정"""
    log_dir = os.path.abspath("logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'run_{datetime.now().strftime("%Y%m%d")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def run_cli_mode(args):
    """CLI 모드 실행"""
    from main import main as cli_main
    cli_main()

def run_web_mode(args):
    """웹 인터페이스 모드 실행"""
    try:
        from src.web.app import start_web_server
        start_web_server(host=args.host, port=args.port)
    except ImportError:
        logging.error("웹 인터페이스 모듈을 찾을 수 없습니다.")
        sys.exit(1)

def run_daemon_mode(args):
    """데몬 모드 실행"""
    import daemon
    import lockfile
    
    log_dir = os.path.abspath("logs")
    os.makedirs(log_dir, exist_ok=True)
    
    pid_file = os.path.join(log_dir, 'trading_daemon.pid')
    log_file = os.path.join(log_dir, 'trading_daemon.log')
    
    # 로깅 설정
    daemon_logger = logging.getLogger()
    daemon_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    daemon_logger.addHandler(handler)
    
    # 데몬 컨텍스트 설정
    context = daemon.DaemonContext(
        working_directory=os.path.abspath('.'),
        umask=0o002,
        pidfile=lockfile.FileLock(pid_file)
    )
    
    # 표준 출력/에러를 로그 파일로 리다이렉트
    context.stdout = open(log_file, 'a+')
    context.stderr = open(log_file, 'a+')
    
    with context:
        logging.info("데몬 모드로 거래 시스템 시작")
        from main import main as daemon_main
        daemon_main()

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='자동 주식 거래 시스템')
    subparsers = parser.add_subparsers(dest='mode', help='실행 모드')
    
    # CLI 모드 인자
    cli_parser = subparsers.add_parser('cli', help='명령행 인터페이스')
    cli_parser.add_argument('--config', default='config/api_config.yaml', help='API 설정 파일 경로')
    cli_parser.add_argument('--strategy', default='config/trading_config.yaml', help='전략 설정 파일 경로')
    cli_parser.add_argument('--strategy-type', default='basic', choices=['basic', 'day_trading', 'high_frequency', 'ml_high_frequency'], help='전략 유형')
    cli_parser.add_argument('--once', action='store_true', help='한 번만 실행')
    
    # 웹 모드 인자
    web_parser = subparsers.add_parser('web', help='웹 인터페이스')
    web_parser.add_argument('--host', default='127.0.0.1', help='호스트 주소')
    web_parser.add_argument('--port', type=int, default=5000, help='포트 번호')
    
    # 데몬 모드 인자
    daemon_parser = subparsers.add_parser('daemon', help='데몬 모드')
    daemon_parser.add_argument('--config', default='config/api_config.yaml', help='API 설정 파일 경로')
    daemon_parser.add_argument('--strategy', default='config/trading_config.yaml', help='전략 설정 파일 경로')
    daemon_parser.add_argument('--strategy-type', default='basic', help='전략 유형')
    
    args = parser.parse_args()
    
    # 기본 모드는 CLI
    if not args.mode:
        args.mode = 'cli'
    
    # 로깅 설정
    setup_logging()
    
    logging.info(f"실행 모드: {args.mode}")
    
    # 모드에 따른 실행 함수 호출
    if args.mode == 'cli':
        run_cli_mode(args)
    elif args.mode == 'web':
        run_web_mode(args)
    elif args.mode == 'daemon':
        run_daemon_mode(args)
    else:
        logging.error(f"알 수 없는 모드: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
