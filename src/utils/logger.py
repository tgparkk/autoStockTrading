import logging
import os
from datetime import datetime

def setup_logger(log_file=None, log_level=logging.INFO):
    """로깅 설정 함수
    
    Args:
        log_file (str, optional): 로그 파일 경로
        log_level (int, optional): 로그 레벨
        
    Returns:
        logging.Logger: 설정된 로거 객체
    """
    # 루트 로거 설정
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 포맷 설정
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 콘솔 핸들러 추가
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 추가 (지정된 경우)
    if log_file:
        # 로그 파일명에 날짜가 없는 경우 날짜 추가
        if '%' not in log_file:  # strftime 포맷이 없으면
            from datetime import datetime
            log_dir = os.path.dirname(log_file)
            log_base = os.path.basename(log_file)
            log_name, log_ext = os.path.splitext(log_base)
            
            # 날짜 형식 추가
            current_date = datetime.now().strftime('%Y%m%d')
            new_log_file = os.path.join(log_dir, f"{log_name}_{current_date}{log_ext}")
            log_file = new_log_file
            
        # 디렉토리가 없는 경우 생성
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger