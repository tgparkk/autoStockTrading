import os
import yaml
import json
import logging
import re

logger = logging.getLogger(__name__)

class ConfigManager:
    """설정 관리 클래스
    
    설정 파일을 로드하고 관리하는 클래스입니다.
    API 설정, 거래 설정, 대상 종목 등의 설정을 중앙에서 관리합니다.
    """
    
    def __init__(self, config_dir="config"):
        """설정 관리자 초기화
        
        Args:
            config_dir (str): 설정 디렉토리 경로
        """
        self.config_dir = os.path.abspath(config_dir)
        self.api_config = None
        self.trading_config = None
        self.target_stocks = []
        
        # 설정 파일 로드
        self.load_config()
    
    def load_config(self):
        """모든 설정 파일 로드"""
        self.api_config = self.load_api_config()
        self.trading_config = self.load_trading_config()
        self.target_stocks = self.load_target_stocks()
        
        logger.info("모든 설정 파일이 로드되었습니다.")
    
    def load_api_config(self):
        """API 설정 로드"""
        api_config_path = os.path.join(self.config_dir, "api_config.yaml")
        if os.path.exists(api_config_path):
            try:
                with open(api_config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                logger.info("API 설정 파일이 로드되었습니다.")
                return config
            except Exception as e:
                logger.error(f"API 설정 로드 중 오류: {str(e)}")
        else:
            logger.warning(f"API 설정 파일이 존재하지 않습니다: {api_config_path}")
        return {}
    
    def load_trading_config(self):
        """거래 설정 로드"""
        trading_config_path = os.path.join(self.config_dir, "trading_config.yaml")
        if os.path.exists(trading_config_path):
            try:
                with open(trading_config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                logger.info("거래 설정 파일이 로드되었습니다.")
                return config
            except Exception as e:
                logger.error(f"거래 설정 로드 중 오류: {str(e)}")
        else:
            logger.warning(f"거래 설정 파일이 존재하지 않습니다: {trading_config_path}")
        return {}
    
    def load_target_stocks(self):
        """대상 종목 로드"""
        target_stocks_path = os.path.join(self.config_dir, "target_stocks.txt")
        if os.path.exists(target_stocks_path):
            try:
                with open(target_stocks_path, 'r', encoding='utf-8') as f:
                    stocks = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
                logger.info(f"대상 종목 파일이 로드되었습니다: {len(stocks)}개 종목")
                return stocks
            except Exception as e:
                logger.error(f"대상 종목 로드 중 오류: {str(e)}")
        else:
            logger.warning(f"대상 종목 파일이 존재하지 않습니다: {target_stocks_path}")
        return []
    
    def save_trading_config(self, config):
        """거래 설정 저장 (주석 유지)
        
        Args:
            config (dict): 저장할 설정
            
        Returns:
            bool: 저장 성공 여부
        """
        trading_config_path = os.path.join(self.config_dir, "trading_config.yaml")
        try:
            # 주석이 유지되는 방식으로 저장하기 위해 라인 단위로 처리
            if os.path.exists(trading_config_path):
                with open(trading_config_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # 주석이 있는 설정 파일 파싱
                updated_lines = []
                in_strategy_section = False
                strategy_indent = ""
                
                for line in lines:
                    # 빈 줄이나 주석은 그대로 유지
                    if line.strip() == "" or line.strip().startswith('#'):
                        updated_lines.append(line)
                        continue
                    
                    # strategy 섹션 시작 감지
                    if line.strip() == "strategy:" or line.strip().startswith("strategy:"):
                        in_strategy_section = True
                        strategy_indent = " " * (line.find("strategy:"))
                        updated_lines.append(line)
                        continue
                    
                    # 다른 최상위 섹션 시작 감지
                    if ":" in line and not line.startswith(" ") and not line.startswith("\t"):
                        if line.split(":")[0].strip() != "strategy":
                            in_strategy_section = False
                    
                    # strategy 섹션 내부의 값들 업데이트
                    if in_strategy_section and ":" in line:
                        key = line.split(":")[0].strip()
                        if key in config.get('strategy', {}):
                            # 주석 유지
                            comment = ""
                            if "#" in line:
                                comment = " " + line.split("#", 1)[1].rstrip("\n")
                            
                            # 들여쓰기 유지
                            indent = ""
                            for char in line:
                                if char == " " or char == "\t":
                                    indent += char
                                else:
                                    break
                            
                            # 새 라인 구성
                            updated_lines.append(f"{indent}{key}: {config['strategy'][key]}{comment}\n")
                            continue
                    
                    # interval 값 업데이트
                    if not in_strategy_section and ":" in line:
                        key = line.split(":")[0].strip()
                        if key == "interval" and "interval" in config:
                            # 주석 유지
                            comment = ""
                            if "#" in line:
                                comment = " " + line.split("#", 1)[1].rstrip("\n")
                            
                            # 들여쓰기 유지
                            indent = ""
                            for char in line:
                                if char == " " or char == "\t":
                                    indent += char
                                else:
                                    break
                            
                            updated_lines.append(f"{indent}interval: {config['interval']}{comment}\n")
                            continue
                    
                    # 그 외 라인은 그대로 유지
                    updated_lines.append(line)
                
                # 파일에 다시 쓰기
                with open(trading_config_path, 'w', encoding='utf-8') as f:
                    f.writelines(updated_lines)
            else:
                # 파일이 없으면 새로 생성
                with open(trading_config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False)
            
            # 내부 설정 업데이트
            if 'strategy' in config:
                if not self.trading_config:
                    self.trading_config = {}
                self.trading_config['strategy'] = config['strategy']
            if 'interval' in config:
                if not self.trading_config:
                    self.trading_config = {}
                self.trading_config['interval'] = config['interval']
            
            logger.info("거래 설정이 저장되었습니다.")
            return True
        except Exception as e:
            logger.error(f"거래 설정 저장 중 오류: {str(e)}")
            return False
    
    def save_target_stocks(self, stocks):
        """대상 종목 저장
        
        Args:
            stocks (list): 저장할 종목 코드 리스트
            
        Returns:
            bool: 저장 성공 여부
        """
        target_stocks_path = os.path.join(self.config_dir, "target_stocks.txt")
        try:
            # 디렉토리 확인
            os.makedirs(os.path.dirname(target_stocks_path), exist_ok=True)
            
            with open(target_stocks_path, 'w', encoding='utf-8') as f:
                for stock in stocks:
                    f.write(f"{stock}\n")
            
            # 내부 리스트 업데이트
            self.target_stocks = stocks
            logger.info(f"대상 종목 {len(stocks)}개가 저장되었습니다.")
            return True
        except Exception as e:
            logger.error(f"대상 종목 저장 중 오류: {str(e)}")
            return False
    
    def get_api_config(self):
        """API 설정 반환"""
        return self.api_config or {}
    
    def get_strategy_config(self):
        """전략 설정 반환"""
        if self.trading_config and 'strategy' in self.trading_config:
            return self.trading_config['strategy']
        return {}
    
    def get_interval(self):
        """작업 실행 간격 반환"""
        if self.trading_config and 'interval' in self.trading_config:
            return self.trading_config['interval']
        return 2  # 기본값 2분
    
    def update_strategy_config(self, strategy_config):
        """전략 설정 업데이트
        
        Args:
            strategy_config (dict): 업데이트할 전략 설정
            
        Returns:
            bool: 업데이트 성공 여부
        """
        config = {'strategy': strategy_config}
        if self.trading_config and 'interval' in self.trading_config:
            config['interval'] = self.trading_config['interval']
        
        return self.save_trading_config(config)
    
    def update_interval(self, interval):
        """작업 실행 간격 업데이트
        
        Args:
            interval (int): 업데이트할 간격(분)
            
        Returns:
            bool: 업데이트 성공 여부
        """
        config = {'interval': interval}
        if self.trading_config and 'strategy' in self.trading_config:
            config['strategy'] = self.trading_config['strategy']
        
        return self.save_trading_config(config)