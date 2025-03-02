import requests
import yaml
import json
import time
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class KoreaInvestmentAuth:
    def __init__(self, config_path="config/api_config.yaml"):
        """한국투자증권 API 인증 클래스 초기화
        
        Args:
            config_path (str): API 설정 파일 경로
        """
        # 설정 파일 로드
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)['api']
        
        self.base_url = self.config['base_url']
        self.app_key = self.config['app_key']
        self.app_secret = self.config['app_secret']
        
        # 토큰 파일 경로
        token_dir = os.path.dirname(os.path.abspath(config_path))
        self.token_file = os.path.join(token_dir, "token_info.json")
        
        # 토큰 정보 초기화
        self.access_token = None
        self.token_issued_at = None
        self.token_expired_at = None
        
        # 저장된 토큰 정보 로드
        self._load_token_info()
    
    def _load_token_info(self):
        """저장된 토큰 정보 로드"""
        if os.path.exists(self.token_file):
            try:
                with open(self.token_file, 'r', encoding='utf-8') as f:
                    token_info = json.load(f)
                    
                    self.access_token = token_info.get('access_token')
                    self.token_issued_at = token_info.get('issued_at')
                    self.token_expired_at = token_info.get('expired_at')
                    
                    logger.info("토큰 정보를 파일에서 로드했습니다.")
                    
                    # 토큰 유효성 검증
                    current_time = time.time()
                    if self.token_expired_at and current_time >= self.token_expired_at:
                        logger.info("로드한 토큰이 만료되었습니다.")
                        self.access_token = None
                        self.token_expired_at = None
            except Exception as e:
                logger.error(f"토큰 정보 로드 중 오류 발생: {str(e)}")
                # 오류 발생 시 토큰 정보 초기화
                self.access_token = None
                self.token_issued_at = None
                self.token_expired_at = None

    def _save_token_info(self):
        """토큰 정보 저장"""
        token_info = {
            'access_token': self.access_token,
            'issued_at': self.token_issued_at,
            'expired_at': self.token_expired_at
        }
        
        try:
            # 디렉토리 확인
            token_dir = os.path.dirname(self.token_file)
            os.makedirs(token_dir, exist_ok=True)
            
            with open(self.token_file, 'w', encoding='utf-8') as f:
                json.dump(token_info, f, indent=2)
            logger.info(f"토큰 정보를 파일에 저장했습니다: {self.token_file}")
        except Exception as e:
            logger.error(f"토큰 정보 저장 중 오류 발생: {str(e)}")

    def get_access_token(self, force_new=False):
        """액세스 토큰 발급 또는 캐시된 토큰 반환
        
        Args:
            force_new (bool): 강제로 새 토큰 발급 여부
            
        Returns:
            str: 액세스 토큰
        """
        current_time = time.time()
        
        # 토큰 유효성 확인
        token_is_valid = (
            self.access_token is not None and
            not force_new and
            self.token_expired_at is not None and
            current_time < self.token_expired_at
        )
        
        if token_is_valid:
            logger.debug("캐시된 토큰을 사용합니다.")
            return self.access_token
        
        # 하루에 한 번만 토큰 발급 (강제 갱신 제외)
        if self.token_issued_at and not force_new:
            issued_date = datetime.fromtimestamp(self.token_issued_at).date()
            today = datetime.now().date()
            
            if issued_date == today:
                logger.warning("오늘 이미 토큰이 발급되었습니다. 기존 토큰을 사용합니다.")
                if self.access_token:
                    return self.access_token
                else:
                    logger.warning("기존 토큰이 유효하지 않습니다. 새 토큰을 발급합니다.")
        
        # 토큰 발급 API 엔드포인트
        url = f"{self.base_url}/oauth2/tokenP"
        
        # 요청 헤더와 데이터
        headers = {
            "content-type": "application/json"
        }
        
        data = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }
        
        try:
            # API 호출
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()  # 오류가 있는 경우 예외 발생
            
            token_data = response.json()
            self.access_token = token_data.get('access_token')
            
            # 토큰 만료 시간 설정
            expires_in = token_data.get('expires_in', 86400)  # 기본값 24시간
            self.token_issued_at = current_time
            self.token_expired_at = current_time + expires_in - 300  # 5분 여유
            
            # 토큰 정보 저장
            self._save_token_info()
            
            logger.info(f"새 액세스 토큰이 발급되었습니다. 만료 시간: {expires_in}초")
            return self.access_token
            
        except requests.exceptions.RequestException as e:
            logger.error(f"토큰 발급 중 오류 발생: {str(e)}")
            if response:
                logger.error(f"응답: {response.text}")
            raise
    
    def get_hashkey(self, data):
        """데이터로부터 해시키 생성
        
        Args:
            data (dict): 해시키를 생성할 데이터
            
        Returns:
            str: 생성된 해시키
        """
        url = f"{self.base_url}/uapi/hashkey"
        
        headers = {
            "content-type": "application/json",
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }
        
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            
            hashkey = response.json()['HASH']
            return hashkey
        except requests.exceptions.RequestException as e:
            logger.error(f"해시키 생성 중 오류 발생: {str(e)}")
            if response:
                logger.error(f"응답: {response.text}")
            raise
    
    def get_auth_headers(self, include_hashkey=False, body=None):
        """인증 헤더 생성
        
        Args:
            include_hashkey (bool): 해시키 포함 여부
            body (dict): 해시키 생성에 사용할 요청 바디
            
        Returns:
            dict: 인증 헤더
        """
        token = self.get_access_token()
        
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "",  # 필요에 따라 설정
        }
        
        if include_hashkey and body:
            headers["hashkey"] = self.get_hashkey(body)
        
        return headers