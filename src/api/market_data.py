import requests
import pandas as pd
import logging
import yaml
from .auth import KoreaInvestmentAuth

logger = logging.getLogger(__name__)

class MarketData:
    def __init__(self, auth, config_path="config/api_config.yaml"):
        """시장 데이터 조회 클래스 초기화
        
        Args:
            auth (KoreaInvestmentAuth): 인증 객체
            config_path (str): 설정 파일 경로
        """
        self.auth = auth
        
        # 설정 파일 로드
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)['api']
        
        self.base_url = self.config['base_url']
    
    def get_stock_current_price(self, stock_code):
        """특정 종목의 현재가 조회
        
        Args:
            stock_code (str): 종목 코드
            
        Returns:
            dict: 현재가 정보
        """
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        
        # 요청 파라미터
        params = {
            "fid_cond_mrkt_div_code": "J",  # 시장 구분 (J: 주식)
            "fid_input_iscd": stock_code,   # 종목 코드
        }
        
        # 헤더 설정
        headers = self.auth.get_auth_headers()
        headers["tr_id"] = "FHKST01010100"  # 현재가 조회 거래 ID
        
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            # 응답 데이터 확인
            if data.get('rt_cd') != '0':
                logger.error(f"API Error: {data.get('msg_cd')} - {data.get('msg1')}")
                return None
            
            # 결과 반환
            return data.get('output')
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting current price: {str(e)}")
            if response:
                logger.error(f"Response: {response.text}")
            raise
    
    def get_stock_daily_price(self, stock_code, start_date=None, end_date=None, period=None):
        """특정 종목의 일별 가격 조회
        
        Args:
            stock_code (str): 종목 코드
            start_date (str, optional): 시작일(YYYYMMDD)
            end_date (str, optional): 종료일(YYYYMMDD)
            period (int, optional): 기간(일)
            
        Returns:
            pd.DataFrame: 일별 가격 데이터프레임
        """
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
        
        # 요청 파라미터
        params = {
            "fid_cond_mrkt_div_code": "J",  # 시장 구분 (J: 주식)
            "fid_input_iscd": stock_code,   # 종목 코드
            "fid_period_div_code": "D",     # 기간 분류 (D: 일)
            "fid_org_adj_prc": "1",         # 수정주가 여부 (1: 수정주가)
        }
        
        # 옵션 파라미터 설정
        if period:
            params["fid_input_date_1"] = str(period)
        if start_date:
            params["fid_input_date_1"] = start_date
        if end_date:
            params["fid_input_date_2"] = end_date
        
        # 헤더 설정
        headers = self.auth.get_auth_headers()
        headers["tr_id"] = "FHKST01010400"  # 일별 주가 조회 거래 ID
        
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            # 응답 데이터 확인
            if data.get('rt_cd') != '0':
                logger.error(f"API Error: {data.get('msg_cd')} - {data.get('msg1')}")
                return None
            
            # 결과를 데이터프레임으로 변환
            daily_prices = data.get('output')
            if not daily_prices:
                return pd.DataFrame()
            
            df = pd.DataFrame(daily_prices)
            
            # 필요한 컬럼 타입 변환
            numeric_cols = ['stck_oprc', 'stck_hgpr', 'stck_lwpr', 'stck_clpr', 'acml_vol', 'acml_tr_pbmn']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
            
            # 날짜 컬럼 변환
            if 'stck_bsop_date' in df.columns:
                df['stck_bsop_date'] = pd.to_datetime(df['stck_bsop_date'])
            
            return df
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting daily price: {str(e)}")
            if response:
                logger.error(f"Response: {response.text}")
            raise
    
    def get_account_balance(self, account_no=None):
        """계좌 잔고 조회
        
        Args:
            account_no (str, optional): 계좌번호. 미지정시 설정 파일 사용
            
        Returns:
            dict: 계좌 잔고 정보
        """
        if not account_no:
            account_no = self.config['account_no']
        
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
        
        # 요청 파라미터
        params = {
            "CANO": account_no[:8],              # 계좌번호 앞 8자리
            "ACNT_PRDT_CD": account_no[8:],      # 계좌상품코드
            "AFHR_FLPR_YN": "N",                 # 시간외단일가여부
            "OFL_YN": "",                        # 오프라인여부
            "INQR_DVSN": "02",                   # 조회구분
            "UNPR_DVSN": "01",                   # 단가구분
            "FUND_STTL_ICLD_YN": "N",            # 펀드결제분포함여부
            "FNCG_AMT_AUTO_RDPT_YN": "N",        # 융자금액자동상환여부
            "PRCS_DVSN": "01",                   # 처리구분
            "CTX_AREA_FK100": "",                # 연속조회검색조건
            "CTX_AREA_NK100": ""                 # 연속조회키
        }
        
        # 헤더 설정
        headers = self.auth.get_auth_headers()
        headers["tr_id"] = "TTTC8434R"  # 잔고 조회 거래 ID (실전 계좌)
        # 모의투자의 경우 "VTTC8434R" 사용
        
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            # 응답 데이터 확인
            if data.get('rt_cd') != '0':
                logger.error(f"API Error: {data.get('msg_cd')} - {data.get('msg1')}")
                return None
            
            # 결과 반환
            return {
                'account_summary': data.get('output1', []),
                'stocks': data.get('output2', [])
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting account balance: {str(e)}")
            if response:
                logger.error(f"Response: {response.text}")
            raise