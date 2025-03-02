import requests
import logging
import yaml
from .auth import KoreaInvestmentAuth

logger = logging.getLogger(__name__)

class OrderAPI:
    def __init__(self, auth, config_path="config/api_config.yaml"):
        """주문 실행 클래스 초기화
        
        Args:
            auth (KoreaInvestmentAuth): 인증 객체
            config_path (str): 설정 파일 경로
        """
        self.auth = auth
        
        # 설정 파일 로드
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)['api']
        
        self.base_url = self.config['base_url']
        self.account_no = self.config.get('account_no', '')
    
    def place_order(self, stock_code, order_type, quantity, price=0, account_no=None, order_division="00"):
        """주식 주문 실행
        
        Args:
            stock_code (str): 종목 코드
            order_type (str): 주문 유형 (01: 매도, 02: 매수)
            quantity (int): 주문 수량
            price (int, optional): 주문 가격. 시장가의 경우 0
            account_no (str, optional): 계좌번호. 미지정시 설정 파일 사용
            order_division (str, optional): 주문구분 (00: 지정가, 01: 시장가)
            
        Returns:
            dict: 주문 결과
        """
        if not account_no:
            account_no = self.account_no
        
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
        
        # 요청 데이터
        body = {
            "CANO": account_no[:8],              # 계좌번호 앞 8자리
            "ACNT_PRDT_CD": account_no[8:],      # 계좌상품코드
            "PDNO": stock_code,                  # 종목코드
            "ORD_DVSN": order_division,          # 주문구분 (00: 지정가, 01: 시장가)
            "ORD_QTY": str(quantity),            # 주문수량
            "ORD_UNPR": str(price),              # 주문단가
            "CTAC_TLNO": "",                     # 연락전화번호
            "SLL_BUY_DVSN_CD": order_type,       # 매도매수구분 (01: 매도, 02: 매수)
            "YNGCLNG_PBLC_OTHR_ISNC_YN_P": ""    # 영번호공개여부
        }
        
        # 헤더 설정 (해시키 포함)
        headers = self.auth.get_auth_headers(include_hashkey=True, body=body)
        
        # 실전계좌 거래ID 설정
        if order_type == "01":  # 매도
            headers["tr_id"] = "TTTC0801U"
        else:  # 매수
            headers["tr_id"] = "TTTC0802U"
        
        # 모의투자의 경우 각각 "VTTC0801U", "VTTC0802U" 사용
        
        try:
            response = requests.post(url, headers=headers, json=body)
            response.raise_for_status()
            
            data = response.json()
            
            # 응답 데이터 확인
            if data.get('rt_cd') != '0':
                logger.error(f"Order Error: {data.get('msg_cd')} - {data.get('msg1')}")
                return None
            
            # 결과 반환
            logger.info(f"Order placed successfully: {data}")
            return data.get('output')
        except requests.exceptions.RequestException as e:
            logger.error(f"Error placing order: {str(e)}")
            if response:
                logger.error(f"Response: {response.text}")
            raise
    
    def cancel_order(self, original_order_no, stock_code, quantity, account_no=None):
        """주문 취소
        
        Args:
            original_order_no (str): 원주문번호
            stock_code (str): 종목 코드
            quantity (int): 취소 수량
            account_no (str, optional): 계좌번호. 미지정시 설정 파일 사용
            
        Returns:
            dict: 취소 결과
        """
        if not account_no:
            account_no = self.account_no
        
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
        
        # 요청 데이터
        body = {
            "CANO": account_no[:8],              # 계좌번호 앞 8자리
            "ACNT_PRDT_CD": account_no[8:],      # 계좌상품코드
            "KRX_FWDG_ORD_ORGNO": "",            # KRX 주문조직번호
            "ORGN_ODNO": original_order_no,      # 원주문번호
            "ORD_DVSN": "00",                    # 주문구분 (취소는 지정가만)
            "RVSE_CNCL_DVSN_CD": "02",           # 정정취소구분코드 (02: 취소)
            "PDNO": stock_code,                  # 종목코드
            "ORD_QTY": str(quantity),            # 취소수량
            "ORD_UNPR": "0",                     # 주문단가 (취소는 0)
            "CTAC_TLNO": "",                     # 연락전화번호
            "RSVN_ORD_YN": "N"                   # 예약주문여부
        }
        
        # 헤더 설정 (해시키 포함)
        headers = self.auth.get_auth_headers(include_hashkey=True, body=body)
        
        # 실전계좌 거래ID 설정 (취소)
        headers["tr_id"] = "TTTC0803U"
        # 모의투자의 경우 "VTTC0803U" 사용
        
        try:
            response = requests.post(url, headers=headers, json=body)
            response.raise_for_status()
            
            data = response.json()
            
            # 응답 데이터 확인
            if data.get('rt_cd') != '0':
                logger.error(f"Cancel Error: {data.get('msg_cd')} - {data.get('msg1')}")
                return None
            
            # 결과 반환
            logger.info(f"Order cancelled successfully: {data}")
            return data.get('output')
        except requests.exceptions.RequestException as e:
            logger.error(f"Error cancelling order: {str(e)}")
            if response:
                logger.error(f"Response: {response.text}")
            raise
    
    def get_order_status(self, order_no, stock_code=None, account_no=None):
        """주문 상태 조회
        
        Args:
            order_no (str): 주문번호
            stock_code (str, optional): 종목 코드
            account_no (str, optional): 계좌번호. 미지정시 설정 파일 사용
            
        Returns:
            dict: 주문 상태 정보
        """
        if not account_no:
            account_no = self.account_no
        
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-order"
        
        # 요청 파라미터
        params = {
            "CANO": account_no[:8],          # 계좌번호 앞 8자리
            "ACNT_PRDT_CD": account_no[8:],  # 계좌상품코드
            "INQR_DVSN_1": "0",              # 조회구분1 (0: 전체)
            "INQR_DVSN_2": "0",              # 조회구분2 
            "CTX_AREA_FK100": "",            # 연속조회검색조건
            "CTX_AREA_NK100": "",            # 연속조회키
            "ODNO": order_no,                # 주문번호
            "PDNO": stock_code or "",        # 종목번호 (선택)
            "SORT_SQNO": "1",                # 정렬순번
            "ORD_GNO_BRNO": "",              # 주문채번지점번호
            "ODNO_TYPE": "1"                 # 주문번호 구분
        }
        
        # 헤더 설정
        headers = self.auth.get_auth_headers()
        headers["tr_id"] = "TTTC8001R"  # 주문 조회 거래 ID (실전 계좌)
        # 모의투자의 경우 "VTTC8001R" 사용
        
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
            logger.error(f"Error getting order status: {str(e)}")
            if response:
                logger.error(f"Response: {response.text}")
            raise