from abc import ABC, abstractmethod

class ApiClient(ABC):
    """API 클라이언트 기본 인터페이스
    
    이 추상 클래스는 외부 API와의 통신을 위한 기본 인터페이스를 정의합니다.
    모든 API 클라이언트 클래스는 이 인터페이스를 구현해야 합니다.
    """
    
    @abstractmethod
    def authenticate(self):
        """인증 수행
        
        API 서비스 인증을 수행하고 인증 토큰을 획득합니다.
        
        Returns:
            bool: 인증 성공 여부
        """
        pass
    
    @abstractmethod
    def get_headers(self):
        """요청 헤더 반환
        
        API 요청에 필요한 헤더를 생성합니다.
        
        Returns:
            dict: API 요청 헤더
        """
        pass
    
    @abstractmethod
    def call(self, endpoint, method="GET", params=None, data=None):
        """API 호출
        
        지정된 엔드포인트로 API를 호출합니다.
        
        Args:
            endpoint (str): API 엔드포인트
            method (str, optional): HTTP 메서드 (GET, POST 등). 기본값은 "GET".
            params (dict, optional): URL 파라미터
            data (dict, optional): 요청 바디 데이터
            
        Returns:
            dict: API 응답 데이터
        """
        pass