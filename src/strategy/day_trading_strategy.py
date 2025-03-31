from datetime import datetime
import logging
from .basic_strategy import BasicStrategy

logger = logging.getLogger(__name__)

class DayTradingStrategy(BasicStrategy):
    """일일 트레이딩(Day Trading) 전략 클래스"""
    
    def __init__(self, market_data, order_api, config=None):
        super().__init__(market_data, order_api, config)
        
        # 일일 트레이딩 추가 설정값
        self.day_trading_config = {
            'entry_time_start': '09:30',  # 진입 시작 시간 (9시 30분)
            'entry_time_end': '14:00',    # 진입 종료 시간 (14시)
            'exit_time': '15:00',         # 청산 시간 (15시)
            'min_volume_increase': 1.5,   # 최소 거래량 증가율
            'profit_target': 0.01,        # 1% 목표 수익률 
            'stop_loss': 0.005,           # 0.5% 손절
            'max_positions': 5,           # 최대 포지션 개수
            'check_interval': 1,          # 5분 간격으로 신호 체크
            'max_price': 50000,           # 최대 주가 (5만원)
        }
        
        # 기본 설정 값 덮어쓰기
        self.config['stop_loss'] = self.day_trading_config['stop_loss']
        self.config['take_profit'] = self.day_trading_config['profit_target']
        self.config['max_position'] = self.day_trading_config['max_positions']
        
        # 당일 거래 기록
        self.today_trades = []
    
    def is_entry_time(self):
        """진입 가능 시간인지 확인"""
        now = datetime.now()
        current_time = now.strftime('%H:%M')
        return self.day_trading_config['entry_time_start'] <= current_time <= self.day_trading_config['entry_time_end']
    
    def is_exit_time(self):
        """청산 시간인지 확인"""
        now = datetime.now()
        current_time = now.strftime('%H:%M')
        return current_time >= self.day_trading_config['exit_time']
    
    def analyze_intraday_stock(self, stock_code):
        """일중 데이터 기반 종목 분석"""
        try:
            # 1. 현재가 데이터 조회
            current_data = self.market_data.get_stock_current_price(stock_code)
            logger.info(f"[매매 분석] {stock_code} 현재 데이터: {current_data}")
            if not current_data:
                return {'signal': 'neutral', 'reason': '데이터 없음'}
            
            # 2. 가격 변동성 체크
            price = float(current_data.get('stck_prpr', 0))
            prev_price = float(current_data.get('stck_sdpr', 0))  # 전일 종가
            
            if prev_price <= 0:
                return {'signal': 'neutral', 'reason': '가격 데이터 오류'}
            
            # 가격 변동률
            price_change = (price - prev_price) / prev_price
            
            # 3. 거래량 확인
            volume = int(current_data.get('acml_vol', 0))
            avg_volume = int(current_data.get('avg_vol', 0))  # 평균 거래량
            
            volume_increase = 1.0

            trading_hours_passed = 0.3  # 예: 장 시작 후 30% 시간 경과
            volume = int(current_data.get('acml_vol', 0))
            prev_day_volume = int(current_data.get('prdy_vol', 0))  # 전일 거래량 필드 (실제 API에서 확인 필요)


            if avg_volume > 0:
                volume_increase = volume / (prev_day_volume * trading_hours_passed)

            logger.info(f"[매매 분석] {stock_code} 가격: {price}원, 전일대비: {price_change:.1%}, 거래량비율: {volume_increase:.1f}배")
        
            
            # 4. 매매 신호 생성
            signal = 'neutral'
            reasons = []
            
            # 매수 조건 개선 예시
            # 1. 상승 추세 또는 하락 반전 조건
            if (price_change > 0.003 and volume_increase > 1.2):  # 완화된 상승 조건
                signal = 'buy'
                reasons.append(f'상승 추세 감지 ({price_change:.1%})')
                reasons.append(f'거래량 증가 (평균 대비 {volume_increase:.1f}배)')
            
            # 하락 추세 = 매도 신호
            elif price_change < -0.01:
                signal = 'sell'
                reasons.append(f'하락 추세 감지 ({price_change:.1%})')
            
            # 청산 시간 확인
            if self.is_exit_time():
                signal = 'sell'
                reasons.append('장 마감 전 청산 시간')
            
            return {
                'signal': signal, 
                'reasons': reasons,
                'price': price,
                'price_change': price_change,
                'volume_increase': volume_increase
            }
        
        except Exception as e:
            logger.error(f"일중 종목 분석 오류 ({stock_code}): {str(e)}")
            return {'signal': 'error', 'reason': str(e)}
        

    def _get_all_market_stocks(self):
        """시장의 모든 종목 코드 가져오기"""
        logger.info("시장 종목 목록 조회 중...")
        
        # 기본 종목 리스트 (예시)
        default_stocks = [
            # 코스피 대표 저가 종목들
            '005930', '000660', '035420', '035720', '051910',  # 삼성전자, SK하이닉스, NAVER, 카카오, LG화학
            '005380', '012330', '068270', '207940', '006400',  # 현대차, 현대모비스, 셀트리온, 삼성바이오로직스, 삼성SDI
            '018260', '003550', '036570', '028260', '033780',  # 삼성SDS, LG, 엔씨소프트, 삼성물산, KT&G
            '015760', '017670', '096770', '012330', '039490',  # 한국전력, SK텔레콤, SK이노베이션, 현대모비스, 키움증권
            '016360', '009540', '010130', '011200', '011170',  # 삼성증권, 한국조선해양, 고려아연, HMM, 롯데케미칼
            '055550', '086280', '024110', '003490', '032830',  # 신한지주, 현대글로비스, 기업은행, 대한항공, 삼성생명
        ]
        
        # 한국투자증권 API를 사용하여 전체 종목 목록을 가져올 수도 있습니다.
        # 그러나 API 호출 제한이 있을 수 있으므로, 기본 목록을 사용하는 것이 좋을 수도 있습니다.
        try:
            import requests
            
            # API URL과 헤더 설정
            url = f"{self.market_data.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
            
            # 헤더 설정
            headers = self.market_data.auth.get_auth_headers()
            
            # 주요 거래소 종목 조회 (코스피, 코스닥)
            all_stocks = []
            
            for market_code in ["J", "Q"]:  # J: 코스피, Q: 코스닥
                try:
                    # 거래소별 상위 종목 조회 (API 명세에 따라 다를 수 있음)
                    params = {
                        "fid_cond_mrkt_div_code": market_code,
                        "fid_input_iscd": "",  # 빈 값으로 설정하면 전체 종목 조회 가능한 경우도 있음
                    }
                    
                    response = requests.get(url, params=params, headers=headers)
                    if response.status_code == 200:
                        data = response.json()
                        if 'output' in data:
                            # 응답에서 종목 코드 추출
                            for item in data['output']:
                                if 'prdt_code' in item:
                                    all_stocks.append(item['prdt_code'])
                            
                            logger.info(f"{market_code} 시장에서 {len(all_stocks)}개 종목 조회됨")
                except Exception as e:
                    logger.warning(f"{market_code} 시장 종목 조회 중 오류: {str(e)}")
            
            # 종목이 조회되면 해당 목록 반환, 그렇지 않으면 기본 목록 사용
            if all_stocks:
                logger.info(f"API에서 총 {len(all_stocks)}개 종목 조회됨")
                return all_stocks
            else:
                logger.warning("API에서 종목을 가져오지 못했습니다. 기본 목록 사용.")
                return default_stocks
                
        except Exception as e:
            logger.error(f"시장 종목 목록 조회 중 오류: {str(e)}")
            # 오류 발생 시 기본 목록 반환
            logger.info(f"기본 종목 목록 {len(default_stocks)}개 반환")
            return default_stocks
        
        
    def weekly_update(self):
        """주간 업데이트 - 저가 종목 선정"""
        logger.info("일일 트레이딩 전략: 저가 종목 선정 시작")
        
        # 시장 전체 종목 목록 가져오기 (이 함수는 구현되어 있어야 함)
        all_stocks = self._get_all_market_stocks()
        
        # 저가 종목 필터링
        low_price_stocks = []
        
        for stock_code in all_stocks:
            try:
                # 현재가 조회
                current_data = self.market_data.get_stock_current_price(stock_code)
                if current_data and 'stck_prpr' in current_data:
                    price = float(current_data['stck_prpr'])
                    
                    # 5만원 이하인 종목만 선택
                    if price <= self.day_trading_config['max_price']:
                        # 종목 정보 저장
                        low_price_stocks.append({
                            'code': stock_code,
                            'price': price,
                            'name': current_data.get('prdt_name', '알 수 없음')
                        })
                        logger.info(f"저가 종목 선정: {stock_code} ({current_data.get('prdt_name', '알 수 없음')}) - {price}원")
            except Exception as e:
                logger.warning(f"종목 가격 확인 중 오류 ({stock_code}): {str(e)}")
        
        # 가격 순으로 정렬
        low_price_stocks.sort(key=lambda x: x['price'])
        
        # 최대 20개 종목만 선택 (선택할 종목 수는 필요에 따라 조정)
        selected_codes = [stock['code'] for stock in low_price_stocks[:20]]
        
        # 선정된 종목 저장
        self.selected_stocks = selected_codes
        
        logger.info(f"저가 종목 선정 완료: {len(selected_codes)}개 종목")
        
        return True
    
    def run(self, target_stocks):
        """전략 실행"""
        logger.info(f"일일 트레이딩 전략 실행 (대상: {len(target_stocks)}개 종목)")

        # 각 타겟 종목에 대해 로그 추가
        for stock_code in target_stocks:
            logger.info(f"분석 대상 종목: {stock_code}")
        
        results = {
            'buys': [],
            'sells': [],
            'errors': []
        }
        
        try:
            # 포지션 정보 업데이트
            self.update_positions()
            
            # 장 마감 전이면 모든 포지션 청산
            if self.is_exit_time():
                logger.info("장 마감 전 청산 시간")
                
                for stock_code in list(self.positions.keys()):
                    logger.info(f"장 마감 전 청산: {stock_code}")
                    result = self.execute_sell(stock_code)
                    
                    if result:
                        results['sells'].append({
                            'stock_code': stock_code,
                            'reason': '장 마감 전 청산',
                            'result': result
                        })
                    else:
                        results['errors'].append({
                            'stock_code': stock_code,
                            'action': 'sell',
                            'reason': '장 마감 전 청산',
                            'error': '주문 실패'
                        })
                
                return results
            
            # 진입 가능 시간이 아니면 매수 안 함
            if not self.is_entry_time():
                logger.info("진입 가능 시간이 아닙니다")
                
                # 기존 포지션 점검만 수행
                for stock_code in list(self.positions.keys()):
                    should_sell, reason = self.should_sell(stock_code)
                    
                    if should_sell:
                        logger.info(f"매도 신호: {stock_code} - {reason}")
                        result = self.execute_sell(stock_code)
                        
                        if result:
                            results['sells'].append({
                                'stock_code': stock_code,
                                'reason': reason,
                                'result': result
                            })
                        else:
                            results['errors'].append({
                                'stock_code': stock_code,
                                'action': 'sell',
                                'reason': reason,
                                'error': '주문 실패'
                            })
                
                return results
            
            # 매도 확인 (보유 종목)
            for stock_code in list(self.positions.keys()):
                analysis = self.analyze_intraday_stock(stock_code)
                
                if analysis['signal'] == 'sell':
                    logger.info(f"매도 신호: {stock_code} - {', '.join(analysis.get('reasons', []))}")
                    result = self.execute_sell(stock_code)
                    
                    if result:
                        results['sells'].append({
                            'stock_code': stock_code,
                            'reason': ', '.join(analysis.get('reasons', [])),
                            'result': result
                        })
                    else:
                        results['errors'].append({
                            'stock_code': stock_code,
                            'action': 'sell',
                            'reason': ', '.join(analysis.get('reasons', [])),
                            'error': '주문 실패'
                        })
            
            # 최대 포지션 수 확인
            if len(self.positions) >= self.day_trading_config['max_positions']:
                logger.info(f"최대 포지션 수 도달: {len(self.positions)}/{self.day_trading_config['max_positions']}")
                return results
            
            # 매수 확인 (대상 종목)
            for stock_code in target_stocks:
                logger.info(f"매수 판단 시작: {stock_code}")

                # 이미 보유중인 종목은 스킵
                if stock_code in self.positions:
                    logger.info(f"이미 보유 중인 종목 스킵: {stock_code}")
                    continue
                
                analysis = self.analyze_intraday_stock(stock_code)
                logger.info(f"분석 결과: {stock_code} - 신호: {analysis['signal']}")
                
                if analysis['signal'] == 'buy':
                    logger.info(f"매수 신호: {stock_code} - {', '.join(analysis.get('reasons', []))}")
                    result = self.execute_buy(stock_code)
                    
                    if result:
                        results['buys'].append({
                            'stock_code': stock_code,
                            'reason': ', '.join(analysis.get('reasons', [])),
                            'result': result
                        })
                        
                        # 최대 포지션 수 도달 시 매수 중단
                        if len(self.positions) >= self.day_trading_config['max_positions']:
                            logger.info(f"최대 포지션 수 도달: {len(self.positions)}/{self.day_trading_config['max_positions']}")
                            break
                    else:
                        results['errors'].append({
                            'stock_code': stock_code,
                            'action': 'buy',
                            'reason': ', '.join(analysis.get('reasons', [])),
                            'error': '주문 실패'
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"전략 실행 중 오류: {str(e)}")
            return results