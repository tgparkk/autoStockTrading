import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ..utils.data_utils import calculate_moving_average, calculate_rsi, calculate_bollinger_bands

logger = logging.getLogger(__name__)

class BasicStrategy:
    """기본 매매 전략 클래스"""
    
    def __init__(self, market_data, order_api, config=None):
        """기본 매매 전략 초기화
        
        Args:
            market_data (MarketData): 시장 데이터 객체
            order_api (OrderAPI): 주문 API 객체
            config (dict, optional): 전략 설정
        """
        self.market_data = market_data
        self.order_api = order_api
        self.config = config or {}
        
        # 기본 설정값
        self.default_config = {
            'ma_short': 5,            # 단기 이동평균선 기간
            'ma_long': 20,            # 장기 이동평균선 기간
            'rsi_period': 14,         # RSI 계산 기간
            'rsi_oversold': 30,       # RSI 과매도 기준
            'rsi_overbought': 70,     # RSI 과매수 기준
            'bb_period': 20,          # 볼린저 밴드 기간
            'bb_std': 2,              # 볼린저 밴드 표준편차
            'stop_loss': 0.03,        # 손절 기준 (3%)
            'take_profit': 0.05,      # 익절 기준 (5%)
            'max_position': 5,        # 최대 포지션 개수
            'position_size': 0.2,     # 포지션 크기 (자본금 대비 %)
        }
        
        # 설정 병합
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        # 보유 종목 관리
        self.positions = {}  # {종목코드: {수량, 평균단가, 매수시간}}
        
        logger.info(f"Strategy initialized with config: {self.config}")
    
    def analyze_stock(self, stock_code, days=60):
        """종목 분석
        
        Args:
            stock_code (str): 종목 코드
            days (int, optional): 분석 기간(일)
            
        Returns:
            dict: 분석 결과
        """
        logger.info(f"Analyzing stock: {stock_code}")
        
        try:
            # 일별 주가 데이터 조회
            df = self.market_data.get_stock_daily_price(stock_code, period=days)
            
            if df.empty:
                logger.warning(f"No data found for stock: {stock_code}")
                return {'signal': 'neutral', 'reason': 'No data'}
            
            # 이동평균선 계산
            ma_short = self.config['ma_short']
            ma_long = self.config['ma_long']
            df = calculate_moving_average(df, windows=[ma_short, ma_long])
            
            # RSI 계산
            df = calculate_rsi(df, period=self.config['rsi_period'])
            
            # 볼린저 밴드 계산
            df = calculate_bollinger_bands(df, window=self.config['bb_period'], num_std=self.config['bb_std'])
            
            # 최신 데이터 추출
            current = df.iloc[-1]
            previous = df.iloc[-2] if len(df) > 1 else None
            
            # 분석 결과
            analysis = {}
            
            # 현재가
            current_price = current['stck_clpr']
            analysis['current_price'] = current_price
            
            # 이동평균선 분석
            analysis['ma_short'] = current[f'ma_{ma_short}']
            analysis['ma_long'] = current[f'ma_{ma_long}']
            
            # 골든 크로스 확인 (단기선이 장기선을 상향 돌파)
            if previous is not None:
                golden_cross = previous[f'ma_{ma_short}'] <= previous[f'ma_{ma_long}'] and \
                               current[f'ma_{ma_short}'] > current[f'ma_{ma_long}']
                analysis['golden_cross'] = golden_cross
                
                # 데드 크로스 확인 (단기선이 장기선을 하향 돌파)
                dead_cross = previous[f'ma_{ma_short}'] >= previous[f'ma_{ma_long}'] and \
                             current[f'ma_{ma_short}'] < current[f'ma_{ma_long}']
                analysis['dead_cross'] = dead_cross
            
            # RSI 분석
            analysis['rsi'] = current['rsi']
            analysis['oversold'] = current['rsi'] < self.config['rsi_oversold']
            analysis['overbought'] = current['rsi'] > self.config['rsi_overbought']
            
            # 볼린저 밴드 분석
            analysis['bb_upper'] = current['bb_upper']
            analysis['bb_lower'] = current['bb_lower']
            analysis['bb_ma'] = current['bb_ma']
            analysis['above_upper'] = current_price > current['bb_upper']
            analysis['below_lower'] = current_price < current['bb_lower']
            
            # 신호 결정
            signal = 'neutral'
            reasons = []
            
            # 매수 신호
            if analysis.get('golden_cross', False):
                signal = 'buy'
                reasons.append('골든 크로스 발생')
            
            if analysis['oversold']:
                signal = 'buy'
                reasons.append('RSI 과매도 상태')
            
            if analysis['below_lower']:
                signal = 'buy'
                reasons.append('볼린저 밴드 하단 돌파')
            
            # 매도 신호
            if analysis.get('dead_cross', False):
                signal = 'sell'
                reasons.append('데드 크로스 발생')
            
            if analysis['overbought']:
                signal = 'sell'
                reasons.append('RSI 과매수 상태')
            
            if analysis['above_upper']:
                signal = 'sell'
                reasons.append('볼린저 밴드 상단 돌파')
            
            # 최종 신호 및 이유
            analysis['signal'] = signal
            analysis['reasons'] = reasons
            
            logger.info(f"Analysis result for {stock_code}: {signal} ({', '.join(reasons) if reasons else 'No specific reason'})")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing stock {stock_code}: {str(e)}")
            return {'signal': 'error', 'reason': str(e)}
    
    def should_buy(self, stock_code):
        """매수 결정
        
        Args:
            stock_code (str): 종목 코드
            
        Returns:
            tuple: (매수 여부, 사유)
        """
        # 포지션 개수 확인
        if len(self.positions) >= self.config['max_position']:
            return False, "최대 포지션 개수 초과"
        
        # 종목 분석
        analysis = self.analyze_stock(stock_code)
        
        # 매수 신호 확인
        if analysis['signal'] == 'buy':
            return True, ', '.join(analysis['reasons'])
        
        return False, "매수 신호 없음"
    
    def should_sell(self, stock_code):
        """매도 결정
        
        Args:
            stock_code (str): 종목 코드
            
        Returns:
            tuple: (매도 여부, 사유)
        """
        # 보유 여부 확인
        if stock_code not in self.positions:
            return False, "보유 종목 아님"
        
        # 종목 정보
        position = self.positions[stock_code]
        avg_price = position['avg_price']
        
        # 현재가 조회
        current_data = self.market_data.get_stock_current_price(stock_code)
        if not current_data:
            return False, "현재가 조회 실패"
        
        current_price = float(current_data['stck_prpr'])  # 현재가
        
        # 수익률 계산
        profit_ratio = (current_price - avg_price) / avg_price
        
        # 손절 확인
        if profit_ratio <= -self.config['stop_loss']:
            return True, f"손절 조건 충족 (수익률: {profit_ratio:.2%})"
        
        # 익절 확인
        if profit_ratio >= self.config['take_profit']:
            return True, f"익절 조건 충족 (수익률: {profit_ratio:.2%})"
        
        # 종목 분석
        analysis = self.analyze_stock(stock_code)
        
        # 매도 신호 확인
        if analysis['signal'] == 'sell':
            return True, ', '.join(analysis['reasons'])
        
        return False, "매도 신호 없음"
    
    def execute_buy(self, stock_code, investment_amount=None):
        """매수 실행
        
        Args:
            stock_code (str): 종목 코드
            investment_amount (float, optional): 투자금액. 미지정시 설정 사용
            
        Returns:
            dict: 주문 결과
        """
        logger.info(f"Executing buy order for {stock_code}")
        
        try:
            # 현재가 조회
            current_data = self.market_data.get_stock_current_price(stock_code)
            if not current_data:
                logger.error(f"Failed to get current price for {stock_code}")
                return None
            
            current_price = float(current_data['stck_prpr'])  # 현재가
            
            # 계좌 잔고 조회
            balance = self.market_data.get_account_balance()
            if not balance:
                logger.error("Failed to get account balance")
                return None
            
            # 가용 현금
            available_cash = float(balance['account_summary'][0]['dnca_tot_amt'])
            
            # 투자금액 계산
            if not investment_amount:
                investment_amount = available_cash * self.config['position_size']
            
            # 매수 가능 수량 계산 (소수점 이하 버림)
            quantity = int(investment_amount / current_price)
            
            if quantity <= 0:
                logger.warning(f"Calculated quantity is 0 or negative: {quantity}")
                return None
            
            # 주문 실행
            order_result = self.order_api.place_order(
                stock_code=stock_code,
                order_type="02",  # 매수
                quantity=quantity,
                price=current_price
            )
            
            if order_result:
                # 포지션 정보 업데이트
                self.positions[stock_code] = {
                    'quantity': quantity,
                    'avg_price': current_price,
                    'buy_time': datetime.now()
                }
                
                logger.info(f"Buy order executed: {stock_code}, {quantity} shares at {current_price}")
                
            return order_result
            
        except Exception as e:
            logger.error(f"Error executing buy order for {stock_code}: {str(e)}")
            return None
    
    def execute_sell(self, stock_code, quantity=None):
        """매도 실행
        
        Args:
            stock_code (str): 종목 코드
            quantity (int, optional): 매도 수량. 미지정시 전량 매도
            
        Returns:
            dict: 주문 결과
        """
        logger.info(f"Executing sell order for {stock_code}")
        
        try:
            # 보유 여부 확인
            if stock_code not in self.positions:
                logger.warning(f"Stock {stock_code} not in positions")
                return None
            
            # 현재가 조회
            current_data = self.market_data.get_stock_current_price(stock_code)
            if not current_data:
                logger.error(f"Failed to get current price for {stock_code}")
                return None
            
            current_price = float(current_data['stck_prpr'])  # 현재가
            
            # 매도 수량 설정
            if not quantity:
                quantity = self.positions[stock_code]['quantity']
            
            # 주문 실행
            order_result = self.order_api.place_order(
                stock_code=stock_code,
                order_type="01",  # 매도
                quantity=quantity,
                price=current_price
            )
            
            if order_result:
                # 포지션 정보 업데이트
                if quantity >= self.positions[stock_code]['quantity']:
                    # 전량 매도면 포지션 삭제
                    del self.positions[stock_code]
                else:
                    # 일부 매도면 수량만 업데이트
                    self.positions[stock_code]['quantity'] -= quantity
                
                logger.info(f"Sell order executed: {stock_code}, {quantity} shares at {current_price}")
                
            return order_result
            
        except Exception as e:
            logger.error(f"Error executing sell order for {stock_code}: {str(e)}")
            return None
    
    def update_positions(self):
        """보유 종목 정보 업데이트"""
        logger.info("Updating positions information")
        
        try:
            # 계좌 잔고 조회
            balance = self.market_data.get_account_balance()
            if not balance:
                logger.error("Failed to get account balance")
                return
            
            # 현재 보유 종목 리스트
            current_positions = {}
            
            # 잔고 내역 확인
            stocks = balance.get('stocks', [])
            for stock in stocks:
                stock_code = stock['pdno']  # 종목코드
                quantity = int(stock['hldg_qty'])  # 보유수량
                avg_price = float(stock['pchs_avg_pric'])  # 평균단가
                
                # 이미 있는 종목이면 매수 시간 유지, 없으면 현재 시간
                buy_time = datetime.now()
                if stock_code in self.positions:
                    buy_time = self.positions[stock_code]['buy_time']
                
                current_positions[stock_code] = {
                    'quantity': quantity,
                    'avg_price': avg_price,
                    'buy_time': buy_time
                }
            
            # 포지션 업데이트
            self.positions = current_positions
            
            logger.info(f"Current positions: {self.positions}")
            
        except Exception as e:
            logger.error(f"Error updating positions: {str(e)}")
    
    def run(self, target_stocks):
        """전략 실행
        
        Args:
            target_stocks (list): 대상 종목 코드 리스트
            
        Returns:
            dict: 실행 결과
        """
        logger.info(f"Running strategy on {len(target_stocks)} stocks")
        
        results = {
            'buys': [],
            'sells': [],
            'errors': []
        }
        
        try:
            # 포지션 정보 업데이트
            self.update_positions()
            
            # 매도 검사 (보유 종목)
            for stock_code in list(self.positions.keys()):
                should_sell, reason = self.should_sell(stock_code)
                
                if should_sell:
                    logger.info(f"Selling {stock_code}: {reason}")
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
                            'error': 'Order execution failed'
                        })
            
            # 매수 검사 (대상 종목)
            for stock_code in target_stocks:
                # 이미 보유중인 종목은 스킵
                if stock_code in self.positions:
                    continue
                    
                should_buy, reason = self.should_buy(stock_code)
                
                if should_buy:
                    logger.info(f"Buying {stock_code}: {reason}")
                    result = self.execute_buy(stock_code)
                    
                    if result:
                        results['buys'].append({
                            'stock_code': stock_code,
                            'reason': reason,
                            'result': result
                        })
                    else:
                        results['errors'].append({
                            'stock_code': stock_code,
                            'action': 'buy',
                            'reason': reason,
                            'error': 'Order execution failed'
                        })
            
            logger.info(f"Strategy execution results: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error running strategy: {str(e)}")
            return results