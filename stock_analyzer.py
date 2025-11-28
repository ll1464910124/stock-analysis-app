import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import tushare as ts
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 设置页面
st.set_page_config(page_title="股票多指标决策系统", page_icon="📈", layout="wide")

class AdvancedTradingDecisionSystem:
    def __init__(self, token):
        self.token = token
        ts.set_token(token)
        self.pro = ts.pro_api()
        
    def get_stock_basic_info(self, ts_code):
        """获取股票基本信息"""
        try:
            df = self.pro.stock_basic(ts_code=ts_code, 
                                     fields='ts_code,symbol,name,area,industry,list_date')
            if not df.empty:
                return df.iloc[0]['name']
            return None
        except:
            return None
        
    def get_stock_data(self, ts_code, start_date, end_date):
        """获取股票数据"""
        try:
            df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            df = df.sort_values('trade_date')
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df.set_index('trade_date', inplace=True)
            return df
        except Exception as e:
            st.error(f"获取数据失败: {e}")
            return None

    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """计算MACD指标"""
        df = df.copy()
        df['EMA_fast'] = df['close'].ewm(span=fast).mean()
        df['EMA_slow'] = df['close'].ewm(span=slow).mean()
        df['MACD'] = df['EMA_fast'] - df['EMA_slow']
        df['MACD_signal'] = df['MACD'].ewm(span=signal).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # 计算MACD斜率和DEA斜率
        df['MACD_slope'] = df['MACD'].diff()
        df['DEA_slope'] = df['MACD_signal'].diff()
        
        return df

    def calculate_ma_system(self, df):
        """计算均线系统"""
        df = df.copy()
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        df['MA60'] = df['close'].rolling(60).mean()
        df['MA120'] = df['close'].rolling(120).mean()
        
        # 计算均线方向
        df['MA20_direction'] = df['MA20'].diff()
        df['MA60_direction'] = df['MA60'].diff()
        df['MA120_direction'] = df['MA120'].diff()
        
        return df

    def calculate_rsi(self, df, periods=[6, 12, 24]):
        """计算RSI指标（多周期）"""
        df = df.copy()
        for period in periods:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        return df

    def calculate_kdj(self, df, n=9, m1=3, m2=3):
        """计算KDJ指标"""
        df = df.copy()
        low_min = df['low'].rolling(window=n).min()
        high_max = df['high'].rolling(window=n).max()
        
        df['RSV'] = (df['close'] - low_min) / (high_max - low_min) * 100
        df['K'] = df['RSV'].ewm(alpha=1/m1).mean()
        df['D'] = df['K'].ewm(alpha=1/m2).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']
        
        # 添加前一日数据用于金叉死叉判断
        df['K_prev'] = df['K'].shift(1)
        df['D_prev'] = df['D'].shift(1)
        return df

    def calculate_bollinger_bands(self, df, period=20, std=2):
        """计算布林带"""
        df = df.copy()
        df['BB_middle'] = df['close'].rolling(window=period).mean()
        bb_std = df['close'].rolling(window=period).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * std)
        df['BB_lower'] = df['BB_middle'] - (bb_std * std)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        
        # 计算布林带位置
        df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        return df

    def calculate_volume_indicators(self, df):
        """计算成交量指标"""
        df = df.copy()
        df['VMA5'] = df['vol'].rolling(5).mean()
        df['VMA20'] = df['vol'].rolling(20).mean()
        df['volume_ratio'] = df['vol'] / df['VMA5']
        
        # 计算OBV
        df['OBV'] = 0
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                df['OBV'].iloc[i] = df['OBV'].iloc[i-1] + df['vol'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                df['OBV'].iloc[i] = df['OBV'].iloc[i-1] - df['vol'].iloc[i]
            else:
                df['OBV'].iloc[i] = df['OBV'].iloc[i-1]
                
        # 计算OBV趋势
        df['OBV_trend'] = df['OBV'].diff()
        return df

    def calculate_atr(self, df, period=14):
        """计算ATR"""
        df = df.copy()
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(period).mean()
        return df

    def calculate_cci(self, df, period=14):
        """计算CCI"""
        df = df.copy()
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        df['CCI'] = (tp - sma) / (0.015 * mad)
        return df

    def calculate_sar(self, df, acceleration=0.02, maximum=0.2):
        """计算SAR指标"""
        df = df.copy()
        high = df['high'].values
        low = df['low'].values
        sar = np.zeros(len(df))
        trend = np.zeros(len(df))
        af = acceleration
        ep = low[0]
        hp = high[0]
        lp = low[0]
        
        sar[0] = low[0] - (high[0] - low[0]) * 0.1
        trend[0] = 1 if sar[0] < low[0] else -1
        
        for i in range(1, len(df)):
            if trend[i-1] < 0:
                sar[i] = sar[i-1] - af * (sar[i-1] - hp)
                if high[i] > hp:
                    af = min(af + acceleration, maximum)
                    hp = high[i]
                if sar[i] < low[i]:
                    trend[i] = -1
                else:
                    trend[i] = 1
                    sar[i] = lp
                    af = acceleration
                    lp = low[i]
            else:
                sar[i] = sar[i-1] + af * (lp - sar[i-1])
                if low[i] < lp:
                    af = min(af + acceleration, maximum)
                    lp = low[i]
                if sar[i] > high[i]:
                    trend[i] = 1
                else:
                    trend[i] = -1
                    sar[i] = hp
                    af = acceleration
                    hp = high[i]
        
        df['SAR'] = sar
        df['SAR_trend'] = trend
        return df

    def calculate_all_indicators(self, df):
        """计算所有技术指标"""
        df = self.calculate_macd(df)
        df = self.calculate_ma_system(df)
        df = self.calculate_rsi(df)
        df = self.calculate_kdj(df)
        df = self.calculate_bollinger_bands(df)
        df = self.calculate_volume_indicators(df)
        df = self.calculate_atr(df)
        df = self.calculate_cci(df)
        df = self.calculate_sar(df)
        return df.dropna()

class TradingDecisionEngine:
    def __init__(self):
        self.decision_rules = self.define_decision_rules()
    
    def define_decision_rules(self):
        """定义分层指挥体系决策规则"""
        rules = {
            'trend_indicators': {
                'MACD': {
                    'S级': {'condition': '(MACD > 0) & (MACD > MACD_signal) & (MACD_slope > 0) & (DEA_slope > 0)', 'score': 40},
                    'A级': {'condition': '(MACD < 0) & (MACD > MACD_signal) & (MACD_hist > 0)', 'score': 15},
                    'B级': {'condition': '(MACD > 0) & (MACD < MACD_signal)', 'score': -20},
                    'C级': {'condition': '(MACD < 0) & (MACD < MACD_signal) & (MACD_hist < 0)', 'score': -40}
                },
                'MA': {
                    '多头排列': {'condition': '(MA20 > MA60) & (MA60 > MA120) & (MA60_direction > 0)', 'score': 10},
                    '空头排列': {'condition': '(MA20 < MA60) & (MA60 < MA120) & (MA60_direction < 0)', 'score': -10},
                    '金叉': {'condition': '(MA60 > MA60.shift(1)) & (MA20 > MA60)', 'score': 5},
                    '纠缠': {'condition': 'abs(MA20-MA60)/MA60 < 0.02', 'score': 0}
                },
                'SAR': {
                    '多头': {'condition': 'SAR_trend > 0', 'score': 2},
                    '空头': {'condition': 'SAR_trend < 0', 'score': -2}
                }
            },
            'volume_indicators': {
                'VMA': {
                    '放量': {'condition': 'volume_ratio > 1.5', 'score': 30},
                    '温和': {'condition': '(volume_ratio > 1.2) & (volume_ratio <= 1.5)', 'score': 10},
                    '缩量': {'condition': 'volume_ratio <= 1.0', 'score': -30}
                },
                'OBV': {
                    '健康': {'condition': '(close > close_prev) & (OBV > OBV_prev)', 'score': 5},
                    '背离': {'condition': '(close > close_prev) & (OBV < OBV_prev)', 'score': -20}
                }
            },
            'momentum_indicators': {
                'RSI': {
                    '强势': {'condition': 'RSI_12 > 50', 'score': 15},
                    '弱势': {'condition': 'RSI_12 <= 50', 'score': 0},
                    '超买': {'condition': 'RSI_12 > 70', 'score': -10},
                    '超卖': {'condition': 'RSI_12 < 30', 'score': 5}
                },
                'KDJ': {
                    '金叉': {'condition': '(K > D) & (K_prev <= D_prev)', 'score': 5},
                    '死叉': {'condition': '(K < D) & (K_prev >= D_prev)', 'score': -5}
                },
                'CCI': {
                    '强势': {'condition': 'CCI > 100', 'score': 3},
                    '弱势': {'condition': 'CCI < -100', 'score': -3}
                }
            },
            'volatility_indicators': {
                'BOLL': {
                    '强势': {'condition': 'close > BB_middle', 'score': 5},
                    '弱势': {'condition': 'close <= BB_middle', 'score': -5},
                    '突破': {'condition': '(BB_position > 0.8) & (volume_ratio > 1.5)', 'score': 2}
                },
                'ATR': {
                    '高波动': {'condition': 'ATR > ATR.rolling(20).mean()', 'score': -2},
                    '低波动': {'condition': 'ATR <= ATR.rolling(20).mean()', 'score': 1}
                }
            }
        }
        return rules

    def evaluate_conditions(self, current_data, prev_data):
        """评估所有条件"""
        scores = {
            'trend_score': 0,
            'volume_score': 0,
            'momentum_score': 0,
            'volatility_score': 0,
            'total_score': 0,
            'signals': [],
            'warnings': [],
            'detailed_analysis': {}
        }
        
        # 准备数据
        data = current_data.copy()
        data['close_prev'] = prev_data['close']
        data['OBV_prev'] = prev_data['OBV']
        data['K_prev'] = prev_data['K']
        data['D_prev'] = prev_data['D']
        
        # 趋势指标评估 (主帅级 - 50%)
        trend_score = 0
        trend_signals = []
        trend_analysis = []
        
        # MACD评估 (元帅)
        macd_conditions = self.decision_rules['trend_indicators']['MACD']
        macd_evaluated = False
        for level, config in macd_conditions.items():
            try:
                if eval(config['condition'], {}, data.to_dict()):
                    trend_score += config['score']
                    trend_signals.append(f"MACD {level}信号")
                    trend_analysis.append(f"MACD({level}): {config['condition']}")
                    macd_evaluated = True
                    break
            except:
                continue
        
        # 均线评估 (将军)
        ma_conditions = self.decision_rules['trend_indicators']['MA']
        for level, config in ma_conditions.items():
            try:
                if eval(config['condition'], {}, data.to_dict()):
                    trend_score += config['score']
                    trend_signals.append(f"均线{level}")
                    trend_analysis.append(f"MA({level}): {config['condition']}")
            except:
                continue
        
        # SAR评估 (先锋)
        sar_conditions = self.decision_rules['trend_indicators']['SAR']
        for level, config in sar_conditions.items():
            try:
                if eval(config['condition'], {}, data.to_dict()):
                    trend_score += config['score']
                    trend_signals.append(f"SAR{level}")
                    trend_analysis.append(f"SAR({level}): {config['condition']}")
            except:
                continue
        
        scores['trend_score'] = trend_score
        scores['signals'].extend(trend_signals)
        scores['detailed_analysis']['trend'] = trend_analysis
        
        # 成交量指标评估 (政委级 - 30%)
        volume_score = 0
        volume_signals = []
        volume_analysis = []
        
        # 成交量评估
        volume_conditions = self.decision_rules['volume_indicators']['VMA']
        for level, config in volume_conditions.items():
            try:
                if eval(config['condition'], {}, data.to_dict()):
                    volume_score += config['score']
                    volume_signals.append(f"成交量{level}")
                    volume_analysis.append(f"Volume({level}): {config['condition']}")
                    break
            except:
                continue
        
        # OBV评估
        obv_conditions = self.decision_rules['volume_indicators']['OBV']
        for level, config in obv_conditions.items():
            try:
                if eval(config['condition'], {}, data.to_dict()):
                    volume_score += config['score']
                    volume_signals.append(f"OBV{level}")
                    volume_analysis.append(f"OBV({level}): {config['condition']}")
                    break
            except:
                continue
        
        scores['volume_score'] = volume_score
        scores['signals'].extend(volume_signals)
        scores['detailed_analysis']['volume'] = volume_analysis
        
        # 动量指标评估 (参谋级 - 15%)
        momentum_score = 0
        momentum_signals = []
        momentum_analysis = []
        
        # RSI评估
        rsi_conditions = self.decision_rules['momentum_indicators']['RSI']
        for level, config in rsi_conditions.items():
            try:
                if eval(config['condition'], {}, data.to_dict()):
                    momentum_score += config['score']
                    momentum_signals.append(f"RSI{level}")
                    momentum_analysis.append(f"RSI({level}): {config['condition']}")
            except:
                continue
        
        # KDJ评估
        kdj_conditions = self.decision_rules['momentum_indicators']['KDJ']
        for level, config in kdj_conditions.items():
            try:
                if eval(config['condition'], {}, data.to_dict()):
                    momentum_score += config['score']
                    momentum_signals.append(f"KDJ{level}")
                    momentum_analysis.append(f"KDJ({level}): {config['condition']}")
                    break
            except:
                continue
        
        # CCI评估
        cci_conditions = self.decision_rules['momentum_indicators']['CCI']
        for level, config in cci_conditions.items():
            try:
                if eval(config['condition'], {}, data.to_dict()):
                    momentum_score += config['score']
                    momentum_signals.append(f"CCI{level}")
                    momentum_analysis.append(f"CCI({level}): {config['condition']}")
            except:
                continue
        
        scores['momentum_score'] = momentum_score
        scores['signals'].extend(momentum_signals)
        scores['detailed_analysis']['momentum'] = momentum_analysis
        
        # 波动率指标评估 (工兵级 - 5%)
        volatility_score = 0
        volatility_signals = []
        volatility_analysis = []
        
        # 布林带评估
        boll_conditions = self.decision_rules['volatility_indicators']['BOLL']
        for level, config in boll_conditions.items():
            try:
                if eval(config['condition'], {}, data.to_dict()):
                    volatility_score += config['score']
                    volatility_signals.append(f"布林带{level}")
                    volatility_analysis.append(f"BOLL({level}): {config['condition']}")
            except:
                continue
        
        # ATR评估
        atr_conditions = self.decision_rules['volatility_indicators']['ATR']
        for level, config in atr_conditions.items():
            try:
                if eval(config['condition'], {}, data.to_dict()):
                    volatility_score += config['score']
                    volatility_signals.append(f"ATR{level}")
                    volatility_analysis.append(f"ATR({level}): {config['condition']}")
            except:
                continue
        
        scores['volatility_score'] = volatility_score
        scores['signals'].extend(volatility_signals)
        scores['detailed_analysis']['volatility'] = volatility_analysis
        
        # 计算总分 (按照分层权重)
        weighted_total = (
            trend_score * 0.50 +      # 趋势指标权重50%
            volume_score * 0.30 +     # 成交量指标权重30%
            momentum_score * 0.15 +   # 动量指标权重15%
            volatility_score * 0.05   # 波动率指标权重5%
        )
        scores['total_score'] = weighted_total
        
        # 生成决策建议
        decision = self.generate_decision(scores)
        scores['decision'] = decision
        
        return scores
    
    def generate_decision(self, scores):
        """生成交易决策"""
        total_score = scores['total_score']
        
        if total_score >= 70:
            return "🚀 强烈买入 (仓位70%+)"
        elif total_score >= 50:
            return "✅ 建议买入 (仓位30-50%)"
        elif total_score >= 30:
            return "🤔 谨慎买入 (仓位<30%)"
        elif total_score >= 0:
            return "⚖️ 持有观望"
        elif total_score >= -30:
            return "🧐 谨慎卖出"
        elif total_score >= -50:
            return "❌ 建议卖出"
        else:
            return "🔥 强烈卖出"

def display_price_charts(df, stock_name):
    """显示价格走势图表（包含MACD）"""
    st.subheader(f"{stock_name} - 价格走势与技术指标")
    
    # 使用plotly创建交互式图表
    fig = make_subplots(rows=3, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.05,
                       subplot_titles=('K线图与均线系统', 'MACD指标', '成交量'),
                       row_heights=[0.5, 0.25, 0.25])
    
    # K线图
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['open'],
                                high=df['high'],
                                low=df['low'],
                                close=df['close'],
                                name='K线'), row=1, col=1)
    
    # 均线
    if 'MA5' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA5'], 
                               line=dict(color='yellow', width=1), name='MA5'), row=1, col=1)
    if 'MA20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], 
                               line=dict(color='orange', width=1), name='MA20'), row=1, col=1)
    if 'MA60' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], 
                               line=dict(color='red', width=1.5), name='MA60'), row=1, col=1)
    if 'MA120' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA120'], 
                               line=dict(color='purple', width=2), name='MA120'), row=1, col=1)
    
    # 布林带
    if 'BB_upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], 
                               line=dict(color='gray', width=1, dash='dash'), 
                               name='布林带上轨', showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], 
                               line=dict(color='gray', width=1, dash='dash'),
                               name='布林带下轨', showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_middle'], 
                               line=dict(color='blue', width=1),
                               name='布林带中轨', showlegend=False), row=1, col=1)
    
    # MACD指标
    if 'MACD' in df.columns:
        # MACD线
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'],
                               line=dict(color='blue', width=1.5), name='MACD'), row=2, col=1)
        # 信号线
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'],
                               line=dict(color='red', width=1.5), name='MACD Signal'), row=2, col=1)
        # 柱状图
        colors = ['green' if x >= 0 else 'red' for x in df['MACD_hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'],
                           name='MACD Hist', marker_color=colors), row=2, col=1)
        # 零轴线
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
    
    # 成交量
    colors = ['red' if row['close'] >= row['open'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['vol'], 
                        name='成交量', marker_color=colors), row=3, col=1)
    
    # 成交量均线
    if 'VMA5' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['VMA5'], 
                               line=dict(color='blue', width=1), name='VMA5'), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True, 
                     xaxis_rangeslider_visible=False)
    
    st.plotly_chart(fig, use_container_width=True)

def display_mini_price_chart(df_period, stock_name):
    """显示迷你版价格走势图"""
    fig = make_subplots(rows=2, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.05,
                       subplot_titles=('K线图', '成交量'),
                       row_heights=[0.7, 0.3])
    
    # K线图
    fig.add_trace(go.Candlestick(x=df_period.index,
                                open=df_period['open'],
                                high=df_period['high'],
                                low=df_period['low'],
                                close=df_period['close'],
                                name='K线'), row=1, col=1)
    
    # 均线
    if 'MA5' in df_period.columns:
        fig.add_trace(go.Scatter(x=df_period.index, y=df_period['MA5'], 
                               line=dict(color='yellow', width=1), name='MA5'), row=1, col=1)
    if 'MA20' in df_period.columns:
        fig.add_trace(go.Scatter(x=df_period.index, y=df_period['MA20'], 
                               line=dict(color='orange', width=1), name='MA20'), row=1, col=1)
    
    # 成交量
    colors = ['red' if row['close'] >= row['open'] else 'green' for _, row in df_period.iterrows()]
    fig.add_trace(go.Bar(x=df_period.index, y=df_period['vol'], 
                        name='成交量', marker_color=colors), row=2, col=1)
    
    fig.update_layout(height=400, showlegend=False, 
                     xaxis_rangeslider_visible=False,
                     margin=dict(l=20, r=20, t=40, b=20))
    
    st.plotly_chart(fig, use_container_width=True)

def get_indicator_status(value, indicator_type, comparison_value=None):
    """获取指标状态和颜色"""
    if indicator_type == 'MACD':
        if value > 0:
            return "🟢", "positive"
        else:
            return "🔴", "negative"
    
    elif indicator_type == 'MACD_hist':
        if value > 0:
            return "🟢", "positive"
        else:
            return "🔴", "negative"
    
    elif indicator_type == 'MACD_signal':
        if value > 0:
            return "🟢", "positive"
        else:
            return "🔴", "negative"
    
    elif indicator_type == 'RSI':
        if value > 70:
            return "🔴", "overbought"
        elif value < 30:
            return "🟢", "oversold"
        else:
            return "🟡", "neutral"
    
    elif indicator_type == 'KDJ':
        if value > 80:
            return "🔴", "overbought"
        elif value < 20:
            return "🟢", "oversold"
        else:
            return "🟡", "neutral"
    
    elif indicator_type == 'volume_ratio':
        if value > 1.5:
            return "🟢", "high"
        elif value > 1.0:
            return "🟡", "medium"
        else:
            return "🔴", "low"
    
    elif indicator_type == 'BB_position':
        if value > 0.8:
            return "🔴", "upper"
        elif value < 0.2:
            return "🟢", "lower"
        else:
            return "🟡", "middle"
    
    elif indicator_type == 'CCI':
        if value > 100:
            return "🟢", "strong"
        elif value < -100:
            return "🔴", "weak"
        else:
            return "🟡", "neutral"
    
    elif indicator_type == 'MA_relation':
        if comparison_value is not None:
            if value > comparison_value:
                return "🟢", "above"
            else:
                return "🔴", "below"
        return "⚪", "unknown"
    
    else:
        return "⚪", "unknown"

def display_technical_indicators_table(df):
    """显示技术指标表格 - 使用Streamlit原生DataFrame样式"""
    st.subheader("📊 技术指标详细分析")
    
    # 获取最近22个交易日的数据（一个月）
    recent_data = df.tail(22).copy()
    
    # 显示迷你价格走势图
    st.write("### 当前分析时间段价格走势（最近一个月）")
    display_mini_price_chart(recent_data, "当前分析")
    
    # 定义指标分组和显示格式
    indicator_configs = {
        'MACD': {'column': 'MACD', 'format': '.4f', 'type': 'MACD'},
        'MACD信号': {'column': 'MACD_signal', 'format': '.4f', 'type': 'MACD_signal'},
        'MACD柱状图': {'column': 'MACD_hist', 'format': '.4f', 'type': 'MACD_hist'},
        'MA5': {'column': 'MA5', 'format': '.2f', 'type': 'MA_relation', 'compare_with': 'close'},
        'MA20': {'column': 'MA20', 'format': '.2f', 'type': 'MA_relation', 'compare_with': 'close'},
        'MA60': {'column': 'MA60', 'format': '.2f', 'type': 'MA_relation', 'compare_with': 'close'},
        'MA120': {'column': 'MA120', 'format': '.2f', 'type': 'MA_relation', 'compare_with': 'close'},
        '成交量': {'column': 'vol', 'format': '.0f', 'type': 'volume_ratio'},
        '成交量比': {'column': 'volume_ratio', 'format': '.2f', 'type': 'volume_ratio'},
        'OBV': {'column': 'OBV', 'format': '.0f', 'type': 'volume_ratio'},
        'RSI_6': {'column': 'RSI_6', 'format': '.1f', 'type': 'RSI'},
        'RSI_12': {'column': 'RSI_12', 'format': '.1f', 'type': 'RSI'},
        'RSI_24': {'column': 'RSI_24', 'format': '.1f', 'type': 'RSI'},
        'K值': {'column': 'K', 'format': '.1f', 'type': 'KDJ'},
        'D值': {'column': 'D', 'format': '.1f', 'type': 'KDJ'},
        'J值': {'column': 'J', 'format': '.1f', 'type': 'KDJ'},
        '布林上轨': {'column': 'BB_upper', 'format': '.2f', 'type': 'BB_position'},
        '布林中轨': {'column': 'BB_middle', 'format': '.2f', 'type': 'BB_position'},
        '布林下轨': {'column': 'BB_lower', 'format': '.2f', 'type': 'BB_position'},
        'ATR': {'column': 'ATR', 'format': '.3f', 'type': 'volume_ratio'},
        'CCI': {'column': 'CCI', 'format': '.1f', 'type': 'CCI'},
    }
    
    # 使用折叠面板让用户选择要显示的指标
    with st.expander("🔧 选择显示指标", expanded=False):
        st.write("选择要显示的技术指标:")
        
        # 使用多列布局
        col1, col2, col3, col4 = st.columns(4)
        
        # 初始化session state
        if 'selected_indicators' not in st.session_state:
            st.session_state.selected_indicators = {
                'MACD': True, 'MACD信号': True, 'MACD柱状图': True,
                'MA5': True, 'MA20': True, 'MA60': True, 'MA120': True,
                '成交量': True, '成交量比': True, 'OBV': True,
                'RSI_6': True, 'RSI_12': True, 'RSI_24': True,
                'K值': True, 'D值': True, 'J值': True,
                '布林上轨': True, '布林中轨': True, '布林下轨': True,
                'ATR': True, 'CCI': True
            }
        
        with col1:
            st.write("**趋势指标 (主帅)**")
            st.session_state.selected_indicators['MACD'] = st.checkbox("MACD", value=st.session_state.selected_indicators['MACD'], key="MACD")
            st.session_state.selected_indicators['MACD信号'] = st.checkbox("MACD信号", value=st.session_state.selected_indicators['MACD信号'], key="MACD信号")
            st.session_state.selected_indicators['MACD柱状图'] = st.checkbox("MACD柱状图", value=st.session_state.selected_indicators['MACD柱状图'], key="MACD柱状图")
            st.session_state.selected_indicators['MA5'] = st.checkbox("MA5", value=st.session_state.selected_indicators['MA5'], key="MA5")
            st.session_state.selected_indicators['MA20'] = st.checkbox("MA20", value=st.session_state.selected_indicators['MA20'], key="MA20")
        
        with col2:
            st.write("**趋势指标 (主帅)**")
            st.session_state.selected_indicators['MA60'] = st.checkbox("MA60", value=st.session_state.selected_indicators['MA60'], key="MA60")
            st.session_state.selected_indicators['MA120'] = st.checkbox("MA120", value=st.session_state.selected_indicators['MA120'], key="MA120")
            
            st.write("**成交量指标 (政委)**")
            st.session_state.selected_indicators['成交量'] = st.checkbox("成交量", value=st.session_state.selected_indicators['成交量'], key="成交量")
            st.session_state.selected_indicators['成交量比'] = st.checkbox("成交量比", value=st.session_state.selected_indicators['成交量比'], key="成交量比")
            st.session_state.selected_indicators['OBV'] = st.checkbox("OBV", value=st.session_state.selected_indicators['OBV'], key="OBV")
        
        with col3:
            st.write("**动量指标 (参谋)**")
            st.session_state.selected_indicators['RSI_6'] = st.checkbox("RSI_6", value=st.session_state.selected_indicators['RSI_6'], key="RSI_6")
            st.session_state.selected_indicators['RSI_12'] = st.checkbox("RSI_12", value=st.session_state.selected_indicators['RSI_12'], key="RSI_12")
            st.session_state.selected_indicators['RSI_24'] = st.checkbox("RSI_24", value=st.session_state.selected_indicators['RSI_24'], key="RSI_24")
            st.session_state.selected_indicators['K值'] = st.checkbox("K值", value=st.session_state.selected_indicators['K值'], key="K值")
            st.session_state.selected_indicators['D值'] = st.checkbox("D值", value=st.session_state.selected_indicators['D值'], key="D值")
        
        with col4:
            st.write("**动量指标 (参谋)**")
            st.session_state.selected_indicators['J值'] = st.checkbox("J值", value=st.session_state.selected_indicators['J值'], key="J值")
            
            st.write("**波动率指标 (工兵)**")
            st.session_state.selected_indicators['布林上轨'] = st.checkbox("布林上轨", value=st.session_state.selected_indicators['布林上轨'], key="布林上轨")
            st.session_state.selected_indicators['布林中轨'] = st.checkbox("布林中轨", value=st.session_state.selected_indicators['布林中轨'], key="布林中轨")
            st.session_state.selected_indicators['布林下轨'] = st.checkbox("布林下轨", value=st.session_state.selected_indicators['布林下轨'], key="布林下轨")
            st.session_state.selected_indicators['ATR'] = st.checkbox("ATR", value=st.session_state.selected_indicators['ATR'], key="ATR")
            st.session_state.selected_indicators['CCI'] = st.checkbox("CCI", value=st.session_state.selected_indicators['CCI'], key="CCI")
    
    # 根据用户选择过滤指标
    selected_indicators = {}
    for indicator_name, config in indicator_configs.items():
        if st.session_state.selected_indicators.get(indicator_name, False):
            selected_indicators[indicator_name] = config
    
    if not selected_indicators:
        st.warning("请至少选择一个指标")
        return
    
    # 创建表格数据 - 真正的横向日期排列
    table_data = []
    
    # 为每个选中的指标添加行
    for indicator_name, config in selected_indicators.items():
        if config['column'] not in recent_data.columns:
            continue
            
        row_data = {'指标': indicator_name}
        
        for date in recent_data.index:
            value = recent_data.loc[date, config['column']]
            
            # 格式化数值
            formatted_value = format(value, config['format'])
            
            # 获取状态和颜色
            if config['type'] == 'MA_relation' and 'compare_with' in config:
                compare_value = recent_data.loc[date, config['compare_with']]
                status_emoji, status_type = get_indicator_status(value, config['type'], compare_value)
            else:
                status_emoji, status_type = get_indicator_status(value, config['type'])
            
            # 添加数值和状态 - 使用日期作为列名
            date_str = date.strftime('%m-%d')
            row_data[f'{date_str} 数值'] = formatted_value
            row_data[f'{date_str} 状态'] = status_emoji
        
        table_data.append(row_data)
    
    # 创建DataFrame并显示
    if table_data:
        display_df = pd.DataFrame(table_data)
        
        # 使用Streamlit原生DataFrame显示，保持Excel样式
        st.dataframe(display_df, use_container_width=True, height=min(600, len(selected_indicators) * 35 + 100))
        
        # 显示颜色说明
        st.write("**颜色说明**: 🟢 积极信号 | 🔴 消极信号 | 🟡 中性信号 | ⚪ 未知状态")
        
        # 添加同步滚动功能的CSS和JavaScript
        st.markdown("""
        <style>
        /* 确保表格容器有滚动条 */
        .stDataFrame {
            overflow-x: auto;
        }
        
        /* 为表格添加边框样式，更像Excel */
        .stDataFrame table {
            border-collapse: collapse;
            border-spacing: 0;
        }
        
        .stDataFrame th, .stDataFrame td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }
        
        .stDataFrame th {
            background-color: #f2f2f2;
            position: sticky;
            top: 0;
        }
        
        /* 第一列特殊样式 */
        .stDataFrame th:first-child,
        .stDataFrame td:first-child {
            background-color: #f8f9fa;
            font-weight: bold;
            position: sticky;
            left: 0;
            z-index: 1;
        }
        </style>
        
        <script>
        // 同步滚动功能
        function syncScroll() {
            const tables = document.querySelectorAll('.stDataFrame');
            const plots = document.querySelectorAll('.js-plotly-plot');
            
            // 为所有表格和图表添加滚动监听
            [...tables, ...plots].forEach(element => {
                element.addEventListener('scroll', function(e) {
                    const scrollLeft = e.target.scrollLeft;
                    
                    // 同步所有元素的滚动位置
                    [...tables, ...plots].forEach(otherElement => {
                        if (otherElement !== e.target) {
                            otherElement.scrollLeft = scrollLeft;
                        }
                    });
                });
            });
        }
        
        // 页面加载后执行同步滚动设置
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', syncScroll);
        } else {
            syncScroll();
        }
        
        // 监听Streamlit的内容变化
        const observer = new MutationObserver(syncScroll);
        observer.observe(document.body, { childList: true, subtree: true });
        </script>
        """, unsafe_allow_html=True)
    else:
        st.info("无可用指标数据")

def display_decision_analysis(df):
    """显示决策分析"""
    st.subheader("🤖 多指标决策分析")
    
    # 初始化决策引擎
    decision_engine = TradingDecisionEngine()
    
    # 获取最新数据
    if len(df) < 2:
        st.warning("数据不足进行决策分析")
        return
    
    current_data = df.iloc[-1]
    prev_data = df.iloc[-2]
    
    # 评估当前状态
    scores = decision_engine.evaluate_conditions(current_data, prev_data)
    
    # 显示决策结果
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("最终决策", scores['decision'])
    
    with col2:
        st.metric("综合得分", f"{scores['total_score']:.1f}分")
    
    with col3:
        st.metric("趋势得分", f"{scores['trend_score']}分")
    
    with col4:
        st.metric("量能得分", f"{scores['volume_score']}分")
    
    # 显示详细得分
    st.write("#### 📊 分层指挥体系得分详情")
    
    score_data = {
        '指标类别': ['趋势指标 (主帅)', '成交量指标 (政委)', '动量指标 (参谋)', '波动率指标 (工兵)'],
        '得分': [scores['trend_score'], scores['volume_score'], 
                scores['momentum_score'], scores['volatility_score']],
        '权重': ['50%', '30%', '15%', '5%'],
        '加权得分': [
            f"{scores['trend_score'] * 0.50:.1f}",
            f"{scores['volume_score'] * 0.30:.1f}", 
            f"{scores['momentum_score'] * 0.15:.1f}",
            f"{scores['volatility_score'] * 0.05:.1f}"
        ]
    }
    
    score_df = pd.DataFrame(score_data)
    st.dataframe(score_df, use_container_width=True)
    
    # 显示信号列表
    st.write("#### 📈 技术信号")
    
    if scores['signals']:
        for signal in scores['signals']:
            st.write(f"- {signal}")
    else:
        st.write("暂无明确技术信号")
    
    # 显示详细分析
    with st.expander("🔍 详细指标分析"):
        for category, analyses in scores['detailed_analysis'].items():
            if analyses:
                st.write(f"**{category.upper()}指标分析:**")
                for analysis in analyses:
                    st.write(f"- {analysis}")

def display_indicator_details(df):
    """显示指标详细分析 - 按照分层指挥体系"""
    st.subheader("🎯 分层指挥体系指标详解")
    
    # 获取最新数据
    current_data = df.iloc[-1]
    prev_data = df.iloc[-2] if len(df) > 1 else current_data
    
    # 趋势指标分析 (主帅级)
    st.write("### 🎖️ 趋势指标分析 (主帅级 - 定方向)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### MACD (元帅)")
        # MACD状态分析
        macd_status = "金叉" if current_data['MACD'] > current_data['MACD_signal'] else "死叉"
        macd_position = "0轴上方" if current_data['MACD'] > 0 else "0轴下方"
        macd_slope_status = "向上" if current_data['MACD_slope'] > 0 else "向下"
        dea_slope_status = "向上" if current_data['DEA_slope'] > 0 else "向下"
        
        st.write(f"- **状态**: {macd_status} | {macd_position}")
        st.write(f"- **MACD值**: {current_data['MACD']:.4f}")
        st.write(f"- **信号线**: {current_data['MACD_signal']:.4f}")
        st.write(f"- **柱状图**: {current_data['MACD_hist']:.4f}")
        st.write(f"- **MACD斜率**: {macd_slope_status}")
        st.write(f"- **DEA斜率**: {dea_slope_status}")
        
        # MACD信号分级
        if (current_data['MACD'] > 0 and 
            current_data['MACD'] > current_data['MACD_signal'] and 
            current_data['MACD_slope'] > 0 and 
            current_data['DEA_slope'] > 0):
            st.success("**S级信号**: 0轴上方金叉 + DEA斜率>0 → 满仓信号")
        elif (current_data['MACD'] < 0 and 
              current_data['MACD'] > current_data['MACD_signal'] and 
              current_data['MACD_hist'] > 0):
            st.info("**A级信号**: 0轴下方金叉但红柱持续放大 → 试仓信号")
        elif (current_data['MACD'] > 0 and 
              current_data['MACD'] < current_data['MACD_signal']):
            st.warning("**B级信号**: 死叉但未破0轴 → 减仓")
        elif (current_data['MACD'] < 0 and 
              current_data['MACD'] < current_data['MACD_signal'] and 
              current_data['MACD_hist'] < 0):
            st.error("**C级信号**: 0轴下方死叉 + 绿柱放大 → 空仓")
    
    with col2:
        st.write("#### 均线系统 (将军)")
        # 均线排列分析
        ma20_60 = current_data['MA20'] > current_data['MA60']
        ma60_120 = current_data['MA60'] > current_data['MA120']
        ma60_direction = "向上" if current_data['MA60_direction'] > 0 else "向下"
        
        if ma20_60 and ma60_120 and current_data['MA60_direction'] > 0:
            st.success("**多头排列**: MA20>MA60>MA120 + MA60向上")
            st.write("- **策略**: 任何回踩都是买点")
        elif not ma20_60 and not ma60_120 and current_data['MA60_direction'] < 0:
            st.error("**空头排列**: MA20<MA60<MA120 + MA60向下")
            st.write("- **策略**: 反弹减仓")
        else:
            st.warning("**纠结状态**: 均线方向不明")
            st.write("- **策略**: 观望等待方向")
        
        st.write(f"- **MA20**: {current_data['MA20']:.2f}")
        st.write(f"- **MA60**: {current_data['MA60']:.2f} ({ma60_direction})")
        st.write(f"- **MA120**: {current_data['MA120']:.2f}")
        
        # 均线金叉分析
        if (current_data['MA60'] > current_data['MA60_direction'] and 
            current_data['MA20'] > current_data['MA60']):
            st.info("**MA60上穿MA120金叉**: 牛熊转换信号")
    
    # 成交量指标分析 (政委级)
    st.write("### 📊 成交量指标分析 (政委级 - 验真伪)")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("#### 成交量分析")
        volume_ratio = current_data['volume_ratio']
        if volume_ratio > 1.5:
            volume_status = "🟢 放量"
            st.success(f"**放量**: 比率{volume_ratio:.2f}倍")
            st.write("- **策略**: 真信号，可参与")
        elif volume_ratio > 1.2:
            volume_status = "🟡 温和"
            st.info(f"**温和**: 比率{volume_ratio:.2f}倍")
            st.write("- **策略**: 正常参与")
        else:
            volume_status = "🔴 缩量"
            st.error(f"**缩量**: 比率{volume_ratio:.2f}倍")
            st.write("- **策略**: 假信号，不参与")
        
        st.write(f"- **成交量**: {current_data['vol']:.0f}")
        st.write(f"- **VMA5**: {current_data['VMA5']:.0f}")
    
    with col4:
        st.write("#### OBV能量潮")
        obv_trend = "上升" if current_data['OBV'] > prev_data['OBV'] else "下降"
        price_trend = "上升" if current_data['close'] > prev_data['close'] else "下降"
        
        if price_trend == "上升" and obv_trend == "上升":
            st.success("**健康上涨**: 价涨量增")
            st.write("- **策略**: 可持有")
        elif price_trend == "上升" and obv_trend == "下降":
            st.warning("**顶背离**: 价涨量缩")
            st.write("- **策略**: 准备减仓")
        elif price_trend == "下降" and obv_trend == "下降":
            st.error("**正常下跌**: 价跌量缩")
            st.write("- **策略**: 别抄底")
        elif price_trend == "下降" and obv_trend == "上升":
            st.info("**底背离**: 价跌量增")
            st.write("- **策略**: 关注机会")
        
        st.write(f"- **OBV**: {current_data['OBV']:.0f}")
        st.write(f"- **趋势**: {obv_trend}")
    
    # 动量指标分析 (参谋级)
    st.write("### ⚡ 动量指标分析 (参谋级 - 找时机)")
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.write("#### RSI分析")
        rsi_6 = current_data['RSI_6']
        rsi_12 = current_data['RSI_12']
        rsi_24 = current_data['RSI_24']
        
        # RSI多周期分析
        st.write(f"- **RSI_6**: {rsi_6:.1f}")
        st.write(f"- **RSI_12**: {rsi_12:.1f}")
        st.write(f"- **RSI_24**: {rsi_24:.1f}")
        
        if rsi_12 > 70:
            st.error("**超买区域**: RSI>70")
            st.write("- **策略**: 谨慎，可能回调")
        elif rsi_12 < 30:
            st.success("**超卖区域**: RSI<30")
            st.write("- **策略**: 关注反弹机会")
        elif rsi_12 > 50:
            st.info("**强势区域**: RSI>50")
            st.write("- **策略**: 持仓线之上")
        else:
            st.warning("**弱势区域**: RSI<50")
            st.write("- **策略**: 减仓线之下")
    
    with col6:
        st.write("#### KDJ分析")
        kdj_cross = "金叉" if current_data['K'] > current_data['D'] else "死叉"
        k_prev = prev_data['K'] if 'K' in prev_data else current_data['K']
        d_prev = prev_data['D'] if 'D' in prev_data else current_data['D']
        fresh_cross = (current_data['K'] > current_data['D'] and k_prev <= d_prev) or \
                     (current_data['K'] < current_data['D'] and k_prev >= d_prev)
        
        st.write(f"- **K值**: {current_data['K']:.1f}")
        st.write(f"- **D值**: {current_data['D']:.1f}")
        st.write(f"- **J值**: {current_data['J']:.1f}")
        st.write(f"- **状态**: {kdj_cross}")
        
        if fresh_cross:
            if current_data['K'] > current_data['D']:
                st.success("**新鲜金叉**: 买入时机")
            else:
                st.error("**新鲜死叉**: 卖出时机")
        else:
            st.info("**延续状态**: 保持现有策略")
    
    # 波动率指标分析 (工兵级)
    st.write("### 📏 波动率指标分析 (工兵级 - 划边界)")
    
    col7, col8 = st.columns(2)
    
    with col7:
        st.write("#### 布林带分析")
        boll_position = current_data['BB_position']
        if boll_position > 0.8:
            boll_status = "🔴 上轨压力"
            st.error("**上轨压力**: 位置{:.2f}".format(boll_position))
            st.write("- **策略**: 减仓30%")
        elif boll_position < 0.2:
            boll_status = "🟢 下轨支撑"
            st.success("**下轨支撑**: 位置{:.2f}".format(boll_position))
            st.write("- **策略**: 关注支撑")
        else:
            boll_status = "🟡 中轨附近"
            st.info("**中轨附近**: 位置{:.2f}".format(boll_position))
            st.write("- **策略**: 正常持仓")
        
        st.write(f"- **上轨**: {current_data['BB_upper']:.2f}")
        st.write(f"- **中轨**: {current_data['BB_middle']:.2f}")
        st.write(f"- **下轨**: {current_data['BB_lower']:.2f}")
    
    with col8:
        st.write("#### ATR波动分析")
        atr_value = current_data['ATR']
        atr_ma = df['ATR'].rolling(20).mean().iloc[-1]
        
        st.write(f"- **ATR**: {atr_value:.3f}")
        st.write(f"- **20日均值**: {atr_ma:.3f}")
        
        if atr_value > atr_ma:
            st.warning("**高波动期**: ATR高于均值")
            st.write("- **策略**: 止损放宽1.5倍")
        else:
            st.success("**低波动期**: ATR低于均值")
            st.write("- **策略**: 正常止损")
        
        # 计算止损位
        if 'close' in current_data:
            stop_loss = current_data['close'] - atr_value * 1.5
            st.write(f"- **建议止损**: {stop_loss:.2f}")

def main():
    st.title("🎖️ 股票多指标决策系统")
    st.markdown("基于**分层指挥体系**的智能交易决策平台")
    
    # 侧边栏配置
    st.sidebar.header("配置参数")
    
    # Tushare token输入
    token = st.sidebar.text_input("Tushare API Token", type="password", 
                                 help="请在Tushare官网注册获取API Token")
    
    if not token:
        st.warning("请输入Tushare API Token以继续")
        st.info("""
        **如何获取Tushare Token:**
        1. 访问 [Tushare官网](https://tushare.pro) 注册账号
        2. 在个人中心获取API Token
        3. 将Token粘贴到左侧输入框中
        """)
        return
    
    # 初始化分析器
    analyzer = AdvancedTradingDecisionSystem(token)
    
    # 股票代码输入
    ts_code = st.sidebar.text_input("股票代码", "000001.SZ", 
                                   help="格式：代码.交易所，如000001.SZ, 600000.SH")
    
    # 自动获取股票名称
    stock_name = "未知股票"
    if ts_code:
        with st.spinner("正在获取股票信息..."):
            name = analyzer.get_stock_basic_info(ts_code)
            if name:
                stock_name = name
                st.sidebar.success(f"股票名称: {stock_name}")
            else:
                st.sidebar.warning("未能自动获取股票名称，请检查代码格式")
    
    # 日期范围选择
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
    
    col3, col4 = st.sidebar.columns(2)
    with col3:
        start_date_input = st.text_input("开始日期", start_date)
    with col4:
        end_date_input = st.text_input("结束日期", end_date)
    
    # 获取数据
    if st.sidebar.button("开始分析", type="primary"):
        with st.spinner("正在获取数据并计算指标..."):
            try:
                # 获取股票数据
                df = analyzer.get_stock_data(ts_code, start_date_input, end_date_input)
                
                if df is None or df.empty:
                    st.error("未能获取到股票数据，请检查股票代码和日期范围")
                    return
                
                # 计算所有技术指标
                df_with_indicators = analyzer.calculate_all_indicators(df)
                
                # 显示基本信息
                st.subheader(f"🎯 {stock_name} ({ts_code}) 多指标决策分析")
                
                # 显示分层指挥体系说明
                with st.expander("🎖️ 分层指挥体系说明", expanded=True):
                    st.write("""
                    ### 分层指挥体系 - 优先级铁律
                    
                    | 类别        | **作战任务**        | **主/辅级别**     | **使用场景** | **信号权重** |
                    | :-------- | :-------------- | :------------ | :------- | :------- |
                    | **趋势指标**  | **定方向**（能不能做）   | **主帅**（最高优先级） | 日线以上周期   | **50%**  |
                    | **成交量指标** | **验真伪**（是不是骗）   | **政委**（一票否决制） | 所有场景     | **30%**  |
                    | **动量指标**  | **找时机**（何时进出）   | **参谋**（辅助确认）  | 60分钟-日线  | **15%**  |
                    | **波动率指标** | **划边界**（目标位/止损） | **工兵**（技术支撑）  | 入场后管理    | **5%**   |
                    
                    **优先级铁律**: 
                    - 趋势指标定仓位（50%+还是空仓）
                    - 成交量定是否入场（达标才执行）
                    - 动量指标定买卖点（精细优化）
                    """)
                
                # 创建标签页
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📈 价格走势", "📊 技术指标", "🤖 决策分析", "🔍 指标详解"
                ])
                
                with tab1:
                    display_price_charts(df_with_indicators, stock_name)
                
                with tab2:
                    display_technical_indicators_table(df_with_indicators)
                
                with tab3:
                    display_decision_analysis(df_with_indicators)
                
                with tab4:
                    display_indicator_details(df_with_indicators)
                    
            except Exception as e:
                st.error(f"分析过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()
