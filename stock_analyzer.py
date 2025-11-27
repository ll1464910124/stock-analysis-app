import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import tushare as ts
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 设置页面
st.set_page_config(page_title="股票技术分析平台", page_icon="📈", layout="wide")

class DynamicWeightOptimizer:
    def __init__(self):
        self.optimization_history = []
    
    def prepare_features_for_optimization(self, df):
        """准备用于权重优化的特征数据"""
        features = {}
        
        # MACD特征
        features['macd_golden_cross'] = ((df['MACD'] > df['MACD_signal']) & (df['MACD_hist'] > 0)).astype(int)
        features['macd_death_cross'] = ((df['MACD'] < df['MACD_signal']) & (df['MACD_hist'] < 0)).astype(int)
        features['macd_above_zero'] = (df['MACD'] > 0).astype(int)
        
        # RSI特征
        features['rsi_oversold'] = (df['RSI'] < 30).astype(int)
        features['rsi_overbought'] = (df['RSI'] > 70).astype(int)
        features['rsi_bullish'] = (df['RSI'] > 50).astype(int)
        
        # 布林带特征
        features['bollinger_oversold'] = (df['BB_position'] < 0.2).astype(int)
        features['bollinger_overbought'] = (df['BB_position'] > 0.8).astype(int)
        features['bollinger_middle'] = ((df['BB_position'] >= 0.4) & (df['BB_position'] <= 0.6)).astype(int)
        
        # KDJ特征
        features['kdj_oversold'] = ((df['K'] < 20) & (df['D'] < 20)).astype(int)
        features['kdj_overbought'] = ((df['K'] > 80) & (df['D'] > 80)).astype(int)
        if 'K_prev' in df.columns and 'D_prev' in df.columns:
            features['kdj_golden_cross'] = ((df['K'] > df['D']) & (df['K_prev'] <= df['D_prev'])).astype(int)
            features['kdj_death_cross'] = ((df['K'] < df['D']) & (df['K_prev'] >= df['D_prev'])).astype(int)
        
        # 成交量特征
        volume_ma = df['vol'].rolling(5).mean()
        volume_ratio = df['vol'] / volume_ma
        features['volume_surge'] = (volume_ratio > 1.5).astype(int)
        features['volume_decline'] = (volume_ratio < 0.7).astype(int)
        
        # 价格特征
        features['price_up'] = (df['close'] > df['open']).astype(int)
        features['price_strong_up'] = ((df['close'] - df['open']) / df['open'] > 0.02).astype(int)
        
        # 创建特征DataFrame
        feature_df = pd.DataFrame(features, index=df.index)
        feature_df = feature_df.dropna()
        
        return feature_df
    
    def calculate_signal_score(self, features, weights):
        """根据权重计算信号得分"""
        signal_score = pd.Series(0, index=features.index)
        
        for feature, weight in weights.items():
            if feature in features.columns:
                signal_score += features[feature] * weight
        
        return signal_score
    
    def evaluate_weights(self, features, actual_moves, weights, hold_days=5):
        """评估权重配置的效果"""
        signal_scores = self.calculate_signal_score(features, weights)
        
        # 生成交易信号 (1:买入, -1:卖出, 0:持有)
        buy_threshold = 0.3
        sell_threshold = -0.3
        
        predictions = []
        for score in signal_scores:
            if score > buy_threshold:
                predictions.append(1)  # 买入
            elif score < sell_threshold:
                predictions.append(-1) # 卖出
            else:
                predictions.append(0)  # 持有
        
        predictions = pd.Series(predictions, index=signal_scores.index)
        
        # 只评估有信号的时间点
        signal_mask = predictions != 0
        if not signal_mask.any():
            return 0, 0, 0
        
        signal_predictions = predictions[signal_mask]
        signal_actual = actual_moves[signal_mask]
        
        # 计算准确率
        correct_predictions = (signal_predictions == signal_actual).sum()
        total_signals = len(signal_predictions)
        accuracy = correct_predictions / total_signals if total_signals > 0 else 0
        
        # 计算买入信号准确率
        buy_mask = signal_predictions == 1
        buy_accuracy = (signal_actual[buy_mask] == 1).sum() / len(signal_actual[buy_mask]) if buy_mask.any() else 0
        
        # 计算卖出信号准确率
        sell_mask = signal_predictions == -1
        sell_accuracy = (signal_actual[sell_mask] == -1).sum() / len(signal_actual[sell_mask]) if sell_mask.any() else 0
        
        return accuracy, buy_accuracy, sell_accuracy
    
    def optimize_weights_genetic(self, features, actual_moves, population_size=50, generations=100, hold_days=5):
        """使用遗传算法优化权重"""
        feature_names = features.columns.tolist()
        n_features = len(feature_names)
        
        # 初始化种群
        population = []
        for _ in range(population_size):
            weights = {name: np.random.uniform(-1, 1) for name in feature_names}
            population.append(weights)
        
        best_weights = None
        best_accuracy = 0
        history = []
        
        for generation in range(generations):
            # 评估种群
            accuracies = []
            for weights in population:
                accuracy, _, _ = self.evaluate_weights(features, actual_moves, weights, hold_days)
                accuracies.append(accuracy)
            
            # 选择最佳个体
            best_idx = np.argmax(accuracies)
            if accuracies[best_idx] > best_accuracy:
                best_accuracy = accuracies[best_idx]
                best_weights = population[best_idx].copy()
            
            history.append({
                'generation': generation,
                'best_accuracy': best_accuracy,
                'avg_accuracy': np.mean(accuracies)
            })
            
            # 选择（轮盘赌选择）
            accuracies = np.array(accuracies)
            if accuracies.sum() > 0:
                probabilities = accuracies / accuracies.sum()
            else:
                probabilities = np.ones(len(accuracies)) / len(accuracies)
            
            selected_indices = np.random.choice(
                len(population), 
                size=population_size, 
                p=probabilities
            )
            selected_population = [population[i] for i in selected_indices]
            
            # 交叉和变异
            new_population = []
            for i in range(0, population_size, 2):
                if i + 1 < population_size:
                    parent1 = selected_population[i]
                    parent2 = selected_population[i + 1]
                    
                    # 交叉
                    child1, child2 = self.crossover(parent1, parent2)
                    
                    # 变异
                    child1 = self.mutate(child1, mutation_rate=0.1)
                    child2 = self.mutate(child2, mutation_rate=0.1)
                    
                    new_population.extend([child1, child2])
                else:
                    new_population.append(selected_population[i])
            
            population = new_population
        
        return best_weights, best_accuracy, history
    
    def crossover(self, parent1, parent2):
        """交叉操作"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # 随机选择一些特征进行交换
        features = list(parent1.keys())
        crossover_point = np.random.randint(1, len(features))
        
        for i in range(crossover_point):
            feature = features[i]
            child1[feature], child2[feature] = child2[feature], child1[feature]
        
        return child1, child2
    
    def mutate(self, individual, mutation_rate=0.1):
        """变异操作"""
        mutated = individual.copy()
        
        for feature in mutated:
            if np.random.random() < mutation_rate:
                mutated[feature] += np.random.normal(0, 0.2)
                # 限制在[-1, 1]范围内
                mutated[feature] = max(-1, min(1, mutated[feature]))
        
        return mutated
    
    def backtest_optimization(self, df, optimization_days=90, hold_days=5):
        """回测权重优化效果"""
        features = self.prepare_features_for_optimization(df)
        
        # 计算实际价格变动
        price_changes = df['close'].pct_change(hold_days).shift(-hold_days)
        actual_moves = np.sign(price_changes)  # 1:上涨, -1:下跌, 0:平盘
        actual_moves = actual_moves[features.index]
        
        # 时间序列分割
        tscv = TimeSeriesSplit(n_splits=5)
        fold_results = []
        
        for train_idx, test_idx in tscv.split(features):
            if len(train_idx) < optimization_days:
                continue
                
            # 使用训练数据优化权重
            train_features = features.iloc[train_idx[-optimization_days:]]
            train_actual = actual_moves.iloc[train_idx[-optimization_days:]]
            
            best_weights, best_accuracy, history = self.optimize_weights_genetic(
                train_features, train_actual, 
                population_size=30, generations=50, hold_days=hold_days
            )
            
            # 在测试数据上评估
            test_features = features.iloc[test_idx]
            test_actual = actual_moves.iloc[test_idx]
            
            test_accuracy, test_buy_accuracy, test_sell_accuracy = self.evaluate_weights(
                test_features, test_actual, best_weights, hold_days
            )
            
            fold_results.append({
                'train_accuracy': best_accuracy,
                'test_accuracy': test_accuracy,
                'buy_accuracy': test_buy_accuracy,
                'sell_accuracy': test_sell_accuracy,
                'weights': best_weights,
                'test_size': len(test_idx)
            })
        
        return fold_results
    
    def interpret_optimized_weights(self, weights):
        """解释优化后的权重"""
        positive_weights = {k: v for k, v in weights.items() if v > 0.1}
        negative_weights = {k: v for k, v in weights.items() if v < -0.1}
        
        positive_sorted = dict(sorted(positive_weights.items(), key=lambda x: x[1], reverse=True))
        negative_sorted = dict(sorted(negative_weights.items(), key=lambda x: x[1]))
        
        interpretation = {
            'strong_buy_signals': list(positive_sorted.keys())[:5],
            'strong_sell_signals': list(negative_sorted.keys())[:5],
            'top_positive_weights': positive_sorted,
            'top_negative_weights': negative_sorted
        }
        
        return interpretation

class IntelligentAnalyzer:
    def __init__(self):
        self.analysis_rules = self.define_quantitative_rules()
    
    def define_quantitative_rules(self):
        """定义量化分析规则"""
        rules = {
            # 趋势判断规则
            'trend_rules': {
                'strong_bullish': {'conditions': 4, 'weight': 1.0},
                'bullish': {'conditions': 3, 'weight': 0.7},
                'neutral': {'conditions': 2, 'weight': 0.5},
                'bearish': {'conditions': 1, 'weight': 0.3},
                'strong_bearish': {'conditions': 0, 'weight': 0.1}
            },
            
            # 买入信号规则
            'buy_signals': {
                'macd_golden_cross': {'threshold': 0, 'weight': 0.15},
                'rsi_oversold': {'threshold': 30, 'weight': 0.15},
                'bollinger_oversold': {'threshold': 0.2, 'weight': 0.15},
                'kdj_oversold': {'threshold': 20, 'weight': 0.1},
                'volume_surge': {'threshold': 1.5, 'weight': 0.1},
                'price_support': {'threshold': 0.02, 'weight': 0.1},
                'ml_bullish': {'threshold': 0.6, 'weight': 0.25}
            },
            
            # 卖出信号规则
            'sell_signals': {
                'macd_death_cross': {'threshold': 0, 'weight': 0.15},
                'rsi_overbought': {'threshold': 70, 'weight': 0.15},
                'bollinger_overbought': {'threshold': 0.8, 'weight': 0.15},
                'kdj_overbought': {'threshold': 80, 'weight': 0.1},
                'volume_decline': {'threshold': 0.7, 'weight': 0.1},
                'price_resistance': {'threshold': 0.02, 'weight': 0.1},
                'ml_bearish': {'threshold': 0.4, 'weight': 0.25}
            },
            
            # 风险控制规则
            'risk_control': {
                'max_position_score': 0.8,
                'min_position_score': 0.3,
                'stop_loss_threshold': -0.05,
                'take_profit_threshold': 0.15
            }
        }
        return rules
    
    def calculate_technical_score(self, df):
        """计算技术指标综合得分"""
        current_data = df.iloc[-1]
        scores = {
            'buy_score': 0,
            'sell_score': 0,
            'signals': [],
            'warnings': []
        }
        
        # MACD分析
        macd_score = self.analyze_macd(current_data)
        scores['buy_score'] += macd_score['buy']
        scores['sell_score'] += macd_score['sell']
        scores['signals'].extend(macd_score['signals'])
        
        # RSI分析
        rsi_score = self.analyze_rsi(current_data)
        scores['buy_score'] += rsi_score['buy']
        scores['sell_score'] += rsi_score['sell']
        scores['signals'].extend(rsi_score['signals'])
        
        # 布林带分析
        bollinger_score = self.analyze_bollinger_bands(current_data, df)
        scores['buy_score'] += bollinger_score['buy']
        scores['sell_score'] += bollinger_score['sell']
        scores['signals'].extend(bollinger_score['signals'])
        
        # KDJ分析
        kdj_score = self.analyze_kdj(current_data)
        scores['buy_score'] += kdj_score['buy']
        scores['sell_score'] += kdj_score['sell']
        scores['signals'].extend(kdj_score['signals'])
        
        # 成交量分析
        volume_score = self.analyze_volume(current_data, df)
        scores['buy_score'] += volume_score['buy']
        scores['sell_score'] += volume_score['sell']
        scores['signals'].extend(volume_score['signals'])
        
        # 价格走势分析
        price_score = self.analyze_price_action(df)
        scores['buy_score'] += price_score['buy']
        scores['sell_score'] += price_score['sell']
        scores['signals'].extend(price_score['signals'])
        scores['warnings'].extend(price_score['warnings'])
        
        return scores
    
    def analyze_macd(self, data):
        """MACD指标量化分析"""
        score = {'buy': 0, 'sell': 0, 'signals': []}
        
        try:
            # MACD金叉判断
            if data['MACD'] > data['MACD_signal'] and data['MACD_hist'] > 0:
                score['buy'] += self.analysis_rules['buy_signals']['macd_golden_cross']['weight']
                score['signals'].append("✅ MACD金叉，看涨信号")
            
            # MACD死叉判断
            if data['MACD'] < data['MACD_signal'] and data['MACD_hist'] < 0:
                score['sell'] += self.analysis_rules['sell_signals']['macd_death_cross']['weight']
                score['signals'].append("❌ MACD死叉，看跌信号")
            
            # MACD零轴位置
            if data['MACD'] > 0:
                score['buy'] += 0.05
                score['signals'].append("📈 MACD在零轴上方，多头市场")
            else:
                score['sell'] += 0.05
                score['signals'].append("📉 MACD在零轴下方，空头市场")
                
        except KeyError as e:
            score['signals'].append(f"⚠️ MACD数据不完整: {e}")
            
        return score
    
    def analyze_rsi(self, data):
        """RSI指标量化分析"""
        score = {'buy': 0, 'sell': 0, 'signals': []}
        
        try:
            rsi = data['RSI']
            rsi_oversold = self.analysis_rules['buy_signals']['rsi_oversold']['threshold']
            rsi_overbought = self.analysis_rules['sell_signals']['rsi_overbought']['threshold']
            
            # RSI超卖判断
            if rsi < rsi_oversold:
                score['buy'] += self.analysis_rules['buy_signals']['rsi_oversold']['weight']
                score['signals'].append(f"🎯 RSI超卖({rsi:.1f})，买入机会")
            
            # RSI超买判断
            elif rsi > rsi_overbought:
                score['sell'] += self.analysis_rules['sell_signals']['rsi_overbought']['weight']
                score['signals'].append(f"🚨 RSI超买({rsi:.1f})，卖出信号")
            
            # RSI中性区域
            else:
                if rsi > 50:
                    score['buy'] += 0.03
                else:
                    score['sell'] += 0.03
                    
        except KeyError:
            score['signals'].append("⚠️ RSI数据不可用")
            
        return score
    
    def analyze_bollinger_bands(self, data, df):
        """布林带量化分析"""
        score = {'buy': 0, 'sell': 0, 'signals': []}
        
        try:
            position = data['BB_position']
            close = data['close']
            bb_lower = data['BB_lower']
            bb_upper = data['BB_upper']
            
            oversold_threshold = self.analysis_rules['buy_signals']['bollinger_oversold']['threshold']
            overbought_threshold = self.analysis_rules['sell_signals']['bollinger_overbought']['threshold']
            
            # 布林带下轨支撑
            if position < oversold_threshold:
                score['buy'] += self.analysis_rules['buy_signals']['bollinger_oversold']['weight']
                score['signals'].append(f"📥 价格接近布林带下轨，超卖信号")
            
            # 布林带上轨压力
            elif position > overbought_threshold:
                score['sell'] += self.analysis_rules['sell_signals']['bollinger_overbought']['weight']
                score['signals'].append(f"📤 价格接近布林带上轨，超买信号")
            
            # 布林带突破判断
            if len(df) > 1:
                prev_data = df.iloc[-2]
                if (prev_data['close'] < prev_data['BB_lower'] and 
                    close > bb_lower):
                    score['buy'] += 0.1
                    score['signals'].append("🔄 价格从下轨反弹，看涨信号")
                    
        except KeyError as e:
            score['signals'].append(f"⚠️ 布林带数据不完整: {e}")
            
        return score
    
    def analyze_kdj(self, data):
        """KDJ指标量化分析"""
        score = {'buy': 0, 'sell': 0, 'signals': []}
        
        try:
            k = data['K']
            d = data['D']
            j = data['J']
            
            kdj_oversold = self.analysis_rules['buy_signals']['kdj_oversold']['threshold']
            kdj_overbought = self.analysis_rules['sell_signals']['kdj_overbought']['threshold']
            
            # KDJ超卖判断
            if k < kdj_oversold and d < kdj_oversold:
                score['buy'] += self.analysis_rules['buy_signals']['kdj_oversold']['weight']
                score['signals'].append(f"🎯 KDJ超卖(K:{k:.1f}, D:{d:.1f})")
            
            # KDJ超买判断
            elif k > kdj_overbought and d > kdj_overbought:
                score['sell'] += self.analysis_rules['sell_signals']['kdj_overbought']['weight']
                score['signals'].append(f"🚨 KDJ超买(K:{k:.1f}, D:{d:.1f})")
            
            # KDJ金叉死叉判断
            if 'K_prev' in data and 'D_prev' in data:
                if k > d and data['K_prev'] <= data['D_prev']:
                    score['buy'] += 0.05
                    score['signals'].append("↗️ KDJ金叉形成")
                elif k < d and data['K_prev'] >= data['D_prev']:
                    score['sell'] += 0.05
                    score['signals'].append("↘️ KDJ死叉形成")
                    
        except KeyError:
            score['signals'].append("⚠️ KDJ数据不可用")
            
        return score
    
    def analyze_volume(self, data, df):
        """成交量量化分析"""
        score = {'buy': 0, 'sell': 0, 'signals': []}
        
        try:
            volume = data['vol']
            
            if len(df) > 5:
                # 计算成交量均线
                volume_ma = df['vol'].tail(5).mean()
                volume_ratio = volume / volume_ma
                
                volume_surge_threshold = self.analysis_rules['buy_signals']['volume_surge']['threshold']
                volume_decline_threshold = self.analysis_rules['sell_signals']['volume_decline']['threshold']
                
                # 成交量放大
                if volume_ratio > volume_surge_threshold:
                    if data['close'] > data['open']:  # 放量上涨
                        score['buy'] += self.analysis_rules['buy_signals']['volume_surge']['weight']
                        score['signals'].append(f"📊 放量上涨(量比:{volume_ratio:.2f})")
                    else:  # 放量下跌
                        score['sell'] += 0.1
                        score['signals'].append(f"📊 放量下跌(量比:{volume_ratio:.2f})")
                
                # 成交量萎缩
                elif volume_ratio < volume_decline_threshold:
                    score['sell'] += self.analysis_rules['sell_signals']['volume_decline']['weight']
                    score['signals'].append(f"📉 成交量萎缩(量比:{volume_ratio:.2f})")
                    
        except KeyError:
            score['signals'].append("⚠️ 成交量数据异常")
            
        return score
    
    def analyze_price_action(self, df):
        """价格走势量化分析"""
        score = {'buy': 0, 'sell': 0, 'signals': [], 'warnings': []}
        
        try:
            if len(df) < 10:
                return score
                
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # 支撑阻力判断
            support_level = df['low'].tail(20).min()
            resistance_level = df['high'].tail(20).max()
            
            support_threshold = self.analysis_rules['buy_signals']['price_support']['threshold']
            resistance_threshold = self.analysis_rules['sell_signals']['price_resistance']['threshold']
            
            # 接近支撑位
            if abs(current['close'] - support_level) / support_level < support_threshold:
                score['buy'] += self.analysis_rules['buy_signals']['price_support']['weight']
                score['signals'].append(f"🛡️ 价格接近支撑位: {support_level:.2f}")
            
            # 接近阻力位
            if abs(current['close'] - resistance_level) / resistance_level < resistance_threshold:
                score['sell'] += self.analysis_rules['sell_signals']['price_resistance']['weight']
                score['signals'].append(f"⛰️ 价格接近阻力位: {resistance_level:.2f}")
            
            # 趋势判断
            short_ma = df['close'].tail(5).mean()
            long_ma = df['close'].tail(20).mean()
            
            if short_ma > long_ma:
                score['buy'] += 0.1
                score['signals'].append("📈 短期均线上穿长期均线，趋势向上")
            else:
                score['sell'] += 0.1
                score['signals'].append("📉 短期均线下穿长期均线，趋势向下")
            
            # 波动率警告
            volatility = df['close'].tail(20).std() / df['close'].tail(20).mean()
            if volatility > 0.03:
                score['warnings'].append(f"⚠️ 高波动率警告: {volatility:.2%}")
                
        except Exception as e:
            score['warnings'].append(f"⚠️ 价格分析异常: {str(e)}")
            
        return score
    
    def generate_trading_recommendation(self, scores, ml_confidence=0.5):
        """生成交易建议"""
        buy_score = scores['buy_score']
        sell_score = scores['sell_score']
        
        # 加入机器学习置信度
        if ml_confidence > 0.5:
            buy_score += (ml_confidence - 0.5) * 2
        else:
            sell_score += (0.5 - ml_confidence) * 2
        
        net_score = buy_score - sell_score
        
        # 根据净得分生成建议
        if net_score > 0.6:
            recommendation = "🚀 强烈买入"
            confidence = "高"
        elif net_score > 0.3:
            recommendation = "✅ 建议买入"
            confidence = "中"
        elif net_score > 0.1:
            recommendation = "🤔 谨慎买入"
            confidence = "低"
        elif net_score > -0.1:
            recommendation = "⚖️ 持有观望"
            confidence = "中性"
        elif net_score > -0.3:
            recommendation = "🧐 谨慎卖出"
            confidence = "低"
        elif net_score > -0.6:
            recommendation = "❌ 建议卖出"
            confidence = "中"
        else:
            recommendation = "🔥 强烈卖出"
            confidence = "高"
        
        analysis_report = {
            'recommendation': recommendation,
            'confidence': confidence,
            'net_score': net_score,
            'buy_score': buy_score,
            'sell_score': sell_score,
            'signals': scores['signals'],
            'warnings': scores['warnings'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return analysis_report

class AdvancedStockAnalyzer:
    def __init__(self, token):
        self.token = token
        ts.set_token(token)
        self.pro = ts.pro_api()
        
    def get_stock_data(self, ts_code, start_date, end_date):
        """获取股票数据"""
        try:
            # 获取日线数据
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
        return df[['MACD', 'MACD_signal', 'MACD_hist']]
    
    def calculate_rsi(self, df, period=14):
        """计算RSI指标"""
        df = df.copy()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df['RSI']
    
    def calculate_bollinger_bands(self, df, period=20, std=2):
        """计算布林带"""
        df = df.copy()
        df['BB_middle'] = df['close'].rolling(window=period).mean()
        bb_std = df['close'].rolling(window=period).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * std)
        df['BB_lower'] = df['BB_middle'] - (bb_std * std)
        df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        return df[['BB_upper', 'BB_middle', 'BB_lower', 'BB_position']]
    
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
        
        return df[['K', 'D', 'J', 'K_prev', 'D_prev']]
    
    def calculate_bias(self, df, period=6):
        """计算乖离率BIAS"""
        df = df.copy()
        ma = df['close'].rolling(window=period).mean()
        df['BIAS'] = (df['close'] - ma) / ma * 100
        return df['BIAS']
    
    def calculate_cci(self, df, period=14):
        """计算CCI指标"""
        df = df.copy()
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        df['CCI'] = (tp - sma) / (0.015 * mad)
        return df['CCI']
    
    def calculate_obv(self, df):
        """计算OBV指标"""
        df = df.copy()
        df['OBV'] = 0
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                df['OBV'].iloc[i] = df['OBV'].iloc[i-1] + df['vol'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                df['OBV'].iloc[i] = df['OBV'].iloc[i-1] - df['vol'].iloc[i]
            else:
                df['OBV'].iloc[i] = df['OBV'].iloc[i-1]
        return df['OBV']
    
    def calculate_all_indicators(self, df):
        """计算所有技术指标"""
        indicators_df = df.copy()
        
        # 计算各个指标
        macd_data = self.calculate_macd(indicators_df)
        indicators_df = pd.concat([indicators_df, macd_data], axis=1)
        
        indicators_df['RSI'] = self.calculate_rsi(indicators_df)
        
        bb_data = self.calculate_bollinger_bands(indicators_df)
        indicators_df = pd.concat([indicators_df, bb_data], axis=1)
        
        kdj_data = self.calculate_kdj(indicators_df)
        indicators_df = pd.concat([indicators_df, kdj_data], axis=1)
        
        indicators_df['BIAS'] = self.calculate_bias(indicators_df)
        indicators_df['CCI'] = self.calculate_cci(indicators_df)
        indicators_df['OBV'] = self.calculate_obv(indicators_df)
        
        # 计算价格变化特征
        indicators_df['price_change'] = indicators_df['close'].pct_change()
        indicators_df['volume_change'] = indicators_df['vol'].pct_change()
        
        return indicators_df.dropna()
    
    def create_ml_features(self, df):
        """创建机器学习特征"""
        feature_df = df.copy()
        
        # 添加滞后特征
        for lag in [1, 2, 3, 5]:
            feature_df[f'close_lag_{lag}'] = feature_df['close'].shift(lag)
            feature_df[f'volume_lag_{lag}'] = feature_df['vol'].shift(lag)
        
        # 添加滚动统计特征
        feature_df['close_ma_5'] = feature_df['close'].rolling(5).mean()
        feature_df['close_ma_10'] = feature_df['close'].rolling(10).mean()
        feature_df['volume_ma_5'] = feature_df['vol'].rolling(5).mean()
        
        # 目标变量：未来5天是否上涨
        feature_df['target'] = (feature_df['close'].shift(-5) > feature_df['close']).astype(int)
        
        return feature_df.dropna()

def display_intelligent_analysis(df_with_indicators, ml_confidence=0.5):
    """显示智能分析结果"""
    st.subheader("🤖 智能量化分析")
    
    # 检查是否有优化权重
    optimized_weights = st.session_state.get('optimized_weights', None)
    
    if optimized_weights:
        st.info("🎯 使用优化权重进行分析")
        
        # 创建特征数据
        optimizer = DynamicWeightOptimizer()
        features = optimizer.prepare_features_for_optimization(df_with_indicators)
        
        # 使用优化权重计算得分
        signal_score = optimizer.calculate_signal_score(features, optimized_weights)
        
        # 生成分析报告
        current_score = signal_score.iloc[-1] if len(signal_score) > 0 else 0
        
        # 根据优化结果调整阈值
        buy_threshold = 0.3
        sell_threshold = -0.3
        
        if current_score > buy_threshold:
            recommendation = "🚀 优化权重强烈买入"
            confidence = "高"
        elif current_score > 0.1:
            recommendation = "✅ 优化权重建议买入"
            confidence = "中"
        elif current_score < sell_threshold:
            recommendation = "🔥 优化权重强烈卖出"
            confidence = "高"
        elif current_score < -0.1:
            recommendation = "❌ 优化权重建议卖出"
            confidence = "中"
        else:
            recommendation = "⚖️ 优化权重持有观望"
            confidence = "中性"
        
        analysis_report = {
            'recommendation': recommendation,
            'confidence': confidence,
            'net_score': current_score,
            'buy_score': max(0, current_score),
            'sell_score': max(0, -current_score),
            'signals': [f"优化权重得分: {current_score:.3f}"],
            'warnings': ["基于历史数据优化的权重，请注意市场变化风险"],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'using_optimized_weights': True
        }
        
    else:
        # 使用默认分析
        intelligent_analyzer = IntelligentAnalyzer()
        scores = intelligent_analyzer.calculate_technical_score(df_with_indicators)
        analysis_report = intelligent_analyzer.generate_trading_recommendation(scores, ml_confidence)
        analysis_report['using_optimized_weights'] = False
    
    # 显示分析结果
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("综合推荐", analysis_report['recommendation'])
    
    with col2:
        st.metric("置信度", analysis_report['confidence'])
    
    with col3:
        st.metric("净得分", f"{analysis_report['net_score']:.3f}")
    
    # 显示详细得分
    st.write("#### 📊 详细得分分析")
    score_col1, score_col2 = st.columns(2)
    
    with score_col1:
        st.progress(min(int(analysis_report['buy_score'] * 100), 100), 
                   text=f"买入得分: {analysis_report['buy_score']:.3f}")
    
    with score_col2:
        st.progress(min(int(analysis_report['sell_score'] * 100), 100),
                   text=f"卖出得分: {analysis_report['sell_score']:.3f}")
    
    # 显示信号列表
    st.write("#### 📈 技术信号")
    
    if analysis_report['signals']:
        for signal in analysis_report['signals']:
            st.write(signal)
    else:
        st.write("暂无明确技术信号")
    
    # 显示警告信息
    if analysis_report['warnings']:
        st.write("#### ⚠️ 风险提示")
        for warning in analysis_report['warnings']:
            st.error(warning)
    
    # 显示分析规则或优化信息
    if analysis_report.get('using_optimized_weights', False):
        with st.expander("📋 查看优化权重信息"):
            st.write("当前使用基于历史数据优化的动态权重")
            if 'optimized_weights' in st.session_state:
                st.json({k: round(v, 3) for k, v in st.session_state.optimized_weights.items() if abs(v) > 0.1})
    else:
        with st.expander("📋 查看分析规则"):
            intelligent_analyzer = IntelligentAnalyzer()
            st.json(intelligent_analyzer.analysis_rules)
    
    # 显示时间戳
    st.caption(f"分析时间: {analysis_report['timestamp']}")
    
    return analysis_report

def display_weight_optimization(df_with_indicators):
    """显示权重优化界面"""
    st.subheader("🎯 动态权重优化")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        optimization_days = st.slider("优化周期(天)", 30, 180, 90, 
                                    help="用于权重优化的历史数据天数")
    
    with col2:
        hold_days = st.slider("持有天数", 1, 10, 5, 
                            help="预测未来几天的价格走势")
    
    with col3:
        population_size = st.slider("种群大小", 20, 100, 30,
                                  help="遗传算法种群大小")
    
    if st.button("开始权重优化", type="primary"):
        with st.spinner("正在进行权重优化，这可能需要几分钟..."):
            optimizer = DynamicWeightOptimizer()
            results = optimizer.backtest_optimization(
                df_with_indicators, optimization_days, hold_days
            )
        
        if not results:
            st.error("优化失败，请检查数据质量")
            return
        
        # 显示优化结果
        st.write("### 📊 优化结果统计")
        
        # 计算平均准确率
        train_accuracies = [r['train_accuracy'] for r in results]
        test_accuracies = [r['test_accuracy'] for r in results]
        buy_accuracies = [r['buy_accuracy'] for r in results if r['buy_accuracy'] > 0]
        sell_accuracies = [r['sell_accuracy'] for r in results if r['sell_accuracy'] > 0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("训练集平均准确率", f"{np.mean(train_accuracies):.2%}")
        
        with col2:
            st.metric("测试集平均准确率", f"{np.mean(test_accuracies):.2%}")
        
        with col3:
            if buy_accuracies:
                st.metric("买入信号准确率", f"{np.mean(buy_accuracies):.2%}")
            else:
                st.metric("买入信号准确率", "N/A")
        
        with col4:
            if sell_accuracies:
                st.metric("卖出信号准确率", f"{np.mean(sell_accuracies):.2%}")
            else:
                st.metric("卖出信号准确率", "N/A")
        
        # 显示权重分布
        st.write("### 🔧 最优权重配置")
        
        # 合并所有权重
        all_weights = {}
        for result in results:
            for feature, weight in result['weights'].items():
                if feature not in all_weights:
                    all_weights[feature] = []
                all_weights[feature].append(weight)
        
        # 计算平均权重
        avg_weights = {feature: np.mean(weights) for feature, weights in all_weights.items()}
        
        # 显示最重要的权重
        sorted_weights = dict(sorted(avg_weights.items(), key=lambda x: abs(x[1]), reverse=True))
        
        # 创建权重可视化
        fig = go.Figure()
        
        positive_features = {k: v for k, v in sorted_weights.items() if v > 0}
        negative_features = {k: v for k, v in sorted_weights.items() if v < 0}
        
        if positive_features:
            fig.add_trace(go.Bar(
                x=list(positive_features.values())[:10],
                y=list(positive_features.keys())[:10],
                orientation='h',
                name='正向信号',
                marker_color='green'
            ))
        
        if negative_features:
            fig.add_trace(go.Bar(
                x=list(negative_features.values())[:10],
                y=list(negative_features.keys())[:10],
                orientation='h',
                name='负向信号',
                marker_color='red'
            ))
        
        fig.update_layout(
            title="Top 10 最重要的技术指标权重",
            xaxis_title="权重值",
            yaxis_title="技术指标",
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示权重解释
        interpretation = optimizer.interpret_optimized_weights(avg_weights)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### ✅ 强烈买入信号")
            for signal in interpretation['strong_buy_signals']:
                weight = avg_weights[signal]
                st.write(f"**{signal}**: {weight:.3f}")
        
        with col2:
            st.write("#### ❌ 强烈卖出信号")
            for signal in interpretation['strong_sell_signals']:
                weight = avg_weights[signal]
                st.write(f"**{signal}**: {weight:.3f}")
        
        # 显示详细的回测结果
        st.write("### 📈 详细回测结果")
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df.drop('weights', axis=1).round(4))
        
        # 保存优化结果
        st.session_state.optimized_weights = avg_weights
        st.session_state.optimization_results = results
        
        st.success("权重优化完成！优化后的权重已保存，可以在智能分析模块中使用。")
    
    # 显示如何使用优化权重的说明
    with st.expander("💡 如何使用优化权重"):
        st.markdown("""
        **优化权重的使用流程：**
        
        1. **运行优化**：点击"开始权重优化"按钮，系统会自动寻找最优的指标权重配比
        2. **查看结果**：分析优化结果，了解哪些指标对当前股票最有效
        3. **应用权重**：优化后的权重会自动保存，在智能分析模块中使用
        4. **持续优化**：建议定期重新优化权重，适应市场变化
        
        **优化原理：**
        - 使用遗传算法在历史数据上寻找最优权重
        - 考虑不同持有期的表现
        - 交叉验证确保权重稳定性
        - 自动识别当前市场环境下最有效的技术指标
        """)

def display_price_charts(df, stock_name):
    """显示价格走势图表"""
    st.subheader("价格走势与技术指标")
    
    # 使用plotly创建交互式图表
    fig = make_subplots(rows=4, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.05,
                       subplot_titles=('K线图与布林带', '成交量', 'MACD', 'RSI'),
                       row_heights=[0.4, 0.15, 0.2, 0.25])
    
    # K线图
    fig.add_trace(go.Candlestick(x=df.index,
                                open=df['open'],
                                high=df['high'],
                                low=df['low'],
                                close=df['close'],
                                name='K线'), row=1, col=1)
    
    # 布林带
    if 'BB_upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], 
                               line=dict(color='red', width=1), name='布林带上轨'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_middle'], 
                               line=dict(color='blue', width=1), name='布林带中轨'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], 
                               line=dict(color='green', width=1), name='布林带下轨'), row=1, col=1)
    
    # 成交量
    colors = ['red' if row['close'] >= row['open'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['vol'], 
                        name='成交量', marker_color=colors), row=2, col=1)
    
    # MACD
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], 
                               line=dict(color='blue', width=1), name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], 
                               line=dict(color='red', width=1), name='信号线'), row=3, col=1)
        
        # MACD柱状图
        colors_macd = ['green' if x >= 0 else 'red' for x in df['MACD_hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], 
                           name='MACD柱', marker_color=colors_macd), row=3, col=1)
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], 
                               line=dict(color='purple', width=1), name='RSI'), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=[70]*len(df), 
                               line=dict(color='red', dash='dash'), name='超买线'), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=[30]*len(df), 
                               line=dict(color='green', dash='dash'), name='超卖线'), row=4, col=1)
    
    fig.update_layout(height=800, showlegend=True, 
                     xaxis_rangeslider_visible=False)
    
    st.plotly_chart(fig, use_container_width=True)

def display_technical_indicators(df):
    """显示技术指标分析"""
    st.subheader("技术指标详细分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # MACD信号分析
        st.write("**MACD信号分析**")
        latest_macd = df['MACD'].iloc[-1]
        latest_signal = df['MACD_signal'].iloc[-1]
        macd_hist = df['MACD_hist'].iloc[-1]
        
        if latest_macd > latest_signal and macd_hist > 0:
            st.success("MACD金叉，看涨信号")
        elif latest_macd < latest_signal and macd_hist < 0:
            st.error("MACD死叉，看跌信号")
        else:
            st.info("MACD中性")
        
        # RSI分析
        st.write("**RSI分析**")
        latest_rsi = df['RSI'].iloc[-1]
        if latest_rsi > 70:
            st.error(f"RSI: {latest_rsi:.2f} - 超买区域")
        elif latest_rsi < 30:
            st.success(f"RSI: {latest_rsi:.2f} - 超卖区域")
        else:
            st.info(f"RSI: {latest_rsi:.2f} - 正常区域")
    
    with col2:
        # 布林带分析
        st.write("**布林带分析**")
        latest_close = df['close'].iloc[-1]
        latest_bb_upper = df['BB_upper'].iloc[-1]
        latest_bb_lower = df['BB_lower'].iloc[-1]
        
        if latest_close > latest_bb_upper:
            st.error("价格突破布林带上轨，可能超买")
        elif latest_close < latest_bb_lower:
            st.success("价格突破布林带下轨，可能超卖")
        else:
            st.info("价格在布林带内运行")
        
        # KDJ分析
        st.write("**KDJ分析**")
        latest_k = df['K'].iloc[-1]
        latest_d = df['D'].iloc[-1]
        latest_j = df['J'].iloc[-1]
        
        if latest_k > 80 or latest_d > 80:
            st.error("KDJ超买")
        elif latest_k < 20 or latest_d < 20:
            st.success("KDJ超卖")
        else:
            st.info("KDJ中性")

def display_correlation_analysis(df):
    """显示相关性分析"""
    st.subheader("指标相关性分析")
    
    # 选择数值型列进行相关性分析
    numeric_cols = ['open', 'high', 'low', 'close', 'vol', 'MACD', 'RSI', 'K', 'D', 'J', 'BIAS', 'CCI']
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) > 1:
        correlation_matrix = df[available_cols].corr()
        
        # 绘制热力图
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, ax=ax)
        ax.set_title('技术指标相关性热力图')
        st.pyplot(fig)
        
        # 显示与收盘价的相关性
        st.write("**各指标与收盘价的相关系数:**")
        close_corr = correlation_matrix['close'].sort_values(ascending=False)
        for indicator, corr_value in close_corr.items():
            if indicator != 'close':
                st.write(f"{indicator}: {corr_value:.4f}")

def display_ml_analysis(analyzer, df):
    """显示机器学习分析"""
    st.subheader("机器学习预测分析")
    
    # 创建特征数据集
    feature_df = analyzer.create_ml_features(df)
    
    # 准备特征
    feature_cols = [col for col in feature_df.columns if col not in 
                   ['target', 'open', 'high', 'low', 'close', 'vol'] and 
                   not col.startswith('BB_') and feature_df[col].dtype in ['float64', 'int64']]
    
    if len(feature_cols) < 3:
        st.warning("特征数量不足，无法进行机器学习分析")
        return 0.5
    
    X = feature_df[feature_cols]
    y = feature_df['target']
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, shuffle=False
    )
    
    # 训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    st.write(f"**模型准确率:** {accuracy:.4f}")
    
    # 计算最新数据的预测概率
    latest_features = X_scaled[-1:].reshape(1, -1)
    prediction_proba = model.predict_proba(latest_features)[0]
    bullish_probability = prediction_proba[1]  # 看涨概率
    
    st.write(f"**当前看涨概率:** {bullish_probability:.4f}")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    st.write("**特征重要性排名:**")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature', ax=ax)
    ax.set_title('Top 10 重要特征')
    st.pyplot(fig)
    
    return bullish_probability

def display_data_overview(df):
    """显示数据概览"""
    st.subheader("数据概览")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**基本统计信息:**")
        st.dataframe(df[['open', 'high', 'low', 'close', 'vol']].describe())
    
    with col2:
        st.write("**最新指标值:**")
        latest_data = {
            '指标': ['收盘价', '成交量', 'MACD', 'RSI', 'K', 'D', 'BIAS', 'CCI'],
            '数值': [
                f"{df['close'].iloc[-1]:.2f}",
                f"{df['vol'].iloc[-1]:.0f}",
                f"{df['MACD'].iloc[-1]:.4f}",
                f"{df['RSI'].iloc[-1]:.2f}",
                f"{df['K'].iloc[-1]:.2f}",
                f"{df['D'].iloc[-1]:.2f}",
                f"{df['BIAS'].iloc[-1]:.2f}%",
                f"{df['CCI'].iloc[-1]:.2f}"
            ]
        }
        latest_df = pd.DataFrame(latest_data)
        st.dataframe(latest_df)
    
    # 显示原始数据
    st.write("**原始数据 (最近20个交易日):**")
    st.dataframe(df.tail(20))

def main():
    st.title("📈 股票技术分析平台")
    st.markdown("基于Tushare数据的多维度股票技术分析工具")
    
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
    analyzer = AdvancedStockAnalyzer(token)
    
    # 股票代码输入
    col1, col2 = st.sidebar.columns(2)
    with col1:
        ts_code = st.text_input("股票代码", "000001.SZ", 
                               help="格式：代码.交易所，如000001.SZ, 600000.SH")
    with col2:
        stock_name = st.text_input("股票名称(可选)", "平安银行")
    
    # 日期范围选择
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
    
    col3, col4 = st.sidebar.columns(2)
    with col3:
        start_date_input = st.text_input("开始日期", start_date)
    with col4:
        end_date_input = st.text_input("结束日期", end_date)
    
    # 获取数据
    if st.sidebar.button("开始分析"):
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
                st.subheader(f"{stock_name} ({ts_code}) 技术分析")
                
                # 创建标签页 - 现在包含7个标签
                tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                    "价格走势", "技术指标", "相关性分析", "机器学习预测", "数据概览", "🤖智能分析", "🎯权重优化"
                ])
                
                # 首先运行机器学习分析获取预测概率
                with tab4:
                    ml_confidence = display_ml_analysis(analyzer, df_with_indicators)
                
                with tab1:
                    display_price_charts(df_with_indicators, stock_name)
                
                with tab2:
                    display_technical_indicators(df_with_indicators)
                
                with tab3:
                    display_correlation_analysis(df_with_indicators)
                
                with tab5:
                    display_data_overview(df_with_indicators)
                
                # 智能分析标签页 - 使用机器学习置信度
                with tab6:
                    analysis_report = display_intelligent_analysis(df_with_indicators, ml_confidence)
                
                # 权重优化标签页
                with tab7:
                    display_weight_optimization(df_with_indicators)
                    
            except Exception as e:
                st.error(f"分析过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()