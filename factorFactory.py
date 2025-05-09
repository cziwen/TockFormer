import pandas as pd
import ta

class FactorFactory:
    def __init__(self):
        # ...（保留你之前的参数列表配置）...
        self.rsi_windows = [6, 10, 14, 20, 30]
        self.sma_windows = [5, 10, 20, 30, 50, 100, 200]
        self.ema_spans   = [5, 10, 12, 20, 30, 50, 100, 200]
        self.atr_windows = [14, 20, 30, 40]
        self.adx_windows = [14, 20, 30]
        self.cci_windows = [20, 30, 50]
        self.roc_periods = [5, 10, 20, 30]
        self.mom_periods = [3, 5, 10, 20]
        self.wr_periods  = [10, 14, 20, 30]
        self.macd_pairs  = [(12,26), (5,20), (8,17), (10,30)]
        self.bb_windows  = [10, 20, 30]
        self.bb_dev      = [1.5, 2.0, 2.5]
        self.price_cols  = ['open','high','low','close']

    def generate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        df_in = df.copy().sort_values('timestamp').set_index('timestamp')
        feats = {}  # 暂存所有新特征

        # 1. RSI
        for w in self.rsi_windows:
            for c in self.price_cols:
                name = f'rsi_{w}_{c}'
                feats[name] = ta.momentum.RSIIndicator(df_in[c], window=w).rsi()

        # 2. SMA & EMA
        for w in self.sma_windows:
            for c in self.price_cols:
                feats[f'sma_{w}_{c}'] = df_in[c].rolling(w).mean()
        for s in self.ema_spans:
            for c in self.price_cols:
                feats[f'ema_{s}_{c}'] = df_in[c].ewm(span=s).mean()

        # 3. MACD diff
        for fast, slow in self.macd_pairs:
            for c in self.price_cols:
                macd = ta.trend.MACD(df_in[c], window_fast=fast, window_slow=slow)
                feats[f'macd_diff_{fast}_{slow}_{c}'] = macd.macd_diff()

        # 4. Bollinger Bands
        for w in self.bb_windows:
            for dev in self.bb_dev:
                bb = ta.volatility.BollingerBands(df_in['close'], window=w, window_dev=dev)
                feats[f'bb_mavg_{w}_{dev}'] = bb.bollinger_mavg()
                feats[f'bb_hband_{w}_{dev}'] = bb.bollinger_hband()
                feats[f'bb_lband_{w}_{dev}'] = bb.bollinger_lband()

        # 5. 趋势 / 波动
        for w in self.atr_windows:
            feats[f'atr_{w}'] = ta.volatility.AverageTrueRange(
                df_in['high'], df_in['low'], df_in['close'], window=w
            ).average_true_range()
        for w in self.adx_windows:
            feats[f'adx_{w}'] = ta.trend.ADXIndicator(
                df_in['high'], df_in['low'], df_in['close'], window=w
            ).adx()
        for w in self.cci_windows:
            feats[f'cci_{w}'] = ta.trend.CCIIndicator(
                df_in['high'], df_in['low'], df_in['close'], window=w
            ).cci()

        # 6. 变动率 & 动量
        for p in self.roc_periods:
            for c in self.price_cols:
                feats[f'roc_{p}_{c}'] = ta.momentum.ROCIndicator(df_in[c], window=p).roc()
        # 用差分替代不存在的 MomentumIndicator
        for p in self.mom_periods:
            for c in self.price_cols:
                feats[f'mom_{p}_{c}'] = df_in[c] - df_in[c].shift(p)
        for p in self.wr_periods:
            feats[f'wr_{p}'] = ta.momentum.WilliamsRIndicator(
                df_in['high'], df_in['low'], df_in['close'], lbp=p
            ).williams_r()

        # 7. 成交量 / 能量潮
        feats['obv']    = ta.volume.OnBalanceVolumeIndicator(df_in['close'], df_in['volume']).on_balance_volume()
        feats['cmf_20'] = ta.volume.ChaikinMoneyFlowIndicator(
            df_in['high'], df_in['low'], df_in['close'], df_in['volume'], window=20
        ).chaikin_money_flow()
        feats['mfi_14'] = ta.volume.MFIIndicator(
            df_in['high'], df_in['low'], df_in['close'], df_in['volume'], window=14
        ).money_flow_index()
        feats['eom_14'] = ta.volume.EaseOfMovementIndicator(
            df_in['high'], df_in['low'], df_in['volume'], window=14
        ).ease_of_movement()
        feats['adl']    = ta.volume.AccDistIndexIndicator(
            df_in['high'], df_in['low'], df_in['close'], df_in['volume']
        ).acc_dist_index()

        # 8. 价差 & 偏离度
        for w in self.sma_windows:
            key_sma = f'sma_{w}_close'
            # 确保先计算了 sma
            if key_sma not in feats:
                feats[key_sma] = df_in['close'].rolling(w).mean()
            feats[f'price_div_sma_{w}'] = df_in['close'] / feats[key_sma] - 1
            feats[f'sma_diff_{w}']      = df_in['close'] - feats[key_sma]

        # 9. 多期收益率
        for p in [1, 5, 10, 20]:
            for c in self.price_cols:
                feats[f'pct_chg_{p}_{c}'] = df_in[c].pct_change(p)

        # 一次性 concat，避免 DataFrame 碎片化
        df_feats = pd.concat([df_in, pd.DataFrame(feats, index=df_in.index)], axis=1)

        # 重置索引并返回
        return df_feats.reset_index()