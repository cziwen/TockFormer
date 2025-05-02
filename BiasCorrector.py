from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

import numpy as np

class BiasCorrector:
    def __init__(self, mode='linear', scale='standard'):
        """
        初始化 BiasCorrector

        参数:
        - mode: 'linear', 'mlp', 或 'mean'
        - scale: 'standard' 使用 StandardScaler, 'minmax' 使用 MinMaxScaler, 或 None 不使用缩放
        """
        self.mode = mode
        self.scale = scale
        self.model = None
        self.bias = None
        self.scaler_x = None
        self.scaler_y = None

    def fit(self, preds, targets):
        """
        拟合偏移模型，只允许在训练集或验证集上使用
        """

        # === 缩放处理（可选）===
        if self.scale == 'standard':
            self.scaler_x = StandardScaler()
            self.scaler_y = StandardScaler()
        elif self.scale == 'minmax':
            self.scaler_x = MinMaxScaler()
            self.scaler_y = MinMaxScaler()
        else:
            self.scaler_x = None
            self.scaler_y = None

        if self.scaler_x:
            preds_scaled = self.scaler_x.fit_transform(preds)
            targets_scaled = self.scaler_y.fit_transform(targets)
        else:
            preds_scaled = preds
            targets_scaled = targets

        # === 模型拟合 ===
        if self.mode == 'linear':
            self.model = LinearRegression()
            self.model.fit(preds_scaled, targets_scaled)

        elif self.mode == 'mlp':
            self.model = MLPRegressor(
                hidden_layer_sizes=(32, 16),
                solver='adam',
                validation_fraction=0.1,
                max_iter=5000,
                early_stopping=True)
            self.model.fit(preds_scaled, targets_scaled)

        elif self.mode == 'mean':
            self.bias = np.mean(targets - preds, axis=0)

        else:
            raise ValueError(f"不支持的模式: {self.mode}")

    def transform(self, preds):
        """
        应用于预测值，返回偏移校正后的预测值
        """
        if self.mode in ['linear', 'mlp']:
            if self.model is None:
                raise RuntimeError("BiasCorrector 尚未拟合，请先调用 fit()")

            # 标准化
            if self.scaler_x:
                print("used scaler")
                preds_scaled = self.scaler_x.transform(preds)
                corrected_scaled = self.model.predict(preds_scaled)
                corrected = self.scaler_y.inverse_transform(corrected_scaled)
            else:
                print("no scaler")
                corrected = self.model.predict(preds)
            return corrected

        elif self.mode == 'mean':
            if self.bias is None:
                raise RuntimeError("BiasCorrector 尚未拟合，请先调用 fit()")
            return preds + self.bias