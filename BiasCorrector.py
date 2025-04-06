import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

class BiasCorrector:
    def __init__(self, mode='linear'):
        """
        初始化 BiasCorrector

        参数:
        - mode: 'linear' 表示用线性回归校正（ax + b），
                'mean' 表示用均值偏移校正（x + mean(target - pred)）
        """
        self.mode = mode
        self.model = None
        self.bias = None

    def fit(self, preds, targets):
        """
        拟合偏移模型，只允许在训练集或验证集上使用
        """
        if self.mode == 'linear':
            self.model = LinearRegression()
            self.model.fit(preds, targets)
        elif self.mode == 'mean':
            self.bias = np.mean(targets - preds, axis=0)
        else:
            raise ValueError(f"不支持的模式: {self.mode}")

    def transform(self, preds):
        """
        应用于预测值，返回偏移校正后的预测值
        """
        if self.mode == 'linear':
            if self.model is None:
                raise RuntimeError("BiasCorrector 尚未拟合，请先调用 fit()")
            return self.model.predict(preds)
        elif self.mode == 'mean':
            if self.bias is None:
                raise RuntimeError("BiasCorrector 尚未拟合，请先调用 fit()")
            return preds + self.bias

    def save(self, filepath):
        """
        保存 BiasCorrector 到文件（.pkl 格式）
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ BiasCorrector saved to {filepath}")

    @staticmethod
    def load(filepath):
        """
        从文件中加载 BiasCorrector（.pkl 格式）
        """
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, BiasCorrector):
            raise TypeError("文件中不是一个 BiasCorrector 对象")
        print(f"✅ BiasCorrector loaded from {filepath}")
        return obj