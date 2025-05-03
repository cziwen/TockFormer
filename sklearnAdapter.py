from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import torch
from torch.utils.data import TensorDataset

from Util import safe_inverse_transform

from LSTMModel import TimeSeriesLSTM

class SklearnAdapter(BaseEstimator, RegressorMixin):
    """
    一个通用 Adapter，适配任何实现了 train_model 和 predict_model 的模型实例。
    """
    _estimator_type = "regressor"

    
    def __init__(self,
                 model=TimeSeriesLSTM(input_dim=49, hidden_dim=64, num_layers=2, dropout=0.1, seq_length=100, output_dim=4), # 占位符
                 val_dataset=None,              # 模型实例，比如 TimeSeriesTransformer(...)
                 scaler=None,
                 bias_corrector=None,
                 target_indices=None,
                 num_epochs=100,
                 batch_size=32,
                 learning_rate=1e-4,
                 patience=10,
                 min_delta=1e-5,
                 batch_shuffle_threshold=50):

        super().__init__()  # 这一句非常关键！


        self.model = model
        self.scaler = scaler
        self.val_dataset = val_dataset # 验证集，tensordataset
        self.bias_corrector = bias_corrector
        self.target_indices = target_indices
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.min_delta = min_delta
        self.batch_shuffle_threshold = batch_shuffle_threshold
        

    def fit(self, X, y):
        # 把 numpy 转成 TensorDataset
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        ds = TensorDataset(X_t, y_t)

        # 调用你模型里的 train_model
        # 假设你的 model 在 __init__ 时就已经拿到了所有超参
        self.model.train_model(
            train_dataset=ds,
            val_dataset=self.val_dataset,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            scaler=self.scaler,
            target_indices=self.target_indices,
            patience=self.patience,
            min_delta=self.min_delta,
            batch_shuffle_threshold=self.batch_shuffle_threshold,
            log=True
        )
        return self

    def predict(self, X):
        preds = []
        for i in range(X.shape[0]):
            x_i = torch.tensor(X[i:i+1], dtype=torch.float32)
            p = self.model.predict_model(
                X_tensor=x_i,
                scaler=self.scaler,
                bias_corrector=self.bias_corrector,
                target_indices=self.target_indices
            ).reshape(-1)
            preds.append(p)
        return np.vstack(preds)

    def score(self, X, y):
        """
        计算模型的 R2 分数
        参数：
            - X: 输入特征，形状为 (n_samples, n_features)
            - y: 真实标签，形状为 (n_samples, n_outputs)
        """
        from sklearn.metrics import r2_score
        
        return r2_score(y, self.predict(X))