import torch
import torch.nn as nn
import math
import numpy as np
import copy
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, ParameterGrid

from Util import safe_inverse_transform

class TimeSeriesLSTM(nn.Module):
    """
    基于 LSTM 的时间序列预测模型
    """

    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1, seq_length=100, output_dim=1, bidirectional=False):
        """
        参数：
        - input_dim: 输入特征的维度
        - hidden_dim: LSTM 隐层维度
        - num_layers: LSTM 层数
        - dropout: dropout 概率
        - seq_length: 序列最大长度（用于可能的序列处理）
        - output_dim: 模型输出的维度
        - bidirectional: 是否双向 LSTM
        """
        super(TimeSeriesLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional

        # 输入映射到隐藏维度
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers>1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        # 输出映射层，使用最后时间步
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_regressor = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        """
        前向传播
        参数:
          - x: [batch_size, seq_length, input_dim]
        返回:
          - 输出形状为 [batch_size, output_dim]
        """
        # 输入线性变换
        x = self.input_linear(x) * math.sqrt(self.hidden_dim)
        # LSTM 计算
        outputs, (h_n, c_n) = self.lstm(x)
        # 取最后一个时间步（包括双向）
        if self.bidirectional:
            # 拼接前向和后向最后层隐藏状态
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_last = h_n[-1]
        out = self.output_regressor(h_last)
        return out

    # def safe_inverse_transform(self, preds, scaler, target_indices):
    #     """
    #     仅对目标列进行逆缩放
    #     """
    #     preds_inv = preds.copy()
    #     for i, idx in enumerate(target_indices):
    #         preds_inv[:, i] = preds[:, i] * scaler.data_range_[idx] + scaler.data_min_[idx]
    #     return preds_inv

    def evaluate_model(self, dataset, batch_size=32, scaler=None, target_indices=None, bias_corrector=None):
        """
        验证集评估，返回 MSE、R2 及逆缩放结果
        """
        self.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        device = next(self.parameters()).device
        preds, targets = [], []
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = self(x_batch)
                preds.append(out.cpu().numpy())
                targets.append(y_batch.cpu().numpy())
        preds = np.concatenate(preds, axis=0).astype(np.float32)
        targets = np.concatenate(targets, axis=0).astype(np.float32)
        if scaler is not None and target_indices is not None:
            preds = safe_inverse_transform(preds, scaler, target_indices)
            targets = safe_inverse_transform(targets, scaler, target_indices)
        if bias_corrector is not None:
            preds = bias_corrector.transform(preds)
        mse_list = [mean_squared_error(targets[:, i], preds[:, i]) for i in range(preds.shape[1])]
        r2_list = [r2_score(targets[:, i], preds[:, i]) for i in range(preds.shape[1])]
        return mse_list, r2_list, preds, targets

    def train_model(self, train_dataset, val_dataset=None, num_epochs=50, batch_size=32,
                     learning_rate=1e-4, scaler=None, target_indices=None,
                     patience=10, min_delta=1e-5, batch_shuffle_threshold=50, log=False):
        """
        训练模型，早停与内部重新打乱逻辑
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        best_val_mse = float('inf')
        epochs_no_improve = 0
        best_state = None
        train_losses, val_mses, val_r2s = [], [], []

        for epoch in range(num_epochs):
            self.train()
            total_loss, best_batch_loss, bad_count = 0.0, float('inf'), 0
            loader_iter = iter(train_loader)
            num_batches = len(train_loader)
            for batch_idx in range(num_batches):
                try:
                    x_batch, y_batch = next(loader_iter)
                except StopIteration:
                    break
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                out = self(x_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
                batch_loss = loss.item()
                total_loss += batch_loss
                if batch_loss < best_batch_loss - min_delta:
                    best_batch_loss = batch_loss; bad_count = 0
                else:
                    bad_count += 1
                if bad_count >= batch_shuffle_threshold:
                    if log:
                        print(f"Epoch {epoch+1}: reshuffling after {batch_idx+1} bad batches")
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    loader_iter = iter(train_loader)
                    num_batches = len(train_loader)
                    bad_count = 0
            avg_loss = total_loss/num_batches
            train_losses.append(avg_loss)
            if log:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

            if val_dataset is not None:
                mse_list, r2_list, _, _ = self.evaluate_model(val_dataset, batch_size, scaler, target_indices)
                mean_mse = np.mean(mse_list)
                val_mses.append(mse_list)
                val_r2s.append(r2_list)
                
                if log:
                    print (f"Epoch {epoch + 1}/{num_epochs}, Val MSEs: {mse_list}, R²: {r2_list}")

                if mean_mse + min_delta < best_val_mse:
                    best_val_mse = mean_mse; epochs_no_improve = 0; best_state = copy.deepcopy(self.state_dict())
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        if log:
                            print(f"Early stopping at epoch {epoch+1}")
                        self.load_state_dict(best_state)
                        break
        return train_losses, val_mses, val_r2s

    def predict_model(self, X_tensor, scaler=None, bias_corrector=None, target_indices=None):
        """
        单样本预测与逆缩放
        """
        self.eval()
        device = next(self.parameters()).device
        X_tensor = X_tensor.to(device)
        with torch.no_grad():
            pred = self(X_tensor).cpu().numpy().astype(np.float32)
        if scaler is not None and target_indices is not None:
            pred = safe_inverse_transform(pred, scaler, target_indices)
        if bias_corrector is not None:
            pred = bias_corrector.transform(pred)
        return pred

    def cross_validate(self, dataset, k=5, num_epochs=50, batch_size=32,
                       learning_rate=1e-4, scaler=None, target_indices=None,
                       patience=5, min_delta=1e-5, batch_shuffle_threshold=50):
        """
        k 折交叉验证，返回每折指标
        """
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        results = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            train_sub = Subset(dataset, train_idx)
            val_sub = Subset(dataset, val_idx)
            model = copy.deepcopy(self)
            model.train_model(train_sub, val_sub, num_epochs, batch_size,
                              learning_rate, scaler, target_indices,
                              patience, min_delta, batch_shuffle_threshold)
            mse_list, r2_list, _, _ = model.evaluate_model(val_sub, batch_size, scaler, target_indices)
            results.append({'mse': mse_list, 'r2': r2_list})
        return results

