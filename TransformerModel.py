import torch
import torch.nn as nn
import math
import numpy as np
import shap

from src.efficient_kan import KAN
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score


class PositionalEncoding (nn.Module):
    """
    位置编码模块：为输入的序列加上位置信息
    """

    def __init__ (self, d_model, dropout=0.1, max_len=5000):
        super (PositionalEncoding, self).__init__ ()
        self.dropout = nn.Dropout (p=dropout)

        # 构造一个 [max_len, d_model] 的位置编码矩阵
        pe = torch.zeros (max_len, d_model)
        position = torch.arange (0, max_len, dtype=torch.float).unsqueeze (1)
        # 计算每个位置的编码：这里使用的是 sin 和 cos 函数
        div_term = torch.exp (torch.arange (0, d_model, 2).float () * (-math.log (10000.0) / d_model))
        pe[:, 0::2] = torch.sin (position * div_term)
        pe[:, 1::2] = torch.cos (position * div_term)
        pe = pe.unsqueeze (0)  # 变为 [1, max_len, d_model]
        self.register_buffer ('pe', pe)

    def forward (self, x):
        """
        输入 x 的形状应为 [batch_size, seq_length, d_model]
        """
        x = x + self.pe[:, :x.size (1)]
        return self.dropout (x)


class TimeSeriesTransformer (nn.Module):
    """
    基于 Transformer 编码器的时间序列模型
    """

    def __init__ (self, input_dim, model_dim, num_heads, num_layers, dropout=0.1, seq_length=100, output_dim=1):
        """
        参数说明：
        - input_dim: 输入特征的维度
        - model_dim: 模型内部的特征维度
        - num_heads: 多头自注意力中的头数
        - num_layers: Transformer 编码器层数
        - dropout: dropout 概率
        - seq_length: 序列的最大长度，用于位置编码
        - output_dim: 模型最终输出的维度（例如回归问题可以设置为 1）
        """
        super (TimeSeriesTransformer, self).__init__ ()
        self.model_dim = model_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 将原始输入映射到模型维度
        self.input_linear = nn.Linear (input_dim, model_dim)
        # self.input_norm = nn.LayerNorm (model_dim) # 可选

        # 位置编码模块
        self.positional_encoding = PositionalEncoding (model_dim, dropout=dropout, max_len=seq_length)
        # 构建 Transformer 编码器层 Encoder only
        encoder_layer = nn.TransformerEncoderLayer (d_model=model_dim, nhead=num_heads, dropout=dropout,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder (encoder_layer, num_layers=num_layers)
        # 输出层，将 Transformer 输出映射到目标维度
        # self.output_regressor = KAN ([model_dim, 64, output_dim])  # 两层kan: [model_dim -> 64]   [64 -> output_dim]
        self.output_regressor = nn.Sequential (
            nn.Linear (model_dim, 64),
            nn.ReLU (),
            nn.Linear (64, output_dim)
        )
        # self.output_norm = nn.LayerNorm (model_dim) # 可选

    def forward (self, src):
        """
        参数:
        - src: 输入序列数据，形状为 [batch_size, seq_length, input_dim]
        """
        # 线性映射并缩放
        src = self.input_linear (src) * math.sqrt (self.model_dim)
        # src = self.input_norm (src)  # 可选
        # 添加位置编码
        src = self.positional_encoding (src)
        # Transformer 要求的输入形状为 [seq_length, batch_size, model_dim]
        src = src.transpose (0, 1)
        # Transformer 编码器
        encoder_output = self.transformer_encoder (src)
        # 转换回 [batch_size, seq_length, model_dim]
        encoder_output = encoder_output.transpose (0, 1)
        # 此处采用最后一个时间步的输出作为整体序列的表示，也可以根据需求做其他聚合（如均值、加权等）
        # encoder_output = self.output_norm (encoder_output)  # 可选
        out = self.output_regressor (encoder_output[:, -1, :])
        return out

    def evaluate_model (self, dataset, batch_size=32, scaler=None, target_indices=[0, 1, 2, 3]):
        """
        在验证集上评估模型，计算每个目标特征的 MSE、R² 并返回预测值与真实值（已逆缩放）

        参数：
          - dataset: 验证数据集，每个元素为 (input, target)
          - batch_size: batch 大小
          - scaler: 拟合训练集的 scaler（用于预测值逆变换）
          - target_indices: 需要逆缩放的目标列在原始数据中的索引

        返回：
          - mse_list: 每个目标特征的 MSE 列表
          - r2_list: 每个目标特征的 R² 列表
          - preds: 全部逆缩放后的预测值 (N, T)
          - targets: 全部逆缩放后的真实值 (N, T)
        """
        self.eval ()
        val_loader = DataLoader (dataset, batch_size=batch_size, shuffle=False)
        preds = []
        targets = []

        with torch.no_grad ():
            for batch_inputs, batch_targets in val_loader:
                outputs = self (batch_inputs)
                preds.append (outputs.detach ().cpu ().numpy ())
                targets.append (batch_targets.detach ().cpu ().numpy ())

        preds = np.concatenate (preds, axis=0).astype (np.float32)
        targets = np.concatenate (targets, axis=0).astype (np.float32)

        # 逆变换预测值与真实值
        preds = self.safe_inverse_transform (preds, scaler, target_indices)
        targets = self.safe_inverse_transform (targets, scaler, target_indices)

        # 逐列计算 MSE 和 R²
        mse_list = []
        r2_list = []
        for i in range (preds.shape[1]):
            mse_list.append (mean_squared_error (targets[:, i], preds[:, i]))
            r2_list.append (r2_score (targets[:, i], preds[:, i]))

        return mse_list, r2_list, preds, targets

    def safe_inverse_transform (self, preds, scaler, target_indices):
        """
        仅对目标列进行逆变换，使用 MinMaxScaler 的 data_min_ 和 data_range_
        """
        preds_inv = preds.copy ()
        for i, col_idx in enumerate (target_indices):
            preds_inv[:, i] = preds[:, i] * scaler.data_range_[col_idx] + scaler.data_min_[col_idx]
        return preds_inv

    def train_model (self, train_dataset, val_dataset=None, num_epochs=20, batch_size=32, learning_rate=1e-3,
                     scaler=None, target_indices=[0, 1, 2, 3], patience=5, min_delta=1e-5, monitor_target=0):
        """
        训练模型，并在每个 epoch 后在验证集上评估模型表现，并记录训练和验证指标。

        参数：
        - train_dataset: 训练数据集 (input, target)
        - val_dataset: 验证数据集；若不为 None，则每个 epoch 结束后进行评估
        - num_epochs: 训练轮数
        - batch_size: 每个 batch 的样本数
        - learning_rate: 学习率
        - scaler: 用于逆变换的 scaler
        - target_indices: 模型输出中，对应原始特征的列索引
        - patience: early stopping 容忍 epoch 数
        - min_delta: 最小改进量，用于 early stopping
        - monitor_target: 使用哪个目标的 MSE（按输出顺序）作为 early stopping 的判断指标

        返回：
        - train_losses: 每个 epoch 的训练损失
        - val_mse_lists: 每个 epoch 的验证集多目标 MSE 列表
        - val_r2_list: 每个 epoch 的验证集 R²
        """
        best_val_mse = float ('inf')
        epochs_no_improve = 0
        early_stop = False

        train_loader = DataLoader (train_dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss ()
        optimizer = torch.optim.Adam (self.parameters (), lr=learning_rate, weight_decay=1e-5)

        train_losses = []
        val_mse_lists = []
        val_r2_lists = []

        for epoch in range (num_epochs):
            self.train ()
            total_loss = 0.0

            for batch_inputs, batch_targets in train_loader:
                optimizer.zero_grad ()
                outputs = self (batch_inputs)
                loss = criterion (outputs, batch_targets)
                loss.backward ()
                optimizer.step ()
                total_loss += loss.item ()

            avg_loss = total_loss / len (train_loader)
            train_losses.append (avg_loss)
            print (f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.6f}")

            if val_dataset is not None:
                mse_list, r2_list, _, _ = self.evaluate_model (val_dataset, batch_size=batch_size,
                                                               scaler=scaler, target_indices=target_indices)
                val_mse_lists.append (mse_list)
                val_r2_lists.append (r2_list)

                monitored_mse = mse_list[monitor_target]

                if monitored_mse + min_delta < best_val_mse:
                    best_val_mse = monitored_mse
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print (f"Early stopping at epoch {epoch + 1}")
                        early_stop = True
                        break

                print (f"Epoch {epoch + 1}/{num_epochs}, Val MSEs: {mse_list}, R²: {r2_list}")

        if not early_stop:
            print ("Training finished without early stopping.")

        return train_losses, val_mse_lists, val_r2_lists


# 测试模型（示例）
if __name__ == "__main__":
    # 假设输入特征维度为 10，序列长度为 50，批大小为 32
    batch_size = 32
    seq_length = 50
    input_dim = 10

    # 定义模型参数
    model_dim = 64
    num_heads = 4
    num_layers = 2
    output_dim = 1

    # 构造模型实例
    model = TimeSeriesTransformer (input_dim=input_dim,
                                   model_dim=model_dim,
                                   num_heads=num_heads,
                                   num_layers=num_layers,
                                   dropout=0.1,
                                   seq_length=seq_length,
                                   output_dim=output_dim)

    # 构造随机输入数据
    sample_input = torch.randn (batch_size, seq_length, input_dim)
    output = model (sample_input)
    print ("模型输出形状：", output.shape)  # 期望形状为 [batch_size, output_dim]
