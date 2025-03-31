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

        # 将原始输入映射到模型维度
        self.input_linear = nn.Linear (input_dim, model_dim)
        # 位置编码模块
        self.positional_encoding = PositionalEncoding (model_dim, dropout=dropout, max_len=seq_length)
        # 构建 Transformer 编码器层 Encoder only
        encoder_layer = nn.TransformerEncoderLayer (d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder (encoder_layer, num_layers=num_layers)
        # 输出层，将 Transformer 输出映射到目标维度
        self.output_regressor = KAN([model_dim, 64, output_dim]) # 两层kan: [model_dim -> 64]   [64 -> output_dim]

    def forward (self, src):
        """
        参数:
        - src: 输入序列数据，形状为 [batch_size, seq_length, input_dim]
        """
        # 线性映射并缩放
        src = self.input_linear (src) * math.sqrt (self.model_dim)
        # 添加位置编码
        src = self.positional_encoding (src)
        # Transformer 要求的输入形状为 [seq_length, batch_size, model_dim]
        src = src.transpose (0, 1)
        # Transformer 编码器
        encoder_output = self.transformer_encoder (src)
        # 转换回 [batch_size, seq_length, model_dim]
        encoder_output = encoder_output.transpose (0, 1)
        # 此处采用最后一个时间步的输出作为整体序列的表示，也可以根据需求做其他聚合（如均值、加权等）
        out = self.output_regressor (encoder_output[:, -1, :])
        return out

    def evaluate_model (self, dataset, batch_size=32):
        """
        在验证集上评估模型，计算 MSE、R² 和基于 SHAP 的指标（平均绝对 SHAP 值）。

        参数：
        - dataset: 验证数据集，每个元素为 (input, target)
        - batch_size: 数据加载时的 batch 大小

        返回：
        - mse: 均方误差
        - r2: R² 得分
        - shap_metric: 平均绝对 SHAP 值
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
        preds = np.concatenate (preds, axis=0)
        targets = np.concatenate (targets, axis=0)

        mse = mean_squared_error (targets, preds)
        r2 = r2_score (targets, preds)

        # 为 SHAP 计算选择一个 batch作为背景数据（建议不要过大，否则计算较慢）
        # for batch_inputs, _ in val_loader:
        #     background = batch_inputs[:min (100, batch_inputs.shape[0])]
        #     break
        # # 使用 DeepExplainer 计算 SHAP 值（注意：此处仅对背景数据计算）
        # explainer = shap.DeepExplainer (self, background)
        # shap_values = explainer.shap_values (background)
        # # 计算所有输出维度上平均绝对 SHAP 值作为指标
        # shap_metric = np.mean ([np.abs (sv).mean () for sv in shap_values])

        # return mse, r2, shap_metric
        return mse, r2


    def train_model (self, train_dataset, val_dataset=None, num_epochs=20, batch_size=32, learning_rate=1e-3):
        """
        训练模型，并在每个 epoch 后在验证集上评估模型表现，并记录训练和验证指标。
        用法：
        先用 dataloader

        参数：
        - train_dataset: 训练数据集，每个元素为 (input, target)
        - val_dataset: 验证数据集；若不为 None，则每个 epoch 结束后进行评估
        - num_epochs: 训练轮数
        - batch_size: 每个 batch 的样本数
        - learning_rate: 学习率

        返回：
        - train_losses: 每个 epoch 的训练损失
        - val_mse_list: 每个 epoch 的验证集 MSE（若提供 val_dataset，否则为空列表）
        - val_r2_list: 每个 epoch 的验证集 R² 得分
        - val_shap_list: 每个 epoch 的验证集平均绝对 SHAP 值
        """
        train_loader = DataLoader (train_dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss ()
        optimizer = optim.Adam (self.parameters (), lr=learning_rate)

        # 用于记录每个 epoch 的指标
        train_losses = []
        val_mse_list = []
        val_r2_list = []
        val_shap_list = []

        for epoch in range (num_epochs):
            self.train ()  # 设置为训练模式
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
                # mse, r2, shap_metric = self.evaluate_model (val_dataset, batch_size=batch_size)
                mse, r2 = self.evaluate_model (val_dataset, batch_size=batch_size)
                val_mse_list.append (mse)
                val_r2_list.append (r2)
                # val_shap_list.append (shap_metric)
                # print (f"Epoch {epoch + 1}/{num_epochs}, Val MSE: {mse:.4f}, R^2: {r2:.4f}, SHAP: {shap_metric:.4f}")
                print (f"Epoch {epoch + 1}/{num_epochs}, Val MSE: {mse:.6f}, R^2: {r2:.4f}")

        # return train_losses, val_mse_list, val_r2_list, val_shap_list
        return train_losses, val_mse_list, val_r2_list


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