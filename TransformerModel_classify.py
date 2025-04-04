import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

class PositionalEncoding(nn.Module):
    """
    位置编码模块：为输入的序列添加位置信息
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_length, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TimeSeriesTransformer_classify(nn.Module):
    """
    基于 Transformer 编码器的时间序列分类模型（修改自回归模型）
    """
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1, seq_length=100, output_dim=3):
        """
        参数说明：
        - input_dim: 输入特征的维度
        - model_dim: 模型内部的特征维度
        - num_heads: 多头自注意力头数
        - num_layers: Transformer 编码器层数
        - dropout: dropout 概率
        - seq_length: 序列最大长度（用于位置编码）
        - output_dim: 分类类别数（对于分类任务，output_dim 表示类别数）
        """
        super(TimeSeriesTransformer_classify, self).__init__()
        self.model_dim = model_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 输入线性映射
        self.input_linear = nn.Linear(input_dim, model_dim)
        # 位置编码模块
        self.positional_encoding = PositionalEncoding(model_dim, dropout=dropout, max_len=seq_length)
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 输出分类器（修改点：改为分类器，不再做回归输出）
        self.output_classifier = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
            # 注意：不需要 softmax，这里返回 logits，交叉熵损失会内部计算 softmax
        )

    def forward(self, src):
        """
        前向传播
        参数:
          - src: [batch_size, seq_length, input_dim]
        返回:
          - 输出形状为 [batch_size, output_dim]（类别 logits）
        """
        src = self.input_linear(src) * math.sqrt(self.model_dim)
        src = self.positional_encoding(src)
        src = src.transpose(0, 1)  # Transformer 需要 [seq_length, batch_size, model_dim]
        encoder_output = self.transformer_encoder(src)
        encoder_output = encoder_output.transpose(0, 1)  # 还原为 [batch_size, seq_length, model_dim]
        # 采用最后一个时间步的表示作为整体序列表示，传入分类器
        out = self.output_classifier(encoder_output[:, -1, :])
        return out

    # 删除 safe_inverse_transform，因为分类任务不需要逆缩放

    def evaluate_model(self, dataset, batch_size=32):
        """
        在验证集上评估模型，计算准确率
        参数：
          - dataset: 验证数据集，每个元素为 (input, target)，其中 target 为类别标签（整数）
          - batch_size: batch 大小
        返回:
          - accuracy: 模型在验证集上的准确率
        """
        self.eval()
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_preds = []
        all_targets = []
        device = next(self.parameters()).device

        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                outputs = self(batch_inputs)  # 输出 logits，形状 [batch_size, output_dim]
                preds = outputs
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_targets.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_preds)
        return accuracy

    def train_model(self, train_dataset, val_dataset=None, num_epochs=30, batch_size=32,
                    learning_rate=1e-4, patience=5, min_delta=0.0):
        """
        训练模型，每个 epoch 后在验证集上评估准确率，并实现 early stopping。
        修改点：
          - 损失函数从 MSELoss 改为 CrossEntropyLoss（适用于多分类）
          - 评价指标改为准确率
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        self.to(device)

        best_val_acc = 0.0
        epochs_no_improve = 0
        early_stop = False

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # 修改损失函数：使用交叉熵损失（目标标签应为类别索引）
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)

        train_losses = []
        val_accuracies = []

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0.0

            for batch_inputs, batch_targets in train_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)  # 目标应为整数标签
                optimizer.zero_grad()
                outputs = self(batch_inputs)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.6f}")

            if val_dataset is not None:
                val_acc = self.evaluate_model(val_dataset, batch_size=batch_size)
                val_accuracies.append(val_acc)
                print(f"Epoch {epoch + 1}/{num_epochs}, Val Accuracy: {val_acc:.4f}")

                # 若验证准确率提升不明显，则进行 early stopping
                if val_acc - min_delta > best_val_acc:
                    best_val_acc = val_acc
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        early_stop = True
                        break

        if not early_stop:
            print("Training finished without early stopping.")

        return train_losses, val_accuracies


# ===== 测试部分 =====
if __name__ == "__main__":
    # 假设做多分类任务，类别数设为 3（output_dim=3），输入数据保持不变
    batch_size = 32
    seq_length = 50
    input_dim = 10

    model_dim = 64
    num_heads = 4
    num_layers = 2
    output_dim = 3  # 修改点：类别数设为 3

    model = TimeSeriesTransformer_classify(input_dim=input_dim,
                                  model_dim=model_dim,
                                  num_heads=num_heads,
                                  num_layers=num_layers,
                                  dropout=0.1,
                                  seq_length=seq_length,
                                  output_dim=output_dim)

    # 构造随机输入数据，测试模型前向传播
    sample_input = torch.randn(batch_size, seq_length, input_dim)
    output = model(sample_input)
    print("模型输出形状：", output.shape)  # 期望形状为 [batch_size, output_dim]