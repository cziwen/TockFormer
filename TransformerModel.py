import math
import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, ParameterGrid

from BiasCorrector import *


class PositionalEncoding (nn.Module):
    """
    位置编码模块：为输入的序列添加位置信息
    """

    def __init__ (self, d_model, dropout=0.1, max_len=5000):
        super (PositionalEncoding, self).__init__ ()
        self.dropout = nn.Dropout (p=dropout)

        # 构造一个 [max_len, d_model] 的位置编码矩阵
        pe = torch.zeros (max_len, d_model)
        position = torch.arange (0, max_len, dtype=torch.float).unsqueeze (1)
        # 使用 sin 和 cos 函数计算位置编码
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
    基于 Transformer 编码器的时间序列预测模型
    """

    def __init__ (self, input_dim, model_dim, num_heads, num_layers, dropout=0.1, seq_length=100, output_dim=1):
        """
        参数：
        - input_dim: 输入特征的维度
        - model_dim: 模型内部的特征维度
        - num_heads: 多头自注意力头数
        - num_layers: Transformer 编码器层数
        - dropout: dropout 概率
        - seq_length: 序列最大长度（用于位置编码）
        - output_dim: 模型输出的维度
        """
        super (TimeSeriesTransformer, self).__init__ ()
        self.model_dim = model_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 输入线性映射
        self.input_linear = nn.Linear (input_dim, model_dim)
        # 可以再加一层，做升维/降维

        # 位置编码模块
        self.positional_encoding = PositionalEncoding (model_dim, dropout=dropout, max_len=seq_length)
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer (d_model=model_dim, nhead=num_heads, dropout=dropout,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder (encoder_layer, num_layers=num_layers)
        # 输出映射层（回归输出）
        self.output_regressor = nn.Sequential (
            nn.Linear (model_dim, 64),
            nn.ReLU (),
            nn.Linear (64, output_dim)
        )

    def forward (self, src):
        """
        前向传播
        参数:
          - src: [batch_size, seq_length, input_dim]
        返回:
          - 输出形状为 [batch_size, output_dim]
        """
        # 线性映射并缩放
        src = self.input_linear (src) * math.sqrt (self.model_dim)
        # 可选的归一化
        # src = self.input_norm(src)
        # 添加位置编码
        src = self.positional_encoding (src)
        # Transformer 要求的输入形状为 [seq_length, batch_size, model_dim]
        src = src.transpose (0, 1)
        encoder_output = self.transformer_encoder (src)
        # 转回 [batch_size, seq_length, model_dim]
        encoder_output = encoder_output.transpose (0, 1)
        # 采用最后一个时间步作为整体序列表示
        x_last = encoder_output[:, -1, :]
        out = self.output_regressor (x_last)

        return out

    def safe_inverse_transform (self, preds, scaler, target_indices):
        """
        仅对目标列进行逆缩放
        """
        preds_inv = preds.copy ()
        for i, col_idx in enumerate (target_indices):
            preds_inv[:, i] = preds[:, i] * scaler.data_range_[col_idx] + scaler.data_min_[col_idx]
        return preds_inv

    def evaluate_model (self, dataset, batch_size=32, scaler=None, target_indices=[0, 1, 2, 3]):
        """
        在验证集上评估模型，计算各目标特征的 MSE 和 R²，并返回逆缩放后的预测值和真实值

        参数：
          - dataset: 验证数据集，每个元素为 (input, target)
          - batch_size: batch 大小
          - scaler: 用于逆缩放的 scaler
          - target_indices: 目标列索引

        返回：
          - mse_list: 每个目标的 MSE 列表
          - r2_list: 每个目标的 R² 列表
          - preds: 逆缩放后的预测值数组
          - targets: 逆缩放后的真实值数组
        """
        self.eval ()
        val_loader = DataLoader (dataset, batch_size=batch_size, shuffle=False)
        preds = []
        targets = []

        # 获取模型所在设备
        device = next (self.parameters ()).device

        with torch.no_grad ():
            for batch_inputs, batch_targets in val_loader:
                # 将输入迁移到模型所在设备，确保数据一致
                batch_inputs = batch_inputs.to (device)
                batch_targets = batch_targets.to (device)
                outputs = self (batch_inputs)
                preds.append (outputs.detach ().cpu ().numpy ())
                targets.append (batch_targets.detach ().cpu ().numpy ())

        preds = np.concatenate (preds, axis=0).astype (np.float32)
        targets = np.concatenate (targets, axis=0).astype (np.float32)

        # 逆变换预测值
        preds = self.safe_inverse_transform (preds, scaler, target_indices)
        targets = self.safe_inverse_transform (targets, scaler, target_indices)

        mse_list = []
        r2_list = []
        for i in range (preds.shape[1]):
            mse_list.append (mean_squared_error (targets[:, i], preds[:, i]))
            r2_list.append (r2_score (targets[:, i], preds[:, i]))

        return mse_list, r2_list, preds, targets

    def train_model (self, train_dataset, val_dataset=None, num_epochs=50, batch_size=32,
                     learning_rate=1e-4, scaler=None, target_indices=None, patience=10,
                     min_delta=1e-5, batch_shuffle_threshold=50, log=False):
        """
        训练模型，在每个 epoch 后评估验证集性能，并实现 early stopping 以及中间的训练数据重新打乱策略。
        如果在验证集上连续 patience 个 epoch 没有取得更好的性能，则回溯到上次表现最好的模型参数。
        同时，在每个 epoch 内部，对每个训练 batch 计算 loss，
        如果连续 batch 数超过 batch_shuffle_threshold 都没有得到更优的 loss，则重新打乱训练数据继续训练。

        返回：
          - train_losses: 每个 epoch 的训练损失
          - val_mse_lists: 每个 epoch 的验证 MSE 列表
          - val_r2_lists: 每个 epoch 的验证 R² 列表
        """

        device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
        print (f"Using device: {device}")
        self.to (device)

        best_val_mse = float ('inf')
        best_model_state = None  # 保存表现最好的模型参数
        epochs_no_improve = 0
        early_stop = False

        # 初始的训练 DataLoader（shuffle=True，每个 epoch 会重新打乱，但这里还加了内部打乱逻辑）
        train_loader = DataLoader (train_dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss ()
        optimizer = torch.optim.Adam (self.parameters (), lr=learning_rate, weight_decay=1e-5)

        train_losses = []
        val_mse_lists = []
        val_r2_lists = []

        for epoch in range (num_epochs):
            self.train ()
            total_loss = 0.0

            # 新增：在每个 epoch 内记录连续没有改进的 batch 数量
            bad_batch_count = 0
            best_batch_loss = float ('inf')

            # 为了支持中途重新打乱训练数据，使用 while 循环迭代 train_loader
            # 将 train_loader 转换为 iterator
            batch_iterator = iter (train_loader)
            num_batches = len (train_loader)
            batch_idx = 0

            while batch_idx < num_batches:
                try:
                    batch_inputs, batch_targets = next (batch_iterator)
                except StopIteration:
                    break

                batch_inputs = batch_inputs.to (device)
                batch_targets = batch_targets.to (device)

                optimizer.zero_grad ()
                outputs = self (batch_inputs)
                loss = criterion (outputs, batch_targets)
                loss.backward ()
                optimizer.step ()

                batch_loss = loss.item ()
                total_loss += batch_loss

                # 更新当前 epoch 内的最佳 batch loss 和计数连续差的 batch 数
                if batch_loss < best_batch_loss - min_delta:
                    best_batch_loss = batch_loss
                    bad_batch_count = 0
                else:
                    bad_batch_count += 1

                # 如果连续 bad batch 超过阈值，则重新打乱训练数据（内部策略，不做模型回溯）
                if bad_batch_count >= batch_shuffle_threshold:
                    if log:
                        print (
                            f"Epoch {epoch + 1}: Bad batch threshold reached at batch {batch_idx + 1}. Reshuffling training data.")
                    # 重新创建 train_loader
                    train_loader = DataLoader (train_dataset, batch_size=batch_size, shuffle=True)
                    batch_iterator = iter (train_loader)
                    num_batches = len (train_loader)
                    bad_batch_count = 0  # 重置计数
                    # 注意：这里不会回溯模型参数，只是换个顺序继续训练
                batch_idx += 1

            avg_loss = total_loss / num_batches
            train_losses.append (avg_loss)
            if log:
                print (f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_loss:.6f}")

            if val_dataset is not None:
                mse_list, r2_list, _, _ = self.evaluate_model (val_dataset, batch_size=batch_size,
                                                               scaler=scaler,
                                                               target_indices=target_indices)
                val_mse_lists.append (mse_list)
                val_r2_lists.append (r2_list)

                best_avg_mse = np.mean (mse_list)
                if log:
                    print (f"Epoch {epoch + 1}/{num_epochs}, Val MSEs: {mse_list}, R²: {r2_list}")

                # 如果当前验证指标比之前更好，则保存模型参数
                if best_avg_mse + min_delta < best_val_mse:
                    best_val_mse = best_avg_mse
                    epochs_no_improve = 0
                    best_model_state = copy.deepcopy (self.state_dict ())  # 保存最佳模型参数
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print (f"Early stopping triggered at epoch {epoch + 1}. Reverting to best model parameters.")
                        early_stop = True
                        # 回溯至最佳模型参数
                        self.load_state_dict (best_model_state)
                        break

        if not early_stop:
            print ("Training finished without early stopping.")

        return train_losses, val_mse_lists, val_r2_lists

    def predict_model (self, X_tensor, scaler, bias_corrector, target_indices):
        """
        使用模型进行预测，并对预测结果进行逆归一化。
        输入 X_tensor 的形状应为 [1, seq_length, input_dim]，
        输出预测结果（经过逆归一化），形状为 [1, output_dim]。

        参数:
          - X_tensor: torch.Tensor，预测时的输入数据
          - scaler: 用于逆归一化的 scaler
          - bias_corrector: 如果不为 None，则对预测结果进行 bias 校正
          - target_indices: 用于逆归一化的目标列索引

        返回:
          - pred_inv: numpy 数组，逆归一化后的预测输出
        """
        self.eval ()
        device = next (self.parameters ()).device
        X_tensor = X_tensor.to (device)
        with torch.no_grad ():
            pred = self (X_tensor)
        # 将输出转换为 numpy 数组
        pred = pred.detach ().cpu ().numpy ().astype (np.float32)

        # 逆归一化（安全逆变换），使预测结果恢复到原始数值域
        pred_inv = self.safe_inverse_transform (pred, scaler, target_indices)

        # 如果传入 bias_corrector，则进行 bias 校正
        if bias_corrector is not None:
            pred_inv = bias_corrector.transform (pred_inv)

        return pred_inv

    def cross_validate (self, dataset, k=5, num_epochs=50,
                        batch_size=32, learning_rate=1e-4,
                        scaler=None, target_indices=None,
                        patience=5, min_delta=1e-5,
                        batch_shuffle_threshold=50):
        """
        执行 k 折交叉验证，返回每折的验证 MSE 和 R²。

        参数:
          - dataset: 完整数据集 (input, target) tuples, TensorDataset
          - k: 折数
          - 其余参数同 train_model

        返回:
          - results: 一个列表，元素为 dict，包含 'mse' 和 'r2' 两个字段，分别为该折最后一次验证的指标列表
        """
        kf = KFold (n_splits=k, shuffle=True, random_state=42)
        results = []
        for fold, (train_idx, val_idx) in enumerate (kf.split (dataset)):
            print (f"=== Fold {fold + 1}/{k} ===")
            train_sub = Subset (dataset, train_idx)
            val_sub = Subset (dataset, val_idx)
            # 深拷贝模型，保证每折从相同初始化开始
            model = copy.deepcopy (self)
            model.train_model (
                train_sub, val_dataset=val_sub,
                num_epochs=num_epochs, batch_size=batch_size,
                learning_rate=learning_rate, scaler=scaler,
                target_indices=target_indices, patience=patience,
                min_delta=min_delta, batch_shuffle_threshold=batch_shuffle_threshold
            )
            mse_list, r2_list, _, _ = model.evaluate_model (
                val_sub, batch_size=batch_size,
                scaler=scaler, target_indices=target_indices
            )
            results.append ({'mse': mse_list, 'r2': r2_list})
        return results


def grid_search (model_class, init_args, dataset, param_grid, cv=5,
                 scaler=None, target_indices=None):
    """
    对架构超参数进行网格搜索，使用交叉验证选择最优配置。

    参数:
      - model_class: TimeSeriesTransformer 类
      - init_args: dict, 除了可调超参外的固定初始化参数，例如 {'input_dim':49,'output_dim':4,'seq_length':100,'dropout':0.1}
      - dataset: 完整数据集, TensorDataset(input, label)
      - param_grid: dict, key 为模型初始化参数名（如 'model_dim','num_heads','num_layers'），value 为列表
      - cv: 折数
      - scaler, target_indices: 同前

    返回:
      - best_params: 最佳参数组合（仅包含可调参数）
      - best_score: 对应的平均 MSE
    """
    best_score = float ('inf')
    best_params = None
    for params in ParameterGrid (param_grid):
        print (f"Testing architecture params: {params}")
        # 合并固定参数与可调参数，重新实例化模型
        model_kwargs = {**init_args, **params}
        model = model_class (**model_kwargs)
        # 使用默认训练配置进行 CV
        cv_results = model.cross_validate (
            dataset,
            k=cv,
            scaler=scaler,
            target_indices=target_indices
        )
        # 计算平均 MSE
        mean_mse = np.mean ([np.mean (res['mse']) for res in cv_results])
        mean_r2 = np.mean ([np.mean (res['r2']) for res in cv_results])
        print (f" Avg CV MSE: {mean_mse:.6f}, Avg R2: {mean_r2} \n")
        if mean_mse < best_score:
            best_score = mean_mse
            best_params = params
    return best_params, best_score


# ===== 测试部分 =====
# if __name__ == "__main__":
#     # 测试时构造一个简单模型，并用随机数据测试前向传播
#     batch_size = 32
#     seq_length = 50
#     input_dim = 10
#
#     model_dim = 64
#     num_heads = 4
#     num_layers = 2
#     output_dim = 4
#
#     model = TimeSeriesTransformer (input_dim=input_dim,
#                                    model_dim=model_dim,
#                                    num_heads=num_heads,
#                                    num_layers=num_layers,
#                                    dropout=0.1,
#                                    seq_length=seq_length,
#                                    output_dim=output_dim)
#
#     # 构造随机输入数据，测试模型前向传播
#     sample_input = torch.randn (batch_size, seq_length, input_dim)
#     output = model (sample_input)
#     print ("模型输出形状：", output.shape)  # 期望形状为 [batch_size, output_dim]
