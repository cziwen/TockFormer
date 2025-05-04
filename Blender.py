import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import copy

class MLPBlender(nn.Module):
    """
    用多层全连接网络做 stacking，支持 early stopping 和训练时动态打乱数据。
    """
    def __init__(self,
                 base_models: list,
                 output_dim: int,
                 hidden_dims: list = [32, 16],
                 dropout: float = 0.1,
                 scaler=None,
                 device: torch.device = None):
        super().__init__()
        self.base_models = base_models
        self.output_dim = output_dim
        self.scaler = scaler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 冻结 base_models
        for m in self.base_models:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

        # 构建 MLP
        in_dim = len(self.base_models) * output_dim
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.blender = nn.Sequential(*layers).to(self.device)

    def _gather_base_preds(self, x_batch):
        feats = []
        with torch.no_grad():
            for m in self.base_models:
                feats.append(m(x_batch.to(self.device)))
        return torch.cat(feats, dim=1)

    def fit(self,
            train_dataset: TensorDataset,
            val_dataset: TensorDataset,
            num_epochs: int = 50,
            batch_size: int = 32,
            lr: float = 1e-3,
            patience: int = 5,
            bad_batch_threshold: int = 50,
            log: bool = False):
        """
        训练 MLP并支持 early stopping。

        参数：
          - train_dataset: 训练数据集
          - val_dataset: 验证数据集，用于 early stopping
          - num_epochs: 最大训练轮数
          - batch_size: 批大小
          - lr: 学习率
          - patience: early stopping 容忍的最大不提升轮数
          - bad_batch_threshold: 每轮允许的“坏批次”阈值（loss增加）
          - log: 是否打印日志
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        print (f"Using device: {device}")

        # 数据加载
        ds_train = train_dataset
        ds_val = val_dataset
        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(self.blender.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        # 初始化 early stopping
        best_weights = copy.deepcopy(self.blender.state_dict())
        best_score = float('inf')  # 用于比较的 val 评分（MSE 均值）
        epochs_no_improve = 0

        for ep in range(1, num_epochs + 1):
            # --- 训练 ---
            self.blender.train()
            total_loss = 0.0
            bad_batches = 0
            prev_loss = None
            for xb, yb in train_loader:
                optimizer.zero_grad()
                feats = self._gather_base_preds(xb)
                out = self.blender(feats.to(self.device))
                loss = loss_fn(out, yb.to(self.device))
                loss.backward()
                optimizer.step()
                curr_loss = loss.item()
                total_loss += curr_loss
                # bad batch 判断
                if prev_loss is not None and curr_loss > prev_loss:
                    bad_batches += 1
                    if bad_batches > bad_batch_threshold:
                        if log:
                            print(f"Epoch {ep}: bad_batches>{bad_batch_threshold}, reshuffling data.")
                        train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
                        bad_batches = 0
                prev_loss = curr_loss
            avg_train_loss = total_loss / len(train_loader)

            # --- 验证 ---
            mse_list, r2_list, _, _ = self.evaluate_model(ds_val, batch_size=batch_size, scaler=self.scaler)
            avg_val_mse = np.mean(mse_list)

            if log:
                print(f"Epoch {ep}: train_loss={avg_train_loss:.6f}, val_mse={avg_val_mse:.6f}, val_r2={r2_list}")

            # early stopping
            if avg_val_mse < best_score:
                best_score = avg_val_mse
                best_weights = copy.deepcopy(self.blender.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if log:
                        print(f"Early stopping at epoch {ep}, restoring best model.")
                    break

        # 恢复最佳权重
        self.blender.load_state_dict(best_weights)




    def predict(self, x_dataset, batch_size=32):
        loader = DataLoader(x_dataset, batch_size=batch_size, shuffle=False)
        preds = []
        self.blender.eval()
        with torch.no_grad():
            for batch in loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                feats = self._gather_base_preds(x)
                preds.append(self.blender(feats.to(self.device)).cpu())
        return torch.cat(preds, dim=0).numpy()

    def get_model_importance(self):
        first_lin: nn.Linear = self.blender[0]
        W = first_lin.weight.data.abs().cpu().numpy()
        M, od = len(self.base_models), self.output_dim
        contrib = []
        for i in range(M):
            block = W[:, i*od:(i+1)*od]
            contrib.append(block.sum())
        contrib = np.array(contrib)
        if contrib.sum() > 0:
            contrib = contrib / contrib.sum()
        return {f"model_{self.base_models[i].__class__.__name__}": float(contrib[i]) for i in range(M)}

    def evaluate_model(self, dataset, batch_size=32, scaler=None, target_indices=None):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        preds, targets = [], []

        self.blender.eval()
        with torch.no_grad():
            for xb, yb in loader:
                feats = self._gather_base_preds(xb)
                out = self.blender(feats.to(self.device)).cpu().numpy()
                preds.append(out)
                targets.append(yb.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)

        if scaler is not None and target_indices is not None:
            from Util import safe_inverse_transform
            preds = safe_inverse_transform(preds, scaler, target_indices)
            targets = safe_inverse_transform(targets, scaler, target_indices)

        mse_list = [mean_squared_error(targets[:, i], preds[:, i]) for i in range(preds.shape[1])]
        r2_list = [r2_score(targets[:, i], preds[:, i]) for i in range(preds.shape[1])]

        return mse_list, r2_list, preds, targets
