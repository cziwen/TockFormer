import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class MLPBlender(nn.Module):
    """
    用多层全连接网络做 stacking，且可从第一层权重估算每个 base model 的贡献度。
    """
    def __init__(self,
                 base_models: list,
                 output_dim: int,
                 hidden_dims: list = [32, 16],
                 dropout: float = 0.1,
                 device: torch.device = None):
        super(MLPBlender, self).__init__()
        self.base_models = base_models
        self.output_dim = output_dim
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

    def fit(self, val_dataset, num_epochs=50, batch_size=32, lr=1e-3, log=False):
        # 准备 meta-features
        loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        meta_X, meta_y = [], []
        for xb, yb in loader:
            meta_X.append(self._gather_base_preds(xb).cpu())
            meta_y.append(yb)
        X = torch.cat(meta_X, dim=0).to(self.device)
        Y = torch.cat(meta_y, dim=0).to(self.device)

        ds = TensorDataset(X, Y)
        train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        opt = optim.Adam(self.blender.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.blender.train()
        for ep in range(1, num_epochs+1):
            total = 0.0
            for xb, yb in train_loader:
                opt.zero_grad()
                out = self.blender(xb)
                loss = loss_fn(out, yb.to(self.device))
                loss.backward()
                opt.step()
                total += loss.item()
            if log:
                print(f"[MLPBlender] Epoch {ep}/{num_epochs} – Loss {total/len(train_loader):.6f}")

    def predict(self, x_dataset, batch_size=32):
        loader = DataLoader(x_dataset, batch_size=batch_size, shuffle=False)
        preds = []
        self.blender.eval()
        with torch.no_grad():
            for batch in loader:
                x = batch[0] if isinstance(batch, (list,tuple)) else batch
                feats = self._gather_base_preds(x)
                preds.append(self.blender(feats.to(self.device)).cpu())
        return torch.cat(preds, dim=0).numpy()

    def get_model_importance(self):
        """
        从 MLP 第一层权重估算各 base model 贡献度：
        - 第一层是 nn.Linear(in_dim, hidden_dims[0])
        - 它的 weight.shape == [hidden_dims[0], M*output_dim]
        """
        first_lin: nn.Linear = self.blender[0]
        W = first_lin.weight.data.abs().cpu().numpy()    # [H, M*od]
        M, od = len(self.base_models), self.output_dim
        contrib = []
        for i in range(M):
            block = W[:, i*od:(i+1)*od]
            contrib.append(block.sum())
        contrib = np.array(contrib)
        if contrib.sum() > 0:
            contrib = contrib / contrib.sum()
        return {f"model_{i}": float(contrib[i]) for i in range(M)}