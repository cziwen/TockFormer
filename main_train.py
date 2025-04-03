import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import joblib

from TransformerModel import TimeSeriesTransformer
from Util import create_sequences, safeLoadCSV, plot_metric


def train_model_main(train_csv, val_csv, model_out, scaler_out, batch_size=32, seq_length=32,
                     input_dim=33, model_dim=64, num_heads=4, num_layers=2,
                     output_dim=4, dropout=0.2, epochs=30, lr=1e-4, patience=5):

    print("=" * 10 + " 初始化模型中... " + "=" * 10)

    model = TimeSeriesTransformer(
        input_dim=input_dim,
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        seq_length=seq_length,
        output_dim=output_dim
    )

    print("=" * 10 + " 加载训练数据中... " + "=" * 10)
    train_df = safeLoadCSV(pd.read_csv(train_csv))
    x, y, scaler_train, target_indices = create_sequences(train_df, seq_length=seq_length,
                                                          target_cols=['open', 'high', 'low', 'close'])
    train_dataset = TensorDataset(x, y)

    print("=" * 10 + " 加载验证数据中... " + "=" * 10)
    val_df = pd.read_csv(val_csv)
    x_val, y_val, _, _ = create_sequences(val_df, seq_length=seq_length,
                                          target_cols=['open', 'high', 'low', 'close'],
                                          scaler=scaler_train)
    val_dataset = TensorDataset(x_val, y_val)

    print("=" * 10 + " 开始训练模型... " + "=" * 10)
    train_loss, mse_list, r2_list = model.train_model(
        train_dataset,
        val_dataset=val_dataset,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        scaler=scaler_train,
        target_indices=target_indices,
        patience=patience,
    )

    # 创建保存目录
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_out), exist_ok=True)

    print("=" * 10 + " 保存模型和Scaler... " + "=" * 10)
    torch.save(model.state_dict(), model_out)
    joblib.dump(scaler_train, scaler_out)

    print("=" * 10 + " 绘图中... " + "=" * 10)
    mse_list = np.array(mse_list)
    r2_list = np.array(r2_list)
    plot_metric(train_loss, y_label="loss", title="Train Loss", color='red')
    plot_metric(mse_list[:, 3], y_label="mse", title="Val MSE", color='green')
    plot_metric(r2_list[:, 3], y_label="r2", title="Val R²", color='blue')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练时间序列Transformer模型")
    parser.add_argument("--train", type=str, required=True, help="训练数据 CSV 路径")
    parser.add_argument("--val", type=str, required=True, help="验证数据 CSV 路径")
    parser.add_argument("--model_out", type=str, required=True, help="保存模型路径")
    parser.add_argument("--scaler_out", type=str, required=True, help="保存Scaler路径")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")

    args = parser.parse_args()

    train_model_main(
        train_csv=args.train,
        val_csv=args.val,
        model_out=args.model_out,
        scaler_out=args.scaler_out,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr
    )
