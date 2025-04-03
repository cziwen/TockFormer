import argparse
import os
import torch
import joblib
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset

from TransformerModel import TimeSeriesTransformer
from Util import create_sequences, plot_multiple_curves, safeLoadCSV


def evaluate_model_main(test_csv, model_path, scaler_path,
                        batch_size=32, seq_length=32,
                        input_dim=33, model_dim=64, num_heads=4, num_layers=2,
                        output_dim=4, dropout=0.2):

    print("=" * 10 + " 加载模型和 Scaler... " + "=" * 10)
    model = TimeSeriesTransformer(
        input_dim=input_dim,
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        seq_length=seq_length,
        output_dim=output_dim
    )
    model.load_state_dict(torch.load(model_path))
    scaler = joblib.load(scaler_path)

    print("=" * 10 + " 加载测试数据... " + "=" * 10)
    test_df = safeLoadCSV(pd.read_csv(test_csv))
    x_test, y_test, _, target_indices = create_sequences(
        test_df, seq_length=seq_length, scaler=scaler,
        target_cols=['open', 'high', 'low', 'close']
    )
    test_dataset = TensorDataset(x_test, y_test)

    print("=" * 10 + " 开始评估模型... " + "=" * 10)
    mse_list, r2_list, preds, actuals = model.evaluate_model(
        test_dataset,
        batch_size=batch_size,
        scaler=scaler,
        target_indices=target_indices
    )

    print("=" * 10 + " 评估完成... " + "=" * 10)
    print(f"📉 MSE: {mse_list}")
    print(f"📈 R² : {r2_list}")

    # 可视化预测结果（收盘价）
    curve_dict = {
        'Close_pred': preds[:, 3],
        'Close_actual': actuals[:, 3]
    }
    plot_multiple_curves(
        curve_dict,
        x_label='interval',
        y_label='price',
        title='Close Comparison'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在测试集上评估 Transformer 模型性能")
    parser.add_argument('--test', type=str, required=True, help='测试数据 CSV 路径')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径（.pth）')
    parser.add_argument('--scaler', type=str, required=True, help='Scaler 文件路径（.pkl）')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    args = parser.parse_args()

    evaluate_model_main(
        test_csv=args.test,
        model_path=args.model,
        scaler_path=args.scaler,
        batch_size=args.batch_size
    )