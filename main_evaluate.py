# import argparse
# import os
# import torch
# import joblib
# import pandas as pd
# import numpy as np
# from torch.utils.data import TensorDataset
#
# from TransformerModel import TimeSeriesTransformer
# from Util import create_sequences, plot_multiple_curves, safeLoadCSV
#
#
# def evaluate_model_main(test_csv, model_path, scaler_path,
#                         batch_size=32, seq_length=32,
#                         input_dim=33, model_dim=64, num_heads=4, num_layers=2,
#                         output_dim=4, dropout=0.2):
#
#     print("=" * 10 + " 加载模型和 Scaler... " + "=" * 10)
#     model = TimeSeriesTransformer(
#         input_dim=input_dim,
#         model_dim=model_dim,
#         num_heads=num_heads,
#         num_layers=num_layers,
#         dropout=dropout,
#         seq_length=seq_length,
#         output_dim=output_dim
#     )
#
#     device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
#     model.load_state_dict (torch.load (model_path, map_location=device))
#     model.to (device)
#     scaler = joblib.load(scaler_path)
#
#     print("=" * 10 + " 加载测试数据... " + "=" * 10)
#     test_df = safeLoadCSV(pd.read_csv(test_csv))
#     x_test, y_test, _, target_indices = create_sequences(
#         test_df, seq_length=seq_length, scaler=scaler,
#         target_cols=['open', 'high', 'low', 'close']
#     )
#     test_dataset = TensorDataset(x_test, y_test)
#
#     print("=" * 10 + " 开始评估模型... " + "=" * 10)
#     mse_list, r2_list, preds, actuals = model.evaluate_model(
#         test_dataset,
#         batch_size=batch_size,
#         scaler=scaler,
#         target_indices=target_indices
#     )
#
#     print("=" * 10 + " 评估完成... " + "=" * 10)
#     print(f"📉 MSE: {mse_list}")
#     print(f"📈 R² : {r2_list}")
#
#     # 可视化预测结果（收盘价）
#     curve_dict = {
#         'Close_pred': preds[:, 3],
#         'Close_actual': actuals[:, 3]
#     }
#     plot_multiple_curves(
#         curve_dict,
#         x_label='interval',
#         y_label='price',
#         title='Close Comparison'
#     )
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="在测试集上评估 Transformer 模型性能")
#     parser.add_argument('--test', type=str, required=True, help='测试数据 CSV 路径')
#     parser.add_argument('--model', type=str, required=True, help='模型文件路径（.pth）')
#     parser.add_argument('--scaler', type=str, required=True, help='Scaler 文件路径（.pkl）')
#     parser.add_argument('--batch_size', type=int, default=32, help='批大小')
#     parser.add_argument ('--sequential_length', type=int, default=32, help='序列长度')
#     parser.add_argument ("--input_dim", type=int, required=True, help="模型输入维度")
#     parser.add_argument ("--output_dim", type=int, required=True, help="模型输出维度")
#     parser.add_argument ("--num_layers", type=int, required=True, help="模型层数")
#     parser.add_argument ("--num_heads", type=int, required=True, help="注意力头数")
#     parser.add_argument ("--dropout", type=float, default=0.2, help="Dropout")
#
#
#
#     args = parser.parse_args()
#
#     evaluate_model_main(
#         test_csv=args.test,
#         model_path=args.model,
#         scaler_path=args.scaler,
#         batch_size=args.batch_size,
#         seq_length=args.sequential_length,
#         input_dim=args.input_dim,
#         output_dim=args.output_dim,
#         num_layers=args.num_layers,
#         num_heads=args.num_heads,
#         dropout=args.dropout
#     )

import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
import joblib

from TransformerModel import TimeSeriesTransformer
from TransformerModel_classify import TimeSeriesTransformer_classify
from Util import create_sequences, safeLoadCSV, plot_metric, plot_multiple_curves


def evaluate_model_regression_main(test_csv, model_path, scaler_path, batch_size=32, seq_length=32,
                                   input_dim=33, model_dim=64, output_dim=4, num_layers=2,
                                   num_heads=4, dropout=0.2):
    """
    回归任务评估函数：
    - 加载测试数据（假设目标列为 ['open', 'high', 'low', 'close']）
    - 使用 scaler 对模型预测结果逆缩放，并计算 MSE 和 R² 指标
    """
    print("=" * 10 + " 加载回归任务测试数据中... " + "=" * 10)
    test_df = pd.read_csv(test_csv)
    scaler = joblib.load (scaler_path)
    # 注意：这里目标列为回归任务的目标，名称可根据实际情况调整
    x_test, y_test, _, target_indices = create_sequences(test_df, seq_length=seq_length,
                                                              target_cols=['open', 'high', 'low', 'close'], scaler=scaler)
    test_dataset = TensorDataset(x_test, y_test)

    print("=" * 10 + " 构造回归模型并加载参数... " + "=" * 10)
    model = TimeSeriesTransformer(
        input_dim=input_dim,
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        seq_length=seq_length,
        output_dim=output_dim
    )

    device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 使用回归评估函数，返回 MSE、R² 等指标
    mse_list, r2_list, preds, targets = model.evaluate_model(test_dataset, batch_size=batch_size,
                                                              scaler=scaler, target_indices=target_indices)
    print("测试集 MSE: ", mse_list)
    print("测试集 R²: ", r2_list)


    curve_dict = {}
    curve_dict['predicts'] = preds[:, 3]
    curve_dict['targets'] = targets[:, 3]
    plot_multiple_curves(curve_dict, x_label= 'interval', y_label= 'price')


def evaluate_model_classification_main(test_csv, model_path, batch_size=32, seq_length=32,
                                       input_dim=33, model_dim=64, output_dim=3, num_layers=2,
                                       num_heads=4, dropout=0.2):
    """
    分类任务评估函数：
    - 加载测试数据（假设目标列为 'label'，标签为整数）
    - 直接计算分类准确率
    """
    print("=" * 10 + " 加载分类任务测试数据中... " + "=" * 10)
    test_df = pd.read_csv(test_csv)
    # 修改点：针对分类任务，假设目标列名称为 'label'
    x_test, y_test, _, _ = create_sequences(test_df, seq_length=seq_length,
                                            target_cols=['label'])
    # 将标签转换为 LongTensor 以适用于 CrossEntropyLoss
    # y_test = y_test.long()
    test_dataset = TensorDataset(x_test, y_test)

    print("=" * 10 + " 构造分类模型并加载参数... " + "=" * 10)
    model = TimeSeriesTransformer_classify(
        input_dim=input_dim,
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        seq_length=seq_length,
        output_dim=output_dim  # output_dim 表示类别数
    )

    device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
    model.load_state_dict (torch.load (model_path, map_location=device))
    model.eval ()


    # 使用分类评估函数，返回准确率
    accuracy = model.evaluate_model(test_dataset, batch_size=batch_size)
    print("测试集准确率: {:.4f}".format(accuracy))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在测试集上评估 Transformer 模型性能")
    # 新增 --task 参数，用于区分回归和分类任务
    parser.add_argument("--task", type=str, default="regression", choices=["regression", "classification"],
                        help="任务类型：回归或分类")
    parser.add_argument('--test', type=str, required=True, help='测试数据 CSV 路径')
    parser.add_argument('--model', type=str, required=True, help='模型文件路径（.pth）')
    # 对于回归任务需要 scaler 文件，分类任务不需要
    parser.add_argument('--scaler', type=str, help='Scaler 文件路径（.pkl），仅回归任务使用')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--sequential_length', type=int, default=32, help='序列长度')
    parser.add_argument("--input_dim", type=int, required=True, help="模型输入维度")
    parser.add_argument("--output_dim", type=int, required=True, help="模型输出维度（回归：目标数；分类：类别数）")
    # 新增模型映射维度参数，保证与模型结构一致
    parser.add_argument("--model_dim", type=int, required=True, help="模型映射维度")
    parser.add_argument("--num_layers", type=int, required=True, help="模型层数")
    parser.add_argument("--num_heads", type=int, required=True, help="注意力头数")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout")

    args = parser.parse_args()

    if args.task == "regression":
        if args.scaler is None:
            raise ValueError("回归任务必须提供 --scaler 参数")
        evaluate_model_regression_main(
            test_csv=args.test,
            model_path=args.model,
            scaler_path=args.scaler,
            batch_size=args.batch_size,
            seq_length=args.sequential_length,
            input_dim=args.input_dim,
            model_dim=args.model_dim,
            output_dim=args.output_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout
        )
    elif args.task == "classification":
        evaluate_model_classification_main(
            test_csv=args.test,
            model_path=args.model,
            batch_size=args.batch_size,
            seq_length=args.sequential_length,
            input_dim=args.input_dim,
            model_dim=args.model_dim,
            output_dim=args.output_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout
        )