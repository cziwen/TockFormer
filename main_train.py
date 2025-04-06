# import argparse
# import os
# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import TensorDataset
# import joblib
#
# from TransformerModel import TimeSeriesTransformer
# from Util import create_sequences, safeLoadCSV, plot_metric
#
#
# def train_model_main(train_csv, val_csv, model_out, scaler_out, batch_size=32, seq_length=32,
#                      input_dim=33, model_dim=64, num_heads=4, num_layers=2,
#                      output_dim=4, dropout=0.2, epochs=30, lr=1e-4, patience=5):
#
#     print("=" * 10 + " 初始化模型中... " + "=" * 10)
#
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
#     print("=" * 10 + " 加载训练数据中... " + "=" * 10)
#     train_df = safeLoadCSV(pd.read_csv(train_csv))
#     x, y, scaler_train, target_indices = create_sequences(train_df, seq_length=seq_length,
#                                                           target_cols=['open', 'high', 'low', 'close'])
#     train_dataset = TensorDataset(x, y)
#
#     print("=" * 10 + " 加载验证数据中... " + "=" * 10)
#     val_df = pd.read_csv(val_csv)
#     x_val, y_val, _, _ = create_sequences(val_df, seq_length=seq_length,
#                                           target_cols=['open', 'high', 'low', 'close'],
#                                           scaler=scaler_train)
#     val_dataset = TensorDataset(x_val, y_val)
#
#     print("=" * 10 + " 开始训练模型... " + "=" * 10)
#     train_loss, mse_list, r2_list = model.train_model(
#         train_dataset,
#         val_dataset=val_dataset,
#         num_epochs=epochs,
#         batch_size=batch_size,
#         learning_rate=lr,
#         scaler=scaler_train,
#         target_indices=target_indices,
#         patience=patience,
#     )
#
#     # 创建保存目录
#     os.makedirs(os.path.dirname(model_out), exist_ok=True)
#     os.makedirs(os.path.dirname(scaler_out), exist_ok=True)
#
#     print("=" * 10 + " 保存模型和Scaler... " + "=" * 10)
#     torch.save(model.state_dict(), model_out)
#     joblib.dump(scaler_train, scaler_out)
#
#     print("=" * 10 + " 绘图中... " + "=" * 10)
#     mse_list = np.array(mse_list)
#     r2_list = np.array(r2_list)
#     plot_metric(train_loss, y_label="loss", title="Train Loss", color='red')
#     plot_metric(mse_list[:, 3], y_label="mse", title="Val MSE", color='green')
#     plot_metric(r2_list[:, 3], y_label="r2", title="Val R²", color='blue')
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="训练时间序列Transformer模型")
#     parser.add_argument("--train", type=str, required=True, help="训练数据 CSV 路径")
#     parser.add_argument("--val", type=str, required=True, help="验证数据 CSV 路径")
#     parser.add_argument("--model_out", type=str, required=True, help="保存模型路径")
#     parser.add_argument("--scaler_out", type=str, required=True, help="保存Scaler路径")
#     parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
#     parser.add_argument("--batch_size", type=int, default=32, help="批大小")
#     parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
#     parser.add_argument("--sequential_length", type=int, default=32, help="序列长度")
#     parser.add_argument("--input_dim", type=int, required=True, help="模型输入维度")
#     parser.add_argument("--output_dim", type=int, required=True, help="模型输出维度")
#     parser.add_argument ("--model_dim", type=int, required=True, help="模型映射维度")
#     parser.add_argument ("--num_layers", type=int, required=True, help="模型层数")
#     parser.add_argument ("--num_heads", type=int, required=True, help="注意力头数")
#     parser.add_argument ("--dropout", type=float, required=True, help="Dropout")
#
#     args = parser.parse_args()
#
#     train_model_main(
#         train_csv=args.train,
#         val_csv=args.val,
#         model_out=args.model_out,
#         scaler_out=args.scaler_out,
#         batch_size=args.batch_size,
#         epochs=args.epochs,
#         lr=args.lr,
#         seq_length=args.sequential_length,
#         input_dim=args.input_dim,
#         output_dim=args.output_dim,
#         model_dim=args.model_dim,
#         num_layers=args.num_layers,
#         num_heads=args.num_heads,
#         dropout=args.dropout,
#     )

import argparse
import os
import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset
import joblib

from TransformerModel import TimeSeriesTransformer
from TransformerModel_classify import TimeSeriesTransformer_classify
from Util import create_sequences, safeLoadCSV, plot_metric


def train_model_main (train_csv, val_csv, model_out, scaler_out, bias_corrector_out, batch_size=32, seq_length=32,
                      input_dim=33, model_dim=64, num_heads=4, num_layers=2,
                      output_dim=4, dropout=0.2, epochs=30, lr=1e-4, patience=5):
    """
    回归任务训练主函数（原有代码）
    """
    print ("=" * 10 + " 初始化模型中... " + "=" * 10)

    model = TimeSeriesTransformer (
        input_dim=input_dim,
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        seq_length=seq_length,
        output_dim=output_dim
    )

    print ("=" * 10 + " 加载训练数据中... " + "=" * 10)
    train_df = safeLoadCSV (pd.read_csv (train_csv))
    # 对回归任务，目标列设为 ['open', 'high', 'low', 'close']
    x, y, scaler_train, target_indices = create_sequences (train_df, seq_length=seq_length,
                                                           target_cols=['open', 'high', 'low', 'close'])
    train_dataset = TensorDataset (x, y)

    print ("=" * 10 + " 加载验证数据中... " + "=" * 10)
    val_df = pd.read_csv (val_csv)
    x_val, y_val, _, _ = create_sequences (val_df, seq_length=seq_length,
                                           target_cols=['open', 'high', 'low', 'close'],
                                           scaler=scaler_train)
    val_dataset = TensorDataset (x_val, y_val)

    print ("=" * 10 + " 开始训练模型... " + "=" * 10)
    train_loss, mse_list, r2_list, bias_corrector = model.train_model (
        train_dataset,
        val_dataset=val_dataset,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        scaler=scaler_train,  # 回归任务需要 scaler 进行逆缩放
        target_indices=target_indices,  # 回归任务中目标列索引
        patience=patience,
    )

    # 创建保存目录
    os.makedirs (os.path.dirname (model_out), exist_ok=True)
    os.makedirs (os.path.dirname (scaler_out), exist_ok=True)
    os.makedirs (os.path.dirname (bias_corrector_out), exist_ok=True)

    print ("=" * 10 + " 保存模型, Scaler, Corrector... " + "=" * 10)
    torch.save (model.state_dict (), model_out)
    joblib.dump (scaler_train, scaler_out)
    bias_corrector.save (bias_corrector_out)

    print ("=" * 10 + " 绘图中... " + "=" * 10)
    mse_list = np.array (mse_list)
    r2_list = np.array (r2_list)
    plot_metric (train_loss, y_label="loss", title="Train Loss", color='red')
    plot_metric (mse_list[:, 3], y_label="mse", title="Val MSE", color='green')
    plot_metric (r2_list[:, 3], y_label="r2", title="Val R²", color='blue')


def train_model_classification_main (train_csv, val_csv, model_out, batch_size=32, seq_length=32,
                                     input_dim=33, model_dim=64, num_heads=4, num_layers=2,
                                     output_dim=3, dropout=0.2, epochs=30, lr=1e-4, patience=5):
    """
    修改点：这是针对分类任务的训练主函数
    假设分类任务的目标列名为 'label'，且标签为整数
    """
    print ("=" * 10 + " 初始化分类模型中... " + "=" * 10)

    # 修改点：构造用于分类的模型，output_dim 表示类别数（例如本例中设为 3）
    model = TimeSeriesTransformer_classify (
        input_dim=input_dim,
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        seq_length=seq_length,
        output_dim=output_dim
    )

    print ("=" * 10 + " 加载训练数据中... " + "=" * 10)
    train_df = safeLoadCSV (pd.read_csv (train_csv))
    x, y, _, _ = create_sequences (train_df, seq_length=seq_length,
                                   target_cols=['open_label', 'high_label', 'low_label', 'close_label'], scale=False)

    # 修改点：将标签转换为整数类型（LongTensor）用于交叉熵损失
    # y = y.long ()
    train_dataset = TensorDataset (x, y)

    print ("=" * 10 + " 加载验证数据中... " + "=" * 10)
    val_df = pd.read_csv (val_csv)
    x_val, y_val, _, _ = create_sequences (val_df, seq_length=seq_length,
                                           target_cols=['open_label', 'high_label', 'low_label', 'close_label'],
                                           scale=False)
    # y_val = y_val.long ()  # 转换标签类型
    val_dataset = TensorDataset (x_val, y_val)

    print ("=" * 10 + " 开始训练分类模型... " + "=" * 10)
    # 修改点：调用模型中适用于分类任务的训练函数，不传入 scaler 和 target_indices
    # 此处假设 TimeSeriesTransformer 内部已实现适用于分类任务的 train_model 方法，
    # 例如使用 CrossEntropyLoss 及准确率评估指标
    train_loss, val_accuracy = model.train_model (
        train_dataset,
        val_dataset=val_dataset,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        patience=patience
    )

    # 创建保存目录
    os.makedirs (os.path.dirname (model_out), exist_ok=True)
    print ("=" * 10 + " 保存分类模型... " + "=" * 10)
    torch.save (model.state_dict (), model_out)

    print ("=" * 10 + " 绘图中... " + "=" * 10)
    # 修改点：绘制训练损失和验证准确率
    plot_metric (train_loss, y_label="loss", title="Train Loss", color='red')
    plot_metric (val_accuracy, y_label="accuracy", title="Val Accuracy", color='blue')


if __name__ == "__main__":
    parser = argparse.ArgumentParser (description="训练时间序列Transformer模型")
    # 增加任务类型参数，用于选择回归或分类任务
    parser.add_argument ("--task", type=str, default="regression", choices=["regression", "classification"],
                         help="任务类型：回归或分类")
    parser.add_argument ("--train", type=str, required=True, help="训练数据 CSV 路径")
    parser.add_argument ("--val", type=str, required=True, help="验证数据 CSV 路径")
    parser.add_argument ("--model_out", type=str, required=True, help="保存模型路径")
    parser.add_argument ("--bias_Corrector_out", type=str, help="保存bias corrector路径 (回归任务使用)")
    # 对于回归任务需要保存Scaler，分类任务不需要
    parser.add_argument ("--scaler_out", type=str, help="保存Scaler路径 (回归任务使用)")
    parser.add_argument ("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument ("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument ("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument ("--sequential_length", type=int, default=32, help="序列长度")
    parser.add_argument ("--input_dim", type=int, required=True, help="模型输入维度")
    parser.add_argument ("--output_dim", type=int, required=True, help="模型输出维度（回归：目标数；分类：类别数）")
    parser.add_argument ("--model_dim", type=int, required=True, help="模型映射维度")
    parser.add_argument ("--num_layers", type=int, required=True, help="模型层数")
    parser.add_argument ("--num_heads", type=int, required=True, help="注意力头数")
    parser.add_argument ("--dropout", type=float, required=True, help="Dropout")
    parser.add_argument ("--patience", type=int, default=5, help="Early stopping的等待轮数")

    args = parser.parse_args ()

    if args.task == "regression":
        # 回归任务时，scaler_out 参数必须提供
        if args.scaler_out is None:
            raise ValueError ("回归任务需要提供 --scaler_out 参数")
        train_model_main (
            train_csv=args.train,
            val_csv=args.val,
            model_out=args.model_out,
            scaler_out=args.scaler_out,
            bias_corrector_out=args.bias_Corrector_out,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            seq_length=args.sequential_length,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            model_dim=args.model_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            patience=args.patience,
        )
    elif args.task == "classification":
        train_model_classification_main (
            train_csv=args.train,
            val_csv=args.val,
            model_out=args.model_out,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            seq_length=args.sequential_length,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            model_dim=args.model_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            patience=args.patience,
        )
