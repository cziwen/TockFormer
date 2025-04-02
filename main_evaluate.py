import joblib
from torch.utils.data import TensorDataset
from TransformerModel import *
from Util import *
from Util import safeLoadCSV

print ("=" * 10 + " 创建/加载模型中... " + "=" * 10)

# 超参数设置
batch_size = 32
seq_length = 32
input_dim = 33
model_dim = 64
num_heads = 4
num_layers = 2
output_dim = 4


# 加载模型
model = TimeSeriesTransformer (
    input_dim=input_dim,
    model_dim=model_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=0.2,
    seq_length=seq_length,
    output_dim=output_dim
)
model.load_state_dict (torch.load ("models/model_34d_5min.pth"))
scaler_pre = joblib.load ("models/scaler_34d_5min.pkl")

# 加载测试数据
print ("=" * 10 + " 载入数据中... " + "=" * 10)
test_dataset_df = safeLoadCSV (pd.read_csv ("readyData/SPY_5min_test_f.csv"))
x_test, y_test, _, target_indices = create_sequences (test_dataset_df, seq_length=seq_length, scaler=scaler_pre,
                                                      target_cols=['open', 'high', 'low', 'close'])
test_dataset = TensorDataset (x_test, y_test)

# 模型评估
print ("=" * 10 + " 评价模型中... " + "=" * 10)
mse_list, r2_list, preds, actuals = model.evaluate_model (
    test_dataset,
    batch_size=batch_size,
    scaler=scaler_pre,
    target_indices=target_indices
)

print ("=" * 10 + " 评价完成... " + "=" * 10)
print (f'MSE: {mse_list}')
print (f'R2: {r2_list}')

# 可视化预测结果
curve_dict = {
    'Close_pred': preds[:, 3],
    'Close_actual': actuals[:, 3]
}
plot_multiple_curves (curve_dict, x_label='interval', y_label='price', title='Close Comparison')
