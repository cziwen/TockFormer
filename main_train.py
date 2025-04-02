from torch.utils.data import TensorDataset
from TransformerModel import *
from Util import *
import joblib

print ("=" * 10 + " 创建/加载模型中... " + "=" * 10)

# 超参数设置
batch_size = 32
seq_length = 32
input_dim = 33
model_dim = 64
num_heads = 4
num_layers = 2
output_dim = 4

# 创建模型实例
model = TimeSeriesTransformer (
    input_dim=input_dim,
    model_dim=model_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=0.2,
    seq_length=seq_length,
    output_dim=output_dim
)

# 加载训练数据
print ("=" * 10 + " 载入数据中... " + "=" * 10)
train_dataset_df = safeLoadCSV (pd.read_csv ("readyData/SPY_5min_train_f.csv"))
x, y, scaler_train, target_indices = create_sequences (train_dataset_df, seq_length=seq_length,
                                                       target_cols=['open', 'high', 'low', 'close'])
train_dataset = TensorDataset (x, y)

val_dataset_df = pd.read_csv ("readyData/SPY_5min_validate_f.csv")
x_val, y_val, _, _ = create_sequences (val_dataset_df, seq_length=seq_length,
                                       target_cols=['open', 'high', 'low', 'close'], scaler=scaler_train)
val_dataset = TensorDataset (x_val, y_val)

# 模型训练
print ("=" * 10 + " 训练模型中... " + "=" * 10)
train_loss, mse_list, r2_list = model.train_model (
    train_dataset,
    val_dataset=val_dataset,
    num_epochs=30,
    batch_size=batch_size,
    learning_rate=1e-4,
    scaler=scaler_train,
    target_indices=target_indices,
    patience=5,
)

print ("=" * 10 + " 训练完成... " + "=" * 10)

# 模型与Scaler保存
print ("=" * 10 + " 模型保存... " + "=" * 10)
torch.save (model.state_dict (), "models/model_34d_5min.pth")
joblib.dump (scaler_train, 'models/scaler_34d_5min.pkl')

print ("=" * 10 + " 绘制图片... " + "=" * 10)
mse_list = np.array(mse_list)
r2_list = np.array(r2_list)
# 可视化训练过程
plot_metric (train_loss, y_label="loss", title="Train Loss", color='red')
plot_metric (mse_list[:, 3], y_label="mse", title="Val MSE", color='green')
plot_metric (r2_list[:, 3], y_label="r2", title="Val r2", color='blue')

