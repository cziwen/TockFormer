from torch.utils.data import TensorDataset

from TransformerModel import *
from Util import *


# Model Process
print ("=" * 10 + " 创建/加载模型中... " + "=" * 10)
batch_size = 32
seq_length = 30
input_dim = 6

# 定义模型参数
model_dim = 64
num_heads = 4
num_layers = 2
output_dim = 4

# 构造模型实例
model = TimeSeriesTransformer (input_dim=input_dim,
                               model_dim=model_dim,
                               num_heads=num_heads,
                               num_layers=num_layers,
                               dropout=0.1,
                               seq_length=seq_length,
                               output_dim=output_dim)

# 首先，定义和初始化模型（参数要和保存时一致）
# 加载参数
model.load_state_dict(torch.load("model_params.pth"))


# Load Data
print ("=" * 10 + " 载入数据中... " + "=" * 10)
train_dataset_df = pd.read_csv ("data/train_data_5min.csv")
x, y = create_sequences (train_dataset_df, seq_length=seq_length, target_cols=['Close', 'Open', 'High', 'Low'])
train_dataset = TensorDataset (x, y)

val_dataset_df = pd.read_csv ("data/train_data_5min.csv")
x, y = create_sequences (val_dataset_df, seq_length=seq_length, target_cols=['Close', 'Open', 'High', 'Low'])
val_dataset = TensorDataset (x, y)

print ("=" * 10 + " 训练模型中... " + "=" * 10)
train_loss, mse, r2 = model.train_model (train_dataset, val_dataset=val_dataset, num_epochs=10,
                                               batch_size=batch_size, learning_rate=1e-3)

print ("=" * 10 + " 训练完成... " + "=" * 10)
plot_metric (train_loss, y_label="loss", title="Train Loss", color='red')
plot_metric (mse, y_label="mse", title="Val MSE", color='green')
plot_metric (r2, y_label="r2", title="Val r2", color='blue')


# 假设 model 是你的模型实例
torch.save(model.state_dict(), "model_params.pth")