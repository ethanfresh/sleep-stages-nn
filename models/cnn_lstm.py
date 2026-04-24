from torch import nn

class SleepCNNLSTM(nn.Module):
  def __init__(self, n_channels = 3, n_classes = 5):
    super().__init__()
    self.cnn = nn.Sequential(
      nn.Conv1d(n_channels, 32, kernel_size = 50, stride = 6),
      nn.BatchNorm1d(32),
      nn.ReLU(),
      nn.MaxPool1d(4),

      nn.Conv1d(32, 64, kernel_size = 8),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.MaxPool1d(2),

      nn.Conv1d(64, 128, kernel_size = 8),
      nn.BatchNorm1d(128),
      nn.ReLU(),
    )

    self.lstm = nn.LSTM(
      input_size = 128,
      hidden_size = 128,
      num_layers = 2,
      batch_first = True,
      dropout = 0.3,
      bidirectional = True
    )

    self.classifier = nn.Sequential(
      nn.Linear(128 * 2, 64),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(64, n_classes)
    )

  def forward(self, x):
    x = self.cnn(x)
    x = x.permute(0,2,1)
    out, _ = self.lstm(x)
    out = out[:, -1, :]
    return self.classifier(out) 