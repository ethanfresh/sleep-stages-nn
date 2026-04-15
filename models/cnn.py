from torch import nn

class SleepCNN(nn.Module):
  def __init__(self, n_classes = 5):
    super().__init__()
    self.features = nn.Sequential(
      nn.Conv1d(3, 32, kernel_size = 50, stride = 6),
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
      nn.MaxPool1d(2),
    )

    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(128 * 25, 256),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(256, n_classes),
    )

  def forward(self, x):
    return self.classifier(self.features(x))