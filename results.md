# Experiment Results

| Exp | Architecture | Channels    | Best Val Acc | Best Epoch | Final Val Acc |
|-----|--------------|-------------|--------------|------------|---------------|
| A1  | CNN          | EEG         |    0.617     |     12     |     0.567     |
| A2  | CNN          | EEG+EOG     |    0.611     |     12     |     0.555     |
| A3  | CNN          | EEG+EOG+EMG |    0.638     |     10     |     0.582     |
| B   | CNN+LSTM     | EEG+EOG+EMG |    0.618     |     10     |     0.595     |
