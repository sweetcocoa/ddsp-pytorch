# CREPE Pitch Tracker (PyTorch) #

- Original Tensorflow Implementation : [https://github.com/marl/crepe](https://github.com/marl/crepe)

---
CREPE is a monophonic pitch tracker based on a deep convolutional neural network operating directly on the time-domain waveform input. CREPE is originally implemented with tensorflow, which is very inconvenient framework to use.


## Usage

```python
import crepe
import torch
device = torch.device(0)
cr = crepe.CREPE("full").to(device)
cr.predict("path/to/audio.file", "path/to/output/directory/", )
```

## WIP



