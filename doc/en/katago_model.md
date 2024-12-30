# Using KataGo models in TamaGo

1. Copy the required files from KataGo. Refer to `katago/original_katago/README.md` for details.
2. Download "Raw Checkpoint" from [Networks for kata1](https://katagotraining.org/networks/) and extract model.ckpt.
3. Run TamaGo as follows:

```
python main.py --katago-model PATH/TO/model.ckpt --size 19
```
