# bank shot predictor

predict whether a basketball bank shot goes in from how it hits the backboard via a small physics simulator plus. PyTorch.

## files
| `bank_shot_predict.ipynb` | predicts make or miss
| `hoopsim.py` | 2D shot simulator (gravity, drag, backboard and rim bounces with spin) |
| `generate_data.py` | runs the simulator and writes `shots.csv` |
| `shots.csv` | ~150k pre-generated bank shots, so you can start right away |

## TODO
add 3d + more physics :D

