## Overview

This is the PyTorch implementation of the paper "Dilated Convolution Based CSI Feedback Compression for Massive MIMO Systems"
(DCRNet). The work targets CSI feedback in FDD massive MIMO-OFDM systems, where downlink CSI must be estimated at the user
equipment and fed back to the base station, creating heavy overhead as antenna counts grow. DCRNet leverages dilated
convolutions to enlarge the receptive field without increasing kernel size, achieving accurate CSI reconstruction while
reducing computational complexity.

### Paper highlights
- Dilated convolutions enlarge the receptive field to capture wider spatial correlation without expensive large kernels.
- Asymmetric dilated encoder blocks (dilation rates 1/2/3) and multi-branch decoder blocks mitigate gridding effects and
  improve reconstruction quality.
- DCRNet-$1\times$ achieves the lowest FLOPs among lightweight models while maintaining strong NMSE performance.
- DCRNet-$10\times$ reaches near-SOTA NMSE with substantially fewer FLOPs than high-complexity baselines.

### Architecture summary
- **Input:** CSI in the angular-delay domain with shape $2 \times N_a \times N_t$ (real/imaginary channels).
- **Encoder:** A $5 \times 5$ head convolution followed by one encoder block (asymmetric $3 \times 3$ dilated convolutions
  with $d=1,2,3$ plus a parallel $3 \times 3$ convolution, concatenation, $1 \times 1$ fusion, and residual addition).
- **Compression:** Reshape and fully connected layers output a codeword with compression rate $\eta \in (0,1)$.
- **Decoder:** FC layers + reshape, a $5 \times 5$ head convolution, and two dilated decoder blocks with width expansion
  rate $\rho$ and grouped/asymmetric convolutions to recover the CSI.

### Experimental settings (paper)
- COST2100 dataset, indoor (5.3 GHz) and outdoor (300 MHz) scenarios.
- $N_t=32$ antennas, $N_c=1024$ sub-carriers, truncated to $2 \times 32 \times 32$ CSI in the angular-delay domain.
- Train/val/test splits: 100,000 / 30,000 / 20,000 samples.
- Training: Kaiming initialization, Adam optimizer, cosine annealing LR ($5e{-5}$ to $2e{-3}$), $T_w=30$, $T=2500$.

### Evaluation and results summary
- NMSE is used for reconstruction quality; FLOPs/parameters measure complexity.
- DCRNet-$1\times$ reduces FLOPs by 26%/21%/12% versus CsiNet/CRNet/ACRNet-$1\times$ at $\eta=1/4$ and achieves the best
  indoor NMSE among low-complexity models.
- DCRNet-$10\times$ provides near-SOTA NMSE with ~7M fewer FLOPs than CsiNet$+$ and ACRNet-$10\times$ at multiple
  compression rates.

## Requirements

To use this project, you need to ensure the following requirements are installed.

- Python >= 3.6
- PyTorch == 1.7.0
- tqdm
- colorama
## Dataset
The used COST2100 dataset can be found in the paper of  [CsiNet+](https://ieeexplore.ieee.org/abstract/document/8972904/) and the corresponding [GoogDrive](https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj).
## Train 
```python3
python main.py --gpu 0 --lr 2e-3 -v 5 --cr 4 --scenario "in" --expansion 1
```
or ignore the output 
```python3
nohup python main.py --gpu 0 --lr 2e-3 -v 5 --cr 4 --scenario "in" --expansion 1  > /dev/null 2>&1 &
```
The training logs and checkpoints are saved in ```./outputs```

## checkpoints
The checkpoints will be available in GoogleDrive soon. 
