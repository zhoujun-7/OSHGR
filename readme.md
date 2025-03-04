# Hand Gesture Recognition from an Open-Set Perspective

This is the official repository for the paper "Hand Gesture Recognition from an Open-Set Perspective".

We are working on cleaning the code and the dataset, which expected to be finished in few weeks.


## Installation

```bash
conda env create -f environment.yml
conda activate oshgr
```

## MANO for OHG dataset
```
cd GestureMano
python show.py
```

## Pretrain

```bash
python Pretrain.py
```

## Evaluation
### on Few-shot incremental learning under unconstrained viewpoint

```bash
python main.py
```


## TODO

- [ ] Clean the redundant codes
- [x] Upload mano parameters for OHG dataset.
- [ ] upload processed OHG dataset.



if you find this work useful, please consider citing:

```bibtex
@ARTICLE{TMM25_OSHGR,
  author={Zhou, Jun and Xu, Chi and Cheng, Li},
  journal={IEEE Transactions on Multimedia (TMM)}, 
  title={Hand Gesture Recognition from an Open-Set Perspective}, 
  year={2025},

```