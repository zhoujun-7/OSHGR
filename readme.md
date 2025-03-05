# Hand Gesture Recognition from an Open-Set Perspective

This is the official repository for the paper "Hand Gesture Recognition from an Open-Set Perspective".


## Installation

```bash
conda env create -f environment.yml
conda activate oshgr
```

## OHG dataset

Click [here](https://drive.google.com/file/d/1rXRN4GVOqtKDk0UnHRs5P70hJQ7Hayg9/view?usp=sharing) to download the pre-processed OHG dataset.

### MANO for OHG dataset
```
cd GestureMano
python show.py
```

## Checkpoint

Click [here](https://drive.google.com/file/d/11Tl0g_VjV8GY0eFj9i1e1EO5D5j8olCw/view?usp=sharing) to download the checkpoint. Put the checkpoint in the `checkpoint` folder.

## Few-shot incremental Learning novel gesture class
```bash
python main_class.py
```

### Few-shot incremental Learning novel hand shape

```bash
python main_shape.py
```


If you find this work useful, please consider citing:

```bibtex
@ARTICLE{TMM25_OSHGR,
  author={Zhou, Jun and Xu, Chi and Cheng, Li},
  journal={IEEE Transactions on Multimedia (TMM)}, 
  title={Hand Gesture Recognition from an Open-Set Perspective}, 
  year={2025},

```