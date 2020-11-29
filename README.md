Adaptive video highlight detection

### Prerequistes
- Python 3.6.9
- PyTorch 1.2.0

### Dataset and setup
- Download and process the [PHD-GIFs](https://github.com/gifs/personalized-highlights-dataset) dataset.
- See `config/config.py` for different experiment settings and parameters.
- Fix the paths in `config/config.py`. We store the list of train/val/test users and their video and histories path in .json files.
- For each user, we store the video features in a .csv  file and the user's history features in a .json file. For each element in the user's history, we consider the features of the segments that are indicated as highlights in the ground truth. See `dataloader/make_dataloader_final_dumps.py` for details and update the paths for .csv and .json files for users.
- Note that this codebase is a reimplementation. It is very likely that I may have made some mistakes during the process. However, I intend to fix them over time.
- Below I provide example training and testing commands.

### Training
To train, run the following command:
```bash
python train.py --hist_net attn
```

### Testing
To test, run the following command:
```bash
python test.py --hist -m ./checkpoints/adain-attn/checkpoint.pt --hist_net attn
```
