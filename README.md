# Neural Image Compression Explaination

This reposityory contains the PyTorch implementation for [Neural Image Compression and Explanation](https://arxiv.org/abs/1908.08988)

<img src="https://github.com/lxuniverse/neural-image-compression-explain_nice/blob/master/pic/Structure.png" width="500" class="center">

## Expalanation Examples
Learned sparse explanation to LeNet-5 on MNIST datasets:

![demo](https://github.com/lxuniverse/neural-image-compression-explain_nice/blob/master/vis/masks.png)

## Requirements

  PyTorch >= 0.4.0
    

## Quick Start
Follow the below 3 steps to explore our algorithm:

### Train a target model to explain
```
python train_target_model.py 
```

### Train NICE

```
python main.py --r [1] 
```
-r: The hyperparameter to balance the data loss and spase loss. Please read our paper for details.
Please run r = 1, 5, 10, 15, 20 if you want to run next step to visualize results.

### Visualize results
```
python visualize_explanation.py
```

## Citation

If you found this code useful, please cite our paper.

```latex
@article{li2019nice,
  title={Neural Image Compression and Explanation},
  author={Xiang Li and Shihao Ji},
  journal={arXiv preprint arXiv:1908.08988},
  year={2019}
}
```
