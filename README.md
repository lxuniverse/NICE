# Neural Image Compression Explaination

This reposityory contains the PyTorch implementation of "Neural Image Compression and Explanation" (under review).

<img src="https://github.com/lxuniverse/neural-image-compression-explain_nice/blob/master/pic/Structure.png" width="500" class="center">

## Expalanation Examples
Learned sparse explanations from LeNet-5 on MNIST examples:

![demo](https://github.com/lxuniverse/neural-image-compression-explain_nice/blob/master/vis/masks.png)

## Requirements
PyTorch >= 0.4.0
    
## Quick Start
Follow the below 3 steps to run our algorithm:

1. Train a target model to explain
```
python train_target_model.py 
```

2. Train NICE

```
python main.py --r [1] 
```
-r: The hyperparameter to balance data loss and sparsity loss. Please read our paper for details.

3. Visualize results
```
python visualize_explanation.py
```

<!--## Citation
If you found this code useful, please cite our paper.

```latex
@article{li2019nice,
  title={Neural Image Compression and Explanation},
  author={Xiang Li and Shihao Ji},
  journal={arXiv preprint arXiv:1908.08988},
  year={2019}
}
```
-->
