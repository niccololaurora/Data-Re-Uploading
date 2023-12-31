# Data-Re-Uploading

Implementing the architecture described in [arXiv:1907.02085](https://arxiv.org/abs/1907.02085) with [Qibo](https://qibo.science/).

Implemented:

1. Circle geometry;
2. Tricrown geometry;

Missing implementations:

1. Weighted loss function;
2. Other geometries;

## Circle geometry

The universal quantum classifier, as proposed in the aforementioned paper, is tested by determining whether a point lies inside or outside a circle with a radius of sqrt(pi/2).

As shown in the following images (on the left the prediction before training and on the right after training) it improves its classification accuracy as the number of layers grow:

<br/>
- 1 Layer
<br/>

<div style="display:flex">
    <img src="results/circle/layers_1/l-bfgs-b/Trial_l_4000/Trial_before_training.png" width="400" />
    <img src="results/circle/layers_1/l-bfgs-b/Trial_l_4000/Trial_after_training.png" width="400" />
</div>

<br/>
- 2 Layers
<br/>

<div style="display:flex">
    <img src="results/circle/layers_2/l-bfgs-b/Trial_l_4000/Trial_before_training.png" width="400" />
    <img src="results/circle/layers_2/l-bfgs-b/Trial_l_4000/Trial_after_training.png" width="400" />
</div>

<br/>
- 3 Layers
<br/>

<div style="display:flex">
    <img src="results/circle/layers_3/l-bfgs-b/Trial_l_4000/Trial_before_training.png" width="400" />
    <img src="results/circle/layers_3/l-bfgs-b/Trial_l_4000/Trial_after_training.png" width="400" />
</div>

<br/>
- 4 Layers
<br/>

<div style="display:flex">
    <img src="results/circle/layers_4/l-bfgs-b/Trial_l_4000/Trial_before_training.png" width="400" />
    <img src="results/circle/layers_4/l-bfgs-b/Trial_l_4000/Trial_after_training.png" width="400" />
</div>

<br/>
- 8 Layers
<br/>

<div style="display:flex">
    <img src="results/circle/layers_8/l-bfgs-b/Trial_l_4000/Trial_before_training.png" width="400" />
    <img src="results/circle/layers_8/l-bfgs-b/Trial_l_4000/Trial_after_training.png" width="400" />
</div>

<br/>
- 10 Layers
<br/>

<div style="display:flex">
    <img src="results/circle/layers_10/l-bfgs-b/Trial_l_4000/Trial_before_training.png" width="400" />
    <img src="results/circle/layers_10/l-bfgs-b/Trial_l_4000/Trial_after_training.png" width="400" />
</div>
