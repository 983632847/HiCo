# HiCo: Hierarchical Contrastive Learning for Ultrasound Video Model Pretraining


### Abstract
The self-supervised ultrasound (US) video model pretraining can use a small amount of labeled data to achieve one of the most promising results on US diagnosis. However, it does not take full advantage of multi-level knowledge for learning deep neural networks (DNNs), and thus is difficult to learn transferable feature representations. This work proposes a hierarchical contrastive learning (HiCo) method to improve the transferability for the US video model pretraining. HiCo introduces both peer-level semantic alignment and cross-level semantic alignment to facilitate the interaction between different semantic levels, which can effectively accelerate the convergence speed, leading to better generalization and adaptation of the learned model. Additionally, a softened objective function is implemented by smoothing the hard labels, which can alleviate the negative effect caused by local similarities of images between different classes. Experiments with HiCo on five datasets demonstrate its favorable results over state-of-the-art approaches.

![image](https://github.com/983632847/HiCo/blob/main/figs/HiCo.jpg)
