# HiCo: Hierarchical Contrastive Learning for Ultrasound Video Model Pretraining


### Abstract
The self-supervised ultrasound (US) video model pretraining can use a small amount of labeled data to achieve one of the most promising results on US diagnosis. However, it does not take full advantage of multi-level knowledge for learning deep neural networks (DNNs), and thus is difficult to learn transferable feature representations. This work proposes a hierarchical contrastive learning (HiCo) method to improve the transferability for the US video model pretraining. HiCo introduces both peer-level semantic alignment and cross-level semantic alignment to facilitate the interaction between different semantic levels, which can effectively accelerate the convergence speed, leading to better generalization and adaptation of the learned model. Additionally, a softened objective function is implemented by smoothing the hard labels, which can alleviate the negative effect caused by local similarities of images between different classes. Experiments with HiCo on five datasets demonstrate its favorable results over state-of-the-art approaches.

![image](https://github.com/983632847/HiCo/blob/main/figs/HiCo.jpg)

### Quick Start

#### Fine-tune with Pretrained Model
1. Pick a model and its config file, for example, `config.yaml`
2. Download the model [HiCo](https://drive.google.com/file/d/1rfQA5vNrxhj31Ttud3zVMRmH3d4lsbgD/view?usp=sharing)
3. Download the 5 fold cross validation [POCUS](https://drive.google.com/file/d/111lHpStoY_gYMhCQ-Yt95AreDx0G7-2R/view?usp=sharing) dataset
4. Run the demo with
```
python eval_pretrained_model.py
```


#### Train Your Own Model
1. Download the Butterfly ([Baidu pan](https://pan.baidu.com/s/1tQtDzoditkTft3LMeDfGqw) Pwd:butt, [Google drive](https://drive.google.com/file/d/1zefZInevopumI-VdX6r7Bj-6pj_WILrr/view?usp=sharing)) dataset 
2. Train the HiCo model with
```
python run.py
```


#### Environment
The code is developed with a single Nvidia RTX 3090 GPU.

The install script has been tested on an Ubuntu 18.04 system.

:cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud::cloud:

### License

Licensed under a GPL-3.0 license.

### Citation

If you find the code useful in your research, please consider citing:

    @inproceedings{Zhang2022ACCV,
        title={HiCo: Hierarchical Contrastive Learning for Ultrasound Video Model Pretraining},
        author = {Chunhui Zhang, and Yixiong Chen, and Li Liu, and Qiong Liu, and Xi Zhou},
        journal = {ACCV},
        year = {2022}
      }
      
      
     @inproceedings{Chen2021MICCAI,
        title={USCL: Pretraining Deep Ultrasound Image Diagnosis Model through Video Contrastive Representation Learning},
        author = {Yixiong Chen, and Chunhui Zhang, and Li Liu, and Cheng Feng, and Changfeng Dong, and Yongfang Luo, and Xiang Wan},
        journal = {MICCAI},
        year = {2021}
      }

### Contact
Feedbacks and comments are welcome! Feel free to contact us via [andyzhangchunhui@gmail.com](mailto:andyzhangchunhui@gmail.com) or [16307110231@fudan.edu.cn](mailto:16307110231@fudan.edu.cn) or [liuli@cuhk.edu.cn](mailto:liuli@cuhk.edu.cn).

Enjoy!

Our code is largely borrowed from [USCL](https://github.com/983632847/USCL)

