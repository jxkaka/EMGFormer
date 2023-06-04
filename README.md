# EMGFormer
This is the source code for "Movement Recognition via Channel-Activation-Wise sEMG Attention"


# Movement Recognition via Channel-Activation-Wise sEMG Attention

Jiaxuan Zhanga, Yuki Matsuda, Manato Fujimoto, Hirohiko Suwa and Keiichi Yasumotoa

Nara Institute of Science and Technology (NAIST), Ikoma, Nara 630-0192, 

Japan Osaka Metropolitan University, Osaka, Osaka 558-8585, Japan


This work proposed a novel 3-view sEMG feature representation enforced by conventional feature processing as well as the state-of-the-art Transformer.

<img width="532" alt="image" src="https://github.com/jxkaka/EMGFormer/assets/135442676/8aa2ed54-8b64-4c2a-9667-2f95c19967dc">


Traning and validation loss.

<img width="415" alt="image" src="https://github.com/jxkaka/EMGFormer/assets/135442676/8242f1f2-5236-41c9-a393-7f6c8f58afd1">



## Setup

You can install the required dependencies using pip.

```bash
pip install -r requirements.txt
```

If you're using other than CUDA 10.2, you may need to install PyTorch for the proper version of CUDA. See [instructions](https://pytorch.org/get-started/locally/) for more details.

## Description

The main file of the mpde; can be found in _EMG_Trans.ipynb_.

Data propocessing can be found in MATLAB file _gait_data_test.m_ and ipython file _raw_data_process_all.ipynb_

MATLAB file is used to extract different statistical features (e.g., Mean Absolute Value, Maximum Fractal Length) from the "Dual-Tree Complex Wavelet Transform".


## Reference

Now our paper is accepted, we will release the Reference after the preprint.
