# Multimodal Transformer-based Classifier for Deepfake Detection

## Abstract
The rise of generative models capable of manipulating both visual and audio content has led to a surge in deepfake media. Because of this there is a need to build robust detection frameworks. Traditional detection systems tend to be unimodal, focusing on visual artifacts or audio cues in isolation. However, real-world deepfakes increasingly manipulate both modalities simultaneously, introducing cross-modal inconsistencies that go undetected by unimodal systems. Our project proposes a multimodal deepfake detection architecture inspired by the Audio-Visual Fusion Framework (AVFF), which leverages both audio and video streams for classification. Our main experimentation is with our use of a complementary masking and cross-modal reconstruction strategy, where masked audio embeddings are predicted using unmasked video inputs and vice versa. This forces the system to learn the correspondence between modalities. These reconstructed embeddings are concatenated with their original counterparts and fused temporally before being passed to a binary classifier. We evaluate our model Deepfake-Eval-2024, a dataset curated for the detection of modern, cross-modality synthetic content. Our model achieves 92.3 % classification accuracy on average and 85.82% ROC-AUC score on average, showing improvements over unimodal baselines. The results validate our hypothesis that modeling intermodal dependencies enables more robust detection of audiovisual inconsistencies, which are indicative of deepfakes.

## Repository
1. `Reorganized.ipynb`
3. `eval.py`
4. `stored_features.csv`
5. `video-metadata-publish.csv`

## Execution Guide
### `Reorganized.ipynb`
This file contains all major code we used to load and preprocess data, import pretrained encoders, build our model architecture, and replicate our training and evaluation loop. To run our main development pipeline simply run the cells in this notebook in order. The libraries we used are stable with Python 3.11 and `torchcodec` version 0.3.0. All other libraries worked for us in their latest version.
```
pip install torchcodec==0.3.0
```
The data preprocessing uses [`video-metadata-publish.csv`](https://github.com/esha-shaik/deepfake-detection/blob/main/video-metadata-publish.csv), which contains the video file names, real/fake labels, and a short description of each videos content. These videos are from [Deepfake-Eval-2024](https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024), a gated dataset under a Creative Commons License (CC-BY-SA-4.0). Our team was granted permission to use this dataset for this project. [`stored_features.csv`](https://github.com/esha-shaik/deepfake-detection/blob/main/stored_features.csv) is also used during preprocessing as it contains the real/fake labels for a representative subset of the video files.

We use PyTorch and Torchvision Transforms to construct our A2V and V2A conversion networks and the final AVFFDecoder. [`eval.py`](https://github.com/esha-shaik/deepfake-detection/blob/main/eval.py) contains our evaluation metrics, with custom implementation in case of fine tuning or adjustment, which are then used to keep a benchmark of the evaluation during the final stage of traning the classifier network.
