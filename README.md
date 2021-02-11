<br />
<p align="center">
  <h1 align="center">Contrastive Tension</h1>
  <h3 align="center">State of the art Semantic Sentence Embeddings</h3>
  
  <p align="center">  
    <a href="https://openreview.net/pdf?id=Ov_sMNau-PF">Published Paper</a>
    ·
    <a href="https://huggingface.co/Contrastive-Tension">Huggingface Models</a>
    ·
    <a href="https://github.com/FreddeFrallan/Contrastive-Tension/issues">Report Bug</a>
  </p>
</p>


<!-- ABOUT THE PROJECT -->
## Overview
This is the official code accompanied with the paper [Semantic Re-Tuning via Contrastive Tension](https://openreview.net/pdf?id=Ov_sMNau-PF).</br>
The paper was accepted at ICLR-2021 and official reviews and responses can be found at [OpenReview](https://openreview.net/forum?id=Ov_sMNau-PF).

Contrastive Tension(CT) is a fully self-supervised algorithm for re-tuning already pre-trained transformer Language Models, and achieves State-Of-The-Art(SOTA) sentence embeddings for Semantic Textual Similarity(STS). All that is required is hence a pre-trained model and a modestly large text corpus. The results presented in the paper sampled text data from Wikipedia, where it used a maximum of 1.6M unique sentences.

This repository contains:
* Tensorflow 2 implementation of the CT algorithm
* State of the art pre-trained STS models
* Tensorflow 2 inference code
* PyTorch inference code

### Requirements
While it is possible that other versions works equally fine, we have worked with the following:

* Python = 3.6.9
* Transformers = 4.1.1

<!-- GETTING STARTED -->
## Pre-trained Models
Note that these models are <b>not</b> trained with the exact hyperparameters as those disclosed in the original CT paper. Rather, the parameters are from a short follow-up paper currently under review.

All evaluation is done using the [SentEval](https://github.com/facebookresearch/SentEval) framework, and shows the: (Pearson / Spearman) correlations
### Unsupervised / Zero-Shot
As both the training of BERT, and CT itself is fully self-supervised, the models only tuned with CT require no labeled data whatsoever.<br>
The NLI models however, are first fine-tuned towards a natural language inference task, which requires labeled data.

| Model| Avg Unsupervised STS |STS-b | #Parameters|
| ----------------------------------|:-----: |:-----: |:-----: |
|**Fully Unsupervised**    ||
| [BERT-Distil-CT](https://huggingface.co/Contrastive-Tension/BERT-Distil-CT)             | 75.12 / 75.04| 78.63 / 77.91 | 66 M|
| [BERT-Base-CT](https://huggingface.co/Contrastive-Tension/BERT-Base-CT)  | 73.55 / 73.36 | 75.49 / 73.31 | 108 M|
| [BERT-Large-CT](https://huggingface.co/Contrastive-Tension/BERT-Large-CT)        | 77.12 / 76.93| 80.75 / 79.82 | 334 M|
|**Using NLI Data**    ||
| [BERT-Distil-NLI-CT](https://huggingface.co/Contrastive-Tension/BERT-Distil-NLI-CT)             | 76.65 / 76.63 | 79.74 / 81.01 | 66 M|
| [BERT-Base-NLI-CT](https://huggingface.co/Contrastive-Tension/BERT-Base-NLI-CT)  | 76.05 / 76.28 | 79.98 / 81.47  | 108 M|
| [BERT-Large-NLI-CT](https://huggingface.co/Contrastive-Tension/BERT-Large-NLI-CT)        | <b> 77.42 / 77.41 </b> | <b> 80.92 / 81.66 </b>  | 334 M|

### Supervised
These models are fine-tuned directly with STS data, using the training object proposed by [S-BERT](https://arxiv.org/abs/1908.10084) using their [code](https://arxiv.org/abs/1908.10084).<br>
To our knowledge our RoBerta-Large-STSb is the current SOTA model for STS via sentence embeddings.

| Model| STS-b | #Parameters|
| ----------------------------------|:-----: |:-----: |
| BERT-Distil-CT-STSb             | 84.85 / 85.46  | 66 M|
| BERT-Base-CT-STSb  | 85.31 / 85.76  | 108 M|
| BERT-Large-CT-STSb        | 85.86 / 86.47  | 334 M|
| RoBerta-Large-CT-STSb        | <b> 87.56 / 88.42 </b>  | 334 M|



<!-- USAGE EXAMPLES -->
## Usage


<!-- LICENSE -->
## License
Distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact
If you have questions regarding the paper, please consider creating a comment via the official [OpenReview submission](https://openreview.net/forum?id=Ov_sMNau-PF). </br>
If you have questions regarding the code or otherwise related to this Github page, please open an [issue](https://github.com/FreddeFrallan/Contrastive-Tension/issues).

For other purposes, feel free to contact me directly at: Fredrk.Carlsson@ri.se

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [SentEval](https://github.com/facebookresearch/SentEval)
* [Huggingface](https://huggingface.co/)
* [Sentence-Transformer](https://github.com/UKPLab/sentence-transformers)
* [Best Readme Template](https://github.com/othneildrew/Best-README-Template)


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
