<br />
<p align="center">
  <h1 align="center">Contrastive Tension</h1>
  <h3 align="center">State of the art Semantic Sentence Embeddings</h3>
  
  <p align="center">  
    <a href="https://huggingface.co/welcome">Huggingface Models</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#Overview">Overview</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## Overview
This is the official code accompanied with the paper [Semantic Re-Tuning via Contrastive Tension](https://openreview.net/pdf?id=Ov_sMNau-PF).</br>
The paper was accepted at ICLR-2021 and official reviews and responses can be found at [OpenReview](https://openreview.net/forum?id=Ov_sMNau-PF).

This repository contains:
* Tensorflow 2 implementation of the Contrastive Tension algorithm
* State of the art pre-trained STS models
* Tensorflow 2 inference code
* PyTorch inference code

### Requirements
* Python >= 3.6
* Transformers >= 4.1.1

<!-- GETTING STARTED -->
## Pre-trained Models
### Best Performing Models


| Model| Avg Unsupervised STS |STS-b | Parameters|
| ----------------------------------|:-----: |:-----: |:-----: |
|**Fully Unsupervised**    ||
| BERT-Distil-CT             | 75.12 / 75.04| 78.63 / 77.91 | 66 M|
| BERT-Base-CT  | 73.55 / 73.36 | 75.49 / 73.31 | 108 M|
| BERT-Large-CT        | 77.12 / 76.93| 80.75 / 79.82 | 334 M|
|**Using NLI Data**    ||
| BERT-Distil-NLI-CT             | 76.65 / 76.63 | 79.74 / 81.01 | 66 M|
| BERT-Base-NLI-CT  | 76.05 / 76.28 | 79.98 / 81.47  | 108 M|
| BERT-Large-NLI-CT        | <b> 77.42 / 77.41 </b> | <b> 80.92 / 81.66 </b>  | 334 M|


| Model| STS benchmark | Parameters|
| ----------------------------------|:-----: |:-----: |
| BERT-Distil-STSb             | 84.85 / 85.46  | 66 M|
| BERT-Base-STSb  | 85.31 / 85.76  | 108 M|
| BERT-Large-STSb        | 85.86 / 86.47  | 334 M|
| RoBerta-Large-STSb        | <b> 87.50 / 88.33 </b>  | 334 M|



<!-- USAGE EXAMPLES -->
## Usage


<!-- LICENSE -->
## License
Distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact
If you have questions regarding the paper, please consider creating a comment via the official [OpenReview submission](https://openreview.net/forum?id=Ov_sMNau-PF). </br>
If you have questions regarding the code or otherwise related to this Github page, please open an issue.

For other purposes, feel free to contact me directly at: Fredrk.Carlsson@ri.se

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
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
