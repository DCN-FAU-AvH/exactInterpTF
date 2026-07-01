<!-- Title -->
<h1 align="center">
  Exact Sequence Interpolation with Transformers
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2502.02270">
    <img src="https://img.shields.io/badge/arXiv-2502.02270-b31b1b.svg" alt="arXiv:2502.02270">
  </a>
</p>

`Python` code for the paper 
**Exact Sequence Interpolation with Transformers** by Albert Alcalde, Giovanni Fantuzzi, and Enrique Zuazua. 

<p align="center">
  <img src="figs/figure3_thumbnail.svg" width="750"/>
</p>

## Abstract

We prove that transformers can exactly interpolate datasets of finite input sequences in $\mathbb{R}^d$, $d\geq 2$, with corresponding output sequences of smaller or equal length. Specifically, given $N$ sequences of arbitrary but finite lengths in $\mathbb{R}^d$ and output sequences of lengths $m^1, \dots, m^N \in \mathbb{N}$, we construct a transformer with $\mathcal{O}(\sum_{j=1}^N m^j)$ blocks and $\mathcal{O}(d \sum_{j=1}^N m^j)$ parameters that exactly interpolates the dataset.
Our construction provides complexity estimates that are independent of the input sequence length, by alternating feed-forward and self-attention layers and by capitalizing on the clustering effect inherent to the latter. Our novel constructive method also uses low-rank parameter matrices in the self-attention mechanism, a common feature of practical transformer implementations. 
These results are first established in the hardmax self-attention setting, where the geometric structure permits an explicit and quantitative analysis, and are then extended to the softmax setting. Finally, we demonstrate the applicability of our exact interpolation construction to learning problems, in particular by providing convergence guarantees to a global minimizer under regularized training strategies. Our analysis contributes to the theoretical understanding of transformer models, offering an explanation for their excellent performance in exact sequence-to-sequence interpolation tasks.

## Citation

```bibtex
@article{alcalde2026exact,
  title={Exact sequence interpolation with transformers},
  author={Alcalde, Albert and Fantuzzi, Giovanni and Zuazua, Enrique},
  journal={Mathematical Foundations of Machine Learning},
  volume={2},
  number={1},
  pages={2},
  year={2026},
  publisher={Springer}
}
```

## Funding & Acknowledgments
Funded by the European Union’s Horizon Europe MSCA project [ModConFlex](https://modconflex.uni-wuppertal.de/en/) (grant number 101073558).

<img src="./logos/logo-01.jpg" alt="ModConFlex" height="64"/> <img src="./logos/logo-02.png" alt="Funded by the EU" height="64"/>

