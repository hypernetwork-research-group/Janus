![JANUS logo](/assets/figures/janus-logo.png)

This repository contains a Python package that provides the following code:

- 🔥 PyTorch implementation of HyDRA $\beta\text{-VAE}$ and HyDRA DDM;
- ⚡️ Hydra-* training script using [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/);
- 👨‍💻 Hypergraph Analysis script to compare generated vs target dataset hypergraphs;
- 📄 Results report script to construct table and charts from generation results;
- ⌨️ A simple command line interface that can be used to launch training and sampling scripts. 

## Setup

```bash
pip install git+https://github.com/daqh/HyDRA
```

**Requirements**. Additionally to the code provided in this repository, you should install `torch` and `lightning`.

## Training

```bash
hydra [DATASET] <OPTIONS> train <OPTIONS> bvae <OPTIONS> --vertex-encoding/--no-vertex-encoding
```

and then

```
hydra [DATASET] train <OPTIONS> ddm <OPTIONS> --bvae-ckpt
```

### Model size

The size of both $\beta\text{-VAE}$ and DDM can be set using the `--model-size` option of the `hydra train` command. The following models sizes are available.

<table>
  <caption>
    Table 1. Model sizes available for the BETA-VAE.
  </caption>
  <thead>
    <tr>
      <td>Size</td>
      <td># Layers</td>
      <td># Heads</td>
      <td>Hidden size</td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>S</td>
      <td>1</td>
      <td>2</td>
      <td>128</td>
    </tr>
    <tr>
      <td>M</td>
      <td>2</td>
      <td>4</td>
      <td>256</td>
    </tr>
    </tr>
  </tbody>
</table>

<table>
  <caption>
    Table 2. Model sizes available for the DDM.
  </caption>
  <thead>
    <tr>
      <td>Size</td>
      <td># Layers</td>
      <td># Heads</td>
      <td>Hidden size</td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>S</td>
      <td>4</td>
      <td>2</td>
      <td>256</td>
    </tr>
    <tr>
      <td>M</td>
      <td>8</td>
      <td>3</td>
      <td>512</td>
    </tr>
    </tr>
  </tbody>
</table>

### Model hyperparameters

We trained both models fixing the following hyperparameters.

- Shared Hyperparameters
  - batch size 32
  - samples per hyperedge 2
  - biased random walk p 2.0
  - biased random walk q 0.5
- $\beta\text{-VAE}$
  - patience 100
- DDM
  - training timesteps 1000

The learning rate is determined using a learning rate finder strategy.

## Sampling

## Analysis

![Generated Hypergraph Evaluation Metrics](/assets/figures/generated-hypergraph-evaluation-metrics.png)

```bash
hydra-stats analysis <OPTIONS>
```

## Compare

First of all you need to download the datasets and analyze them:

```bash
hydra-stats parse [DATASET_NAME]
hydra-stats analyze --root-dir references
```

At this point, you can perform analysis over all hypergraphs in the `samples/` directory:

```bash
hydra-stats analyze
```

Finally, run the comparison script to compare each sample with its relative reference.

```bash
hydra-stats compare
```

## Reproducibility

### Training

In order to run the training configuration we used in our paper, run the following commands to train both models:

```bash
hydra --batch-size 32 [DATASET] train --model-size S bvae
hydra --batch-size 32 --val-size 0 [DATASET] train --model-size S --samples-per-hyperedge 2 ddm --bvae-ckpt logs/[DATASET]/BVAE-HyDRA-S/logs/version_0/checkpoints/best.ckpt
```

### Sampling

```bash
hydra --batch-size 32 [DATASET] sample --ckpt logs/[DATASET]/logs/version_0/DDM-HyDRA-S/checkpoints/last.ckpt ddm
```

## Datasets

<table>
  <thead>
    <tr>
      <td>Dataset</td>
      <td># Nodes</td>
      <td># Hyperedges</td>
      <td># Connected Components</td>
      <td>LCC Ratio</td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://huggingface.co/datasets/daqh/email-Enron" target="_blank">daqh/email-Enron</a></td>
      <td>143</td>
      <td>1,512</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/datasets/daqh/email-Eu" target="_blank">daqh/email-Eu</a></td>
      <td>998</td>
      <td>25,791</td>
      <td>20</td>
      <td>0.9809619238476954</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/datasets/daqh/contact-high-school" target="_blank">daqh/contact-high-school</a></td>
      <td>327</td>
      <td>7,937</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/datasets/daqh/contact-primary-school" target="_blank">daqh/contact-primary-school</a></td>
      <td>242</td>
      <td>12,799</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/datasets/daqh/NDC-classes" target="_blank">daqh/NDC-classes</a></td>
      <td>1,161</td>
      <td>1,222</td>
      <td>183</td>
      <td>0.540913006029285</td>
  </tbody>
</table>

### Dataset structure

data/
  full-00000-of-00001.jsonl

When running the training script `full` is the default split for train/val/predict (the predict split is only used when performing sampling from the $\beta\text{-VAE}$).
If the split is equal for train/val, the `--val-size` option is used to perform train/val split across that dataset, otherwise this option is ignored.

```bash
hydra daqh/email-Enron train --train-split train --val-split val --predict-split predict bvae
```

### Use different datasets

Our pipeline is able to handle single and multi-hypergraph datasets retrieved from huggingface.
In order to work, an hypergraph dataset must be stored in a `.jsonl` file, where each entry is a JSON following the [HIF standard](https://github.com/HIF-org/HIF-standard):

```jsonl
{
  "network-type": ...,
  "metadata": ...,
  "incidences": [...],
  "nodes": [...],
  "edges": [...]
}
```

In order to provide nodes and edges features, these must be set in the `attrs` dictionary of each node/edge.
