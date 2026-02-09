
## Training

```bash
hydra train [DATASET] bvae --vertex-encoding/--no-vertex-encoding
```

and then

```
hydra train [DATASET] ddm [bvae_version]
```

### Model size

The size of both $\beta\text{-VAE}$ and DDM can be set using the `--model-size` option of the `hydra train` command. The following models sizes are available.

<table>
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
    <tr>
      <td>L</td>
      <td>4</td>
      <td>8</td>
      <td>512</td>
    </tr>
  </tbody>
</table>

## Datasets

<table>
  <thead>
    <tr>
      <td>Dataset</td>
      <td># Nodes</td>
      <td># Hyperedges</td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://huggingface.co/datasets/daqh/email-Enron">daqh/email-Enron</a></td>
      <td>143</td>
      <td>1512</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/datasets/daqh/email-Eu">daqh/email-Eu</a></td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/datasets/daqh/contact-high-school">daqh/contact-high-school</a></td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/datasets/daqh/contact-primary-school">daqh/contact-primary-school</a></td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/datasets/daqh/NDC-classes">daqh/NDC-classes</a></td>
      <td>...</td>
      <td>...</td>
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
