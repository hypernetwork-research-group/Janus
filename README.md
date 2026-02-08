
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

## Use different datasets

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

### Dataset structure

data/
  full-00000-of-00001.jsonl

When running the training script `full` is the default split for train/val/predict (the predict split is only used when performing sampling from the $\beta\text{-VAE}$).
If the split is equal for train/val, the `--val-size` option is used to perform train/val split across that dataset, otherwise this option is ignored.

```bash
hydra train --train-split train --val-split val --predict-split predict daqh/email-Enron bvae
```
