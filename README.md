
## Training

```bash
hydra train [DATASET] bvae --vertex-encoding/--no-vertex-encoding
```

and then

```
hydra train [DATASET] ddm [bvae_version]
```

## Use different datasets

Our pipeline is able to handle single and multi-hypergraph datasets retrieved from huggingface.
In order to work, an hypergraph dataset must be stored in a `.jsonl` file, where each entry is a JSON with the following structure:

```json
{
  "hif": <HIF REPRESENTATION>
}
```

### Dataset structure

data/
  full-00000-of-00001.jsonl

When running the training script `full` is the default split for train/val/predict (the predict split is only used when performing sampling from the $\beta\text{-VAE}$ AutoEncoder).
If the split is equal for train/val, the `--val-size` option is used to perform train/val split across that dataset, otherwise this option is ignored.

```bash
hydra train --train-split train --val-split val --predict-split predict daqh/email-Enron bvae
```
