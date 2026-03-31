from typing import Literal

import torch
import torch.nn as nn
from tqdm.rich import tqdm
import lightning as L
import numpy as np
import torch.nn.functional as F
import logging
from diffusers import DDPMScheduler, DDIMScheduler
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.cluster import KMeans
import xgi

from .parameter_initialization import init_hypergraph_encoder, init_hypergraph_decoder, init_dit_weights
from .components import DiT, HGAT, HypergraphDecoder
from .utils import batch_index_contrastive_loss
from .enums import ModelSize, DDM_CONFIGS, BVAE_CONFIGS

logger = logging.getLogger(__name__)

DEFAULT_LR = 1e-4

class HypergraphBetaVAE(L.LightningModule):

    def __init__(self,
                 x_kl_weight: float | None = None,
                 y_kl_weight: float | None = None,
                 learning_rate: float | None = None,
                 weight_decay: float | None = None,
                 model_size_config: str = ModelSize.M.value,
                 vertex_encoding: bool = True,
                 patience: int = 100,
                 num_node_features: int = 128,
                 num_hyperedge_features: int = 128,
                 latent_dim: int | None = None):
        super().__init__()
        self.x_kl_weight = x_kl_weight
        self.y_kl_weight = y_kl_weight
        self.learning_rate = learning_rate or DEFAULT_LR
        self.weight_decay = weight_decay or 0
        self.model_size_config = BVAE_CONFIGS[model_size_config]
        self.vertex_encoding = vertex_encoding
        self.patience = patience
        self.node_feature_dim = num_node_features
        self.hyperedge_feature_dim = num_hyperedge_features
        self.latent_dim = latent_dim or num_hyperedge_features # Latent dimension is the same as hyperedge feature dimension by default, but can be set to a different value for more compression
        self.save_hyperparameters()

    def configure_callbacks(self):
        return [
            ModelCheckpoint(
                filename="last",
                every_n_epochs=10, # TODO: Add option to save every n epochs and not only on improvement, to have more checkpoints for analysis. Add this to trainer configuration.
            ),
            ModelCheckpoint(
                mode="min",
                monitor="validation/loss",
                filename="best",
                save_top_k=1,
                every_n_epochs=10, # TODO: Add option to save every n epochs and not only on improvement, to have more checkpoints for analysis. Add this to trainer configuration.
            ),
            EarlyStopping(
                monitor="validation/loss",
                patience=self.patience,
                mode="min",
                check_on_train_epoch_end=False, # Check only at the end of validation
            ),
        ]

    def configure_model(self):
        # Encoder
        if self.vertex_encoding:
            self.x_encoder_mu = HGAT(
                in_channels=self.node_feature_dim,
                hidden_channels=self.model_size_config.hidden_dim,
                out_channels=self.latent_dim,
                num_layers=self.model_size_config.num_layers,
                heads=self.model_size_config.heads
            )
            self.x_encoder_log_var = HGAT(
                in_channels=self.node_feature_dim,
                hidden_channels=self.model_size_config.hidden_dim,
                out_channels=self.latent_dim,
                num_layers=self.model_size_config.num_layers,
                heads=self.model_size_config.heads
            )
        else:
            self.x_adapter = nn.Sequential(
                nn.LayerNorm(self.node_feature_dim, elementwise_affine=True),
                nn.Linear(self.node_feature_dim, self.latent_dim),
            )
        self.y_encoder_mu = HGAT(
            in_channels=self.hyperedge_feature_dim,
            hidden_channels=self.model_size_config.hidden_dim,
            out_channels=self.latent_dim,
            num_layers=self.model_size_config.num_layers,
            heads=self.model_size_config.heads
        )
        self.y_encoder_log_var = HGAT(
            in_channels=self.hyperedge_feature_dim,
            hidden_channels=self.model_size_config.hidden_dim,
            out_channels=self.latent_dim,
            num_layers=self.model_size_config.num_layers,
            heads=self.model_size_config.heads
        )
        # Decoder
        self.hypergraph_decoder = HypergraphDecoder(
            in_channels=self.latent_dim,
            num_classes=2
        )
        if self.vertex_encoding:
            self.node_features_decoder = HGAT(
                in_channels=self.latent_dim,
                hidden_channels=self.model_size_config.hidden_dim,
                out_channels=self.node_feature_dim,
                num_layers=self.model_size_config.num_layers,
                heads=self.model_size_config.heads
            )

        # Initialize parameters
        if self.vertex_encoding:
            init_hypergraph_encoder(self.x_encoder_mu)
            init_hypergraph_encoder(self.x_encoder_log_var)
        init_hypergraph_encoder(self.y_encoder_mu)
        init_hypergraph_encoder(self.y_encoder_log_var)
        init_hypergraph_decoder(self.hypergraph_decoder)
        if self.vertex_encoding:
            init_hypergraph_encoder(self.node_features_decoder)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
        return optimizer

    def configure_gradient_clipping(self, optimizer, gradient_clip_val = None, gradient_clip_algorithm = None):
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val or 1.0,
            gradient_clip_algorithm=gradient_clip_algorithm or "norm",
        )

    def on_train_epoch_end(self):
        if self.current_epoch % 50 == 0:
            # Log weights
            for name, param in self.named_parameters():
                self.logger.experiment.add_histogram(
                    tag=f"weights/{name}",
                    values=param,
                    global_step=self.current_epoch
                )

    def forward(self, x: torch.Tensor, y: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if self.vertex_encoding:
            x_mu = self.x_encoder_mu(x, h)
            x_log_var = self.x_encoder_log_var(x, h)
            x_z = x_mu + torch.exp(0.5 * x_log_var) * torch.randn_like(x_log_var)
        else:
            x_z = self.x_adapter(x)
            x_mu = None
            x_log_var = None

        dual_h = h.permute(0, 2, 1)
        y_mu = self.y_encoder_mu(y, dual_h)
        y_log_var = self.y_encoder_log_var(y, dual_h)
        y_z = y_mu + torch.exp(0.5 * y_log_var) * torch.randn_like(y_log_var)

        if self.vertex_encoding:
            x_r = self.node_features_decoder(x_z, h)
        else:
            x_r = None

        h_logits = self.hypergraph_decoder(x_z, y_z)
        return h_logits, x_r, x_z, y_z, x_mu, y_mu, x_log_var, y_log_var

    def training_step(self, batch, batch_idx):
        x = batch['node_features']          # [B, num_nodes, node_feature_dim]
        y = batch['hyperedge_features']     # [B, num_hyperedges, hyperedge_feature_dim]
        h = batch['incidence_matrix']       # [B, num_nodes, num_hyperedges]
        s = batch['touched_hyperedges']     # [B, num_hyperedges]
        m = batch['nodes_mask']             # [B, num_nodes]

        h_logits, x_r, _, _, x_mu, y_mu, x_log_var, y_log_var = self.forward(x, y, h)    # Encode

        # From here, x and y are in the encoded space

        if self.vertex_encoding:
            x_kl_loss = -0.5 * (1 + x_log_var - x_mu.pow(2) - x_log_var.exp()).sum(dim=(1, 2)).mean()
            self.log("training/x_kl_loss", x_kl_loss.item(), prog_bar=False, on_step=True, on_epoch=True)
        else:
            x_kl_loss = 0.0

        y_kl_loss = -0.5 * (1 + y_log_var - y_mu.pow(2) - y_log_var.exp()).sum(dim=(1, 2)).mean()
        self.log("training/y_kl_loss", y_kl_loss.item(), prog_bar=False, on_step=True, on_epoch=True)

        reconstruction_loss = F.cross_entropy(
            h_logits.permute(0, 3, 1, 2),
            h.long(),
            reduction="none"
        ).sum(dim=(1, 2)).mean()
        self.log("training/reconstruction_loss", reconstruction_loss.item(), prog_bar=False, on_step=True, on_epoch=True)

        if self.vertex_encoding:
            # Here the contrastive loss is computed over the reconstructed node features x_r
            # The mask is used to determine which nodes are real and which are padding, and the loss is only computed over the real nodes
            x_recon_loss = batch_index_contrastive_loss(x_r, m, temperature=1.0)
            self.log("training/x_contrastive_loss", x_recon_loss.item(), prog_bar=False, on_step=True, on_epoch=True)
        else:
            x_recon_loss = 0.0

        if self.vertex_encoding and self.global_step % 2:
            # Detach decoder 1
            x_recon_loss = x_recon_loss.detach()
        elif self.vertex_encoding:
            # Detach decoder 2
            reconstruction_loss = reconstruction_loss.detach()

        loss = reconstruction_loss + self.x_kl_weight * x_kl_loss + self.y_kl_weight * y_kl_loss + x_recon_loss
        self.log("training/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['node_features']
        y = batch['hyperedge_features']
        h = batch['incidence_matrix']
        s = batch['touched_hyperedges']
        m = batch['nodes_mask']             # [B, num_nodes]

        h_logits, x_r, _, _, x_mu, y_mu, x_log_var, y_log_var = self.forward(x, y, h)    # Encode

        # From here, x and y are in the encoded space

        if self.vertex_encoding:
            x_kl_loss = -0.5 * (1 + x_log_var - x_mu.pow(2) - x_log_var.exp()).sum(dim=(1, 2)).mean()
            self.log("validation/x_kl_loss", x_kl_loss.item(), prog_bar=False, on_step=False, on_epoch=True)
        else:
            x_kl_loss = 0.0

        y_kl_loss = -0.5 * (1 + y_log_var - y_mu.pow(2) - y_log_var.exp()).sum(dim=(1, 2)).mean()
        self.log("validation/y_kl_loss", y_kl_loss.item(), prog_bar=False, on_step=False, on_epoch=True)

        reconstruction_loss = F.cross_entropy(
            h_logits.permute(0, 3, 1, 2),
            h.long(),
            reduction="none"
        ).sum(dim=(1, 2)).mean()
        self.log("validation/reconstruction_loss", reconstruction_loss.item(), prog_bar=False, on_step=False, on_epoch=True)

        if self.vertex_encoding:
            x_recon_loss = batch_index_contrastive_loss(x_r, m, temperature=1.0)
            self.log("validation/x_contrastive_loss", x_recon_loss.item(), prog_bar=False, on_step=False, on_epoch=True)
        else:
            x_recon_loss = 0.0

        loss = reconstruction_loss + self.x_kl_weight * x_kl_loss + self.y_kl_weight * y_kl_loss + x_recon_loss
        self.log("validation/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def predict_step(self, batch, batch_idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        x = batch['node_features']
        y = batch['hyperedge_features']
        h = batch['incidence_matrix']
        s = batch['touched_hyperedges']

        h_logits, x_r, x_z, y_z, x_mu, y_mu, x_log_var, y_log_var = self.forward(x, y, h)    # Encode

        incidence_matrices = torch.distributions.Categorical(logits=h_logits).sample()

        return incidence_matrices, h_logits, x_r, x_z, y_z, x_mu, y_mu, x_log_var, y_log_var

from lightning.pytorch.callbacks.weight_averaging import EMAWeightAveraging
from .utils import min_snr_weighted_v_mse_loss

type SchedulerType = Literal["ddpm", "ddim"]

class DiffusionTransformer(L.LightningModule):

    def __init__(self,
                 T: int,
                 bvae_ckpt: str,
                 num_inference_steps: int | None = None,
                 inference_scheduler_type: SchedulerType = "ddpm",
                 learning_rate: float | None = None,
                 model_size_config: str = ModelSize.M.value,):
        super().__init__()
        self.bvae_ckpt = bvae_ckpt
        self.num_inference_steps = num_inference_steps or T
        self.inference_scheduler_type = inference_scheduler_type
        self.learning_rate = learning_rate or DEFAULT_LR
        self.model_size_config = DDM_CONFIGS[model_size_config]
        self.train_noise_scheduler = DDPMScheduler(
            num_train_timesteps=T,
            prediction_type="v_prediction",
            timestep_spacing="trailing",
            beta_schedule="linear",
            rescale_betas_zero_snr=True,
            clip_sample=False,
            thresholding=False,
        )
        if self.inference_scheduler_type == "ddpm":
            self.sampling_noise_scheduler = DDPMScheduler.from_config(self.train_noise_scheduler.config)
        elif self.inference_scheduler_type == "ddim":
            self.sampling_noise_scheduler = DDIMScheduler.from_config(self.train_noise_scheduler.config)
        else:
            raise ValueError(f"Invalid inference_scheduler_type: {self.inference_scheduler_type}. Must be one of {SchedulerType.__args__}")
        self.save_hyperparameters()

    def on_train_start(self):
        self.bvae.eval()  # keep in eval mode

    def on_train_epoch_start(self):
        self.bvae.eval()  # keep in eval mode

    def configure_callbacks(self):
        return [
            ModelCheckpoint(
                filename="last",
                every_n_epochs=10, # TODO: Add option to save every n epochs and not only on improvement, to have more checkpoints for analysis. Add this to trainer configuration.
            ),
            ModelCheckpoint(
                mode="min",
                monitor="training/loss",
                filename="best",
                save_top_k=1,
                every_n_epochs=10, # TODO: Add option to save every n epochs and not only on improvement, to have more checkpoints for analysis. Add this to trainer configuration.
            ),
            EMAWeightAveraging(
                decay=0.99,
                update_every_n_steps=1,
                update_starting_at_step=0,
                use_buffers=True,
            ),
        ]

    def configure_model(self):
        self.bvae = HypergraphBetaVAE.load_from_checkpoint(self.bvae_ckpt,
                                                           map_location="cpu",
                                                           weights_only=False).freeze()

        if self.bvae.vertex_encoding:
            self.vertices_dit = DiT(
                in_channels=self.bvae.latent_dim,
                hidden_channels=self.model_size_config.hidden_dim,
                num_blocks=self.model_size_config.num_layers,
                num_heads=self.model_size_config.heads,
                cross_attention=True
            )
        self.hyperedges_dit = DiT(
            in_channels=self.bvae.latent_dim,
            hidden_channels=self.model_size_config.hidden_dim,
            num_blocks=self.model_size_config.num_layers,
            num_heads=self.model_size_config.heads,
            cross_attention=True
        )

        if self.bvae.vertex_encoding:
            init_dit_weights(self.vertices_dit)
        init_dit_weights(self.hyperedges_dit)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.learning_rate,
                                      weight_decay=0)
        return optimizer

    def forward(self, z_x: torch.Tensor, z_y: torch.Tensor, t: torch.Tensor):
        if self.bvae.vertex_encoding:
            x_v_pred, _ = self.vertices_dit(z_x, t, z_y)
        else:
            x_v_pred = None
        y_v_pred, _ = self.hyperedges_dit(z_y, t, z_x)
        return x_v_pred, y_v_pred

    def training_step(self, batch, batch_idx):
        x = batch['node_features']
        y = batch['hyperedge_features']
        h = batch['incidence_matrix']

        B, N, C = x.size()
        _, M, _ = y.size()

        with torch.no_grad():
            _, _, x_z, y_z, _, _, _, _ = self.bvae.forward(x, y, h) # Encode

        t = torch.randint(0,
                          self.train_noise_scheduler.config.num_train_timesteps,
                          (B,),
                          device=self.device,
                          dtype=torch.long)

        if self.bvae.vertex_encoding:
            x_noise = torch.randn_like(x_z)
            x_t = self.train_noise_scheduler.add_noise(x_z, x_noise, t)
            x_target = self.train_noise_scheduler.get_velocity(x_z, x_noise, t) # v prediction
        else:
            x_t = x_z

        y_noise = torch.randn_like(y_z)
        y_t = self.train_noise_scheduler.add_noise(y_z, y_noise, t)
        y_target = self.train_noise_scheduler.get_velocity(y_z, y_noise, t) # v prediction
        x_v_pred, y_v_pred = self.forward(x_t, y_t, t.unsqueeze(-1))

        if self.bvae.vertex_encoding:
            # x_loss = F.mse_loss(x_v_pred, x_target)
            x_loss = min_snr_weighted_v_mse_loss(
                noise_scheduler=self.train_noise_scheduler,
                model_pred_v=x_v_pred,
                latents=x_z,
                noise=x_noise,
                timesteps=t,
                snr_gamma=5.0
            )
            self.log("training/x_loss", x_loss.item(), prog_bar=False, on_step=True, on_epoch=False)

        else:
            x_loss = 0.0

        # y_loss = F.mse_loss(y_v_pred, y_target)
        y_loss = min_snr_weighted_v_mse_loss(
            noise_scheduler=self.train_noise_scheduler,
            model_pred_v=y_v_pred,
            latents=y_z,
            noise=y_noise,
            timesteps=t,
            snr_gamma=5.0
        )

        # y_loss = F.mse_loss(y_v_pred, y_target)
        self.log("training/y_loss", y_loss.item(), prog_bar=False, on_step=True, on_epoch=False)
        loss = x_loss + y_loss
        self.log("training/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def predict_step(self, batch, batch_idx, walk_length: int, tau: float = 1.0):
        z_x_T = batch['node_features']

        B, N, F = z_x_T.size()

        z_y_T = torch.randn(B, int(walk_length), F, device=self.device)

        scheduler = self.sampling_noise_scheduler
        self.sampling_noise_scheduler.set_timesteps(self.num_inference_steps, device=self.device)

        if not self.bvae.vertex_encoding:
            z_x_T = self.bvae.x_adapter(z_x_T)

        for t in tqdm(scheduler.timesteps, leave=False):
            t = t.to(self.device)
            # x_v prediction is only used if nodes are encoded
            x_v_pred, y_v_pred = self(z_x_T, z_y_T, t.expand(B, 1))
            if self.bvae.vertex_encoding: # Reverse on nodes only if the model encodes them
                x_step_out = scheduler.step(x_v_pred, t, z_x_T)
                z_x_T = x_step_out.prev_sample
            y_step_out = scheduler.step(y_v_pred, t, z_y_T)
            z_y_T = y_step_out.prev_sample

        h_logits = self.bvae.hypergraph_decoder(z_x_T, z_y_T)  # Decode
        incidence_matrices = torch.distributions.Categorical(logits=h_logits / tau).sample() # Sample hard incidence matrices
        if self.bvae.vertex_encoding:
            x_rec = self.bvae.node_features_decoder(z_x_T, incidence_matrices)             # Produce node representations
        else:
            x_rec = None

        # This mask indicates hypergraph membership for each node
        membership_mask = incidence_matrices.sum(dim=2).bool() # [B, num_nodes]

        return incidence_matrices, h_logits, x_rec, membership_mask, z_x_T, z_y_T

    @torch.inference_mode()
    def sample_unconditional(self,
                             num_nodes: int,
                             num_hyperedges: int,
                             walk_length: int,
                             batch_size: int,
                             initial_tau: float = 1.0,
                             tau_multiplier: float = 1 + 1e-12):
        assert self.bvae.vertex_encoding, "Unconditional sampling is only supported when vertex encoding is enabled"
        kmeans = KMeans(n_clusters=num_nodes)

        embeddings = []
        membership_masks = []
        incidences = []

        B, F = batch_size, self.bvae.latent_dim

        tau = initial_tau
        with tqdm(total=num_hyperedges, desc="Sampling hyperedges", leave=False) as pbar:
            while True:
                z_x_T = torch.randn(B, num_nodes, F, device=self.device)
                incidence_matrices, _, x_rec, membership_mask, z_x_T, _ = self.predict_step(
                    batch={
                        'node_features': z_x_T,
                    },
                    batch_idx=0,
                    walk_length=walk_length,
                    tau=tau
                )
                embeddings.append(x_rec)
                membership_masks.append(membership_mask)
                incidences.append(incidence_matrices)
                # Assign cluster to each node based on kmeans clusters

                # Prepare data for kmeans
                np_embeddings = torch.cat(embeddings, dim=0).cpu().numpy()  # [B, num_nodes, F]
                np_embeddings = np_embeddings.reshape(-1, np_embeddings.shape[-1]) # [B * num_nodes, F]
                np_membership_masks = torch.cat(membership_masks, dim=0).cpu().float().numpy()  # [B, num_nodes]
                np_membership_masks = np_membership_masks.reshape(-1)  # [B * num_nodes]
                t_incidences = torch.cat(incidences, dim=0).cpu()  # [B, num_hyperedges, num_nodes]
                kmeans = KMeans(n_clusters=num_nodes, random_state=0).fit(np_embeddings, sample_weight=np_membership_masks)

                # At this point, the id of each node is the cluster assigned by kmeans
                kmeans_labels = np.array(kmeans.labels_).reshape(-1, num_nodes)  # [B, num_nodes]
                # TODO: To improve, do not assign same cluster to multiple nodes in the same hyperedge

                # Collect hyperedges represented by cluster ids
                hyperedges = set()
                for incidence_matrix, labels in zip(t_incidences, kmeans_labels):
                    for col in incidence_matrix.T:
                        nodes = torch.nonzero(col).squeeze().tolist()
                        if isinstance(nodes, int):
                            nodes = [nodes]
                        if len(nodes) < 1:
                            continue
                        cluster_ids = tuple(sorted(set(labels[nodes].tolist())))
                        hyperedges.add(cluster_ids)
                        if len(hyperedges) >= num_hyperedges:
                            break
                    if len(hyperedges) >= num_hyperedges:
                        break
                if len(hyperedges) >= num_hyperedges:
                    logging.info(f"Generated {len(hyperedges)} hyperedges, stopping generation.")
                    break
                logging.info(f"Generated {len(hyperedges)} hyperedges, continuing generation.")
                print(f"Generated {len(hyperedges)} hyperedges, continuing generation.")
                pbar.update(len(hyperedges))
                tau = tau * tau_multiplier
            hypergraph = xgi.Hypergraph(list(hyperedges))
            return hypergraph

    @torch.inference_mode()
    def sample_conditional(self,
                           node_features: torch.Tensor,
                           num_hyperedges: int,
                           walk_length: int,
                           batch_size: int,
                           initial_tau: float = 1.0,
                           tau_multiplier: float = 1 + 1e-12):
        assert not self.bvae.vertex_encoding, "Conditional sampling is only supported when vertex encoding is disabled."
        with tqdm(total=num_hyperedges, desc="Sampling hyperedges", leave=False) as pbar:
            pass
