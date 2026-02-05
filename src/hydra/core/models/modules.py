import torch
import torch.nn as nn
from tqdm import tqdm
import lightning as L
import torch.nn.functional as F
import logging

from diffusers import DDPMScheduler

from .parameter_initialization import init_hypergraph_encoder, init_hypergraph_decoder, init_dit_weights
from .components import DiT, HGAT, HypergraphDecoder
from .utils import batch_index_contrastive_loss

DEFAULT_LR = 1e-4

class HypergraphBetaVAE(L.LightningModule):

    def __init__(self,
                 num_hyperedges: int,
                 kl_weight: float = 1.0,
                 learning_rate: float = None,
                 encode_nodes: bool = True):
        super().__init__()
        self.num_hyperedges = num_hyperedges
        self.kl_weight = kl_weight
        self.learning_rate = learning_rate or DEFAULT_LR
        self.encode_nodes = encode_nodes
        self.save_hyperparameters()

    def configure_model(self):

        # Encoder
        if self.encode_nodes:
            self.x_encoder_mu = HGAT(
                in_channels=128,
                hidden_channels=512,
                num_layers=3,
                heads=4
            )
            self.x_encoder_log_var = HGAT(
                in_channels=128,
                hidden_channels=512,
                num_layers=3,
                heads=4
            )
        self.y_encoder_mu = HGAT(
            in_channels=128,
            hidden_channels=512,
            num_layers=3,
            heads=4
        )
        self.y_encoder_log_var = HGAT(
            in_channels=128,
            hidden_channels=512,
            num_layers=3,
            heads=4
        )
        self.y_emb = nn.Embedding(
            num_embeddings=self.num_hyperedges,
            embedding_dim=128
        )
        # Decoder
        self.hypergraph_decoder = HypergraphDecoder(
            in_channels=128,
            num_classes=2
        )
        if self.encode_nodes:
            self.node_features_decoder = HGAT(
                in_channels=128,
                hidden_channels=512,
                num_layers=3,
                heads=4
            )

        nn.init.constant_(self.y_emb.weight, 0.0)
        # Initialize parameters
        if self.encode_nodes:
            init_hypergraph_encoder(self.x_encoder_mu)
            init_hypergraph_encoder(self.x_encoder_log_var)
        init_hypergraph_encoder(self.y_encoder_mu)
        init_hypergraph_encoder(self.y_encoder_log_var)
        init_hypergraph_decoder(self.hypergraph_decoder)
        if self.encode_nodes:
            init_hypergraph_encoder(self.node_features_decoder)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.learning_rate,
                                      weight_decay=1e-5)
        return optimizer

    def on_train_epoch_end(self):
        if self.current_epoch % 50 == 0:
            # Log weights
            for name, param in self.named_parameters():
                self.logger.experiment.add_histogram(
                    tag=f"weights/{name}",
                    values=param,
                    global_step=self.current_epoch
                )

    def forward(self, x: torch.Tensor, y: torch.Tensor, h: torch.Tensor):
        if self.encode_nodes:
            x_mu = self.x_encoder_mu(x, h)
            x_log_var = self.x_encoder_log_var(x, h)
            x_z = x_mu + torch.exp(0.5 * x_log_var) * torch.randn_like(x_log_var)
        else:
            x_z = x
            x_mu = None
            x_log_var = None

        dual_h = h.permute(0, 2, 1)
        y_mu = self.y_encoder_mu(y, dual_h)
        y_log_var = self.y_encoder_log_var(y, dual_h)
        y_z = y_mu + torch.exp(0.5 * y_log_var) * torch.randn_like(y_log_var)

        if self.encode_nodes:
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
        m = batch['nodes_mask']             # [B, num_nodes] TODO: Use this

        y_s = self.y_emb(s)
        y = y + y_s

        y_decay_loss = y_s.pow(2).mean() * 1e-4
        self.log("training/y_decay_loss", y_decay_loss.item(), prog_bar=False, on_step=True, on_epoch=True)

        h_logits, x_r, _, _, x_mu, y_mu, x_log_var, y_log_var = self.forward(x, y, h)    # Encode

        # From here, x and y are in the encoded space

        if self.encode_nodes:
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

        if self.encode_nodes:
            x_recon_loss = batch_index_contrastive_loss(x_r, temperature=1.0) # TODO: add mask on nodes that are not part of the hypergraph
            self.log("training/x_contrastive_loss", x_recon_loss.item(), prog_bar=False, on_step=True, on_epoch=True)
        else:
            x_recon_loss = 0.0

        if self.encode_nodes and self.global_step % 2:
            # Detach decoder 1
            x_recon_loss = x_recon_loss.detach()
        elif self.encode_nodes:
            # Detach decoder 2
            reconstruction_loss = reconstruction_loss.detach()

        loss = reconstruction_loss + self.kl_weight * (x_kl_loss + y_kl_loss) + x_recon_loss + y_decay_loss
        self.log("training/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['node_features']
        y = batch['hyperedge_features']
        h = batch['incidence_matrix']
        s = batch['touched_hyperedges']

        y_s = self.y_emb(s)
        y = y + y_s

        y_decay_loss = y_s.pow(2).mean() * 1e-4
        self.log("validation/y_decay_loss", y_decay_loss.item(), prog_bar=False, on_step=False, on_epoch=True)

        h_logits, x_r, _, _, x_mu, y_mu, x_log_var, y_log_var = self.forward(x, y, h)    # Encode

        # From here, x and y are in the encoded space

        if self.encode_nodes:
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

        if self.encode_nodes:
            x_recon_loss = batch_index_contrastive_loss(x_r, temperature=1.0)
            self.log("validation/x_contrastive_loss", x_recon_loss.item(), prog_bar=False, on_step=False, on_epoch=True)
        else:
            x_recon_loss = 0.0

        loss = reconstruction_loss + self.kl_weight * (x_kl_loss + y_kl_loss) + x_recon_loss + y_decay_loss
        self.log("validation/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def predict_step(self, batch, batch_idx):
        x, y, h, m, s, n = batch

        y_s = self.y_emb(s)
        y = y + y_s

        h_logits, x_r, x_z, y_z, x_mu, y_mu, x_log_var, y_log_var = self.forward(x, y, h)    # Encode

        return h_logits, x_r, x_z, y_z, x_mu, y_mu, x_log_var, y_log_var

class DiffusionTransformer(L.LightningModule):


    # TODO: Pass sampling mode ddpm / ddim
    def __init__(self,
                 T: int,
                 tau: int = 1,
                 scheduler_type: str = "cosine",
                 learning_rate: float = DEFAULT_LR,
                 bvae_version: int = 0):
        super().__init__()
        self.T = T
        self.tau = tau
        self.scheduler_type = scheduler_type
        self.learning_rate = learning_rate or DEFAULT_LR
        self.bvae_ckpt = f"logs/{dataset_name}/bvae/version_{bvae_version}/checkpoints/best.ckpt"
        self.save_hyperparameters()

    def on_train_epoch_end(self):
        if self.current_epoch % 500 == 0:
            # Log weights
            for name, param in self.named_parameters():
                self.logger.experiment.add_histogram(
                    tag=f"weights/{name}",
                    values=param,
                    global_step=self.current_epoch
                )

    def on_train_start(self):
        self.bvae.eval()  # keep in eval mode

    def on_train_epoch_start(self):
        self.bvae.eval()  # keep in eval mode

    def configure_model(self):
        logging.info(f"🔧 Loading BVAE from checkpoint: {self.bvae_ckpt}")
        self.bvae = HypergraphBetaVAE.load_from_checkpoint(self.bvae_ckpt, map_location="cpu")
        self.bvae.freeze()

        logging.info("🔧 Configuring the model")
        if self.scheduler_type == "linear":
            self.scheduler = LinearScheduler(
                self.T,
                device="cuda:0" # TODO: remove hardcoding
            )
        elif self.scheduler_type == "cosine":
            self.scheduler = CosineScheduler(
                self.T,
                device="cuda:0" # TODO: remove hardcoding
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
        self.gaussian_sampler = GaussianSampler(self.scheduler)

        if self.bvae.encode_nodes:
            self.vertices_dit = DiT(
                in_channels=128,
                hidden_channels=512,
                num_blocks=4,
                num_heads=4,
                cross_attention=True
            )
        self.hyperedges_dit = DiT(
            in_channels=128,
            hidden_channels=512,
            num_blocks=4,
            num_heads=4,
            cross_attention=True
        )

        if self.bvae.encode_nodes:
            init_dit_weights(self.vertices_dit)
        init_dit_weights(self.hyperedges_dit)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.learning_rate,
                                      weight_decay=0)
        return optimizer

    def forward(self, z_x: torch.Tensor, z_y: torch.Tensor, t: torch.Tensor):
        if self.bvae.encode_nodes:
            x_v_pred, _ = self.vertices_dit(z_x, t, z_y)
        else:
            x_v_pred = None
        y_v_pred, _ = self.hyperedges_dit(z_y, t, z_x)
        return x_v_pred, y_v_pred

    def training_step(self, batch, batch_idx):
        x, y, h, m, s, n = batch
        B = h.size(0)
        with torch.no_grad():
            y = y + self.bvae.y_emb(s)
            _, _, x_z, y_z, _, _, _, _ = self.bvae.forward(x, y, h) # Encode

        ts = self.scheduler.random_timesteps((B, 1), device=self.device)

        if self.bvae.encode_nodes:
            x_t, x_v = self.gaussian_sampler.forward(x_z, ts, return_v=True)
        else:
            x_t = x_z
            x_v = None
        y_t, y_v = self.gaussian_sampler.forward(y_z, ts, return_v=True)
        x_v_pred, y_v_pred = self.forward(x_t, y_t, ts)

        if self.bvae.encode_nodes:
            x_loss = F.mse_loss(x_v_pred, x_v)
            self.log("training/x_loss", x_loss.item(), prog_bar=False, on_step=True, on_epoch=True)
        else:
            x_loss = 0.0
        y_loss = F.mse_loss(y_v_pred, y_v)
        self.log("training/y_loss", y_loss.item(), prog_bar=False, on_step=True, on_epoch=True)
        loss = x_loss + y_loss
        self.log("training/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def sample(self, batch_size: int):
        pass

    def predict_step(self, batch, batch_idx):
        z_x_T, z_y_T = batch
        B, N, F = z_x_T.size()
        _, M, _ = z_y_T.size()
        ts = torch.full((B, 1), self.T, device=self.device, dtype=torch.long) # [B, 1]

        for _ in tqdm(range(self.T, 0, -1), desc="Sampling"):
            # x_v prediction is only used if nodes are encoded
            x_v_pred, y_v_pred = self(z_x_T, z_y_T, ts)
            if self.bvae.encode_nodes: # Reverse on nodes only if the model encodes them
                z_x_T = self.gaussian_sampler.reverse(z_x_T, ts, v_pred=x_v_pred)
            z_y_T = self.gaussian_sampler.reverse(z_y_T, ts, v_pred=y_v_pred)
            ts -= 1

        h_logits = self.bvae.hypergraph_decoder(z_x_T, z_y_T)  # Decode
        incidence_matrices = torch.distributions.Categorical(logits=h_logits).sample() # Sample hard incidence matrices
        if self.bvae.encode_nodes:
            x_rec = self.bvae.node_features_decoder(z_x_T, incidence_matrices)             # Produce node representations
        else:
            x_rec = None

        # This mask indicates hypergraph membership for each node
        membership_mask = incidence_matrices.sum(dim=2).bool() # [B, num_nodes]

        return h_logits, incidence_matrices, x_rec, membership_mask, z_x_T, z_y_T
