# B-VAE
hydra --batch-size 32 -daqh/email-Enron train --min-epochs 3000 -lr 1e-4 --model-size S bvae
hydra --batch-size 32 -daqh/NDC-classes train --min-epochs 3000 -lr 1e-4 --model-size S bvae
hydra --batch-size 32 -daqh/contact-high-school train --min-epochs 3000 -lr 1e-4 --model-size S bvae
hydra --batch-size 32 -daqh/contact-primary-school train --min-epochs 3000 -lr 1e-4 --model-size S bvae
hydra --batch-size 16 -daqh/email-Eu train --min-epochs 3000 -lr 1e-4 --model-size S bvae

# DDM

hydra --batch-size 32 --val-size 0 daqh/email-Enron train -lr 1e-4 --model-size M ddm --bvae-ckpt logs/daqh/email-Enron/BVAE-HyDRA-S/logs/version_0/checkpoints/last.ckpt
#
hydra --batch-size 32 --val-size 0 daqh/contact-high-school train -lr 1e-4 --model-size M ddm --bvae-ckpt logs/daqh/contact-high-school/BVAE-HyDRA-S/logs/version_0/checkpoints/last.ckpt
#
#
#
