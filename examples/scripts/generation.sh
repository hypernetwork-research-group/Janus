# BVAE
hydra --batch-size 4 daqh/email-Enron sample --ckpt logs/daqh/email-Enron/BVAE-HyDRA-S/logs/version_0/checkpoints/last.ckpt bvae

# DDM
hydra --batch-size 32 daqh/email-Enron sample --ckpt logs/daqh/email-Enron/DDM-HyDRA-S/logs/version_0/checkpoints/last.ckpt ddm
