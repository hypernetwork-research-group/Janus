# BVAE
hydra --batch-size 32 daqh/email-Enron sample --ckpt logs/daqh/email-Enron/BVAE-HyDRA-S/logs/version_0/checkpoints/last.ckpt bvae
hydra --batch-size 32 daqh/email-Eu sample --ckpt logs/daqh/email-Eu/BVAE-HyDRA-S/logs/version_0/checkpoints/last.ckpt bvae
hydra --batch-size 32 daqh/contact-high-school sample --ckpt logs/daqh/contact-high-school/BVAE-HyDRA-S/logs/version_0/checkpoints/last.ckpt bvae
hydra --batch-size 32 daqh/contact-primary-school sample --ckpt logs/daqh/contact-primary-school/BVAE-HyDRA-S/logs/version_0/checkpoints/last.ckpt bvae
hydra --batch-size 32 daqh/NDC-classes sample --ckpt logs/daqh/NDC-classes/BVAE-HyDRA-S/logs/version_0/checkpoints/last.ckpt bvae

# DDM
hydra --batch-size 32 daqh/email-Enron sample --ckpt logs/daqh/email-Enron/DDM-HyDRA-S/logs/version_0/checkpoints/last.ckpt ddm
#
hydra --batch-size 32 daqh/contact-high-school sample --ckpt logs/daqh/contact-high-school/DDM-HyDRA-M/logs/version_0/checkpoints/last.ckpt ddm
hydra --batch-size 32 daqh/contact-primary-school sample --ckpt logs/daqh/contact-primary-school/DDM-HyDRA-M/logs/version_0/checkpoints/last.ckpt ddm
#
