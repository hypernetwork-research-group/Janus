# Conditional

# BVAE
janus --batch-size 32 daqh/email-Enron sample --ckpt logs/daqh/email-Enron/BVAE-Janus-S/logs/version_0/checkpoints/last.ckpt bvae
janus --batch-size 32 daqh/email-Eu sample --ckpt logs/daqh/email-Eu/BVAE-Janus-S/logs/version_0/checkpoints/last.ckpt bvae
janus --batch-size 32 daqh/contact-high-school sample --ckpt logs/daqh/contact-high-school/BVAE-Janus-S/logs/version_0/checkpoints/last.ckpt bvae
janus --batch-size 32 daqh/contact-primary-school sample --ckpt logs/daqh/contact-primary-school/BVAE-Janus-S/logs/version_0/checkpoints/last.ckpt bvae
janus --batch-size 32 daqh/NDC-classes sample --ckpt logs/daqh/NDC-classes/BVAE-Janus-S/logs/version_0/checkpoints/last.ckpt bvae

# DDM
janus --batch-size 32 daqh/email-Enron sample --ckpt logs/daqh/email-Enron/DDM-Janus-M/logs/version_0/checkpoints/last.ckpt ddm
janus --batch-size 32 daqh/NDC-classes sample --ckpt logs/daqh/NDC-classes/DDM-Janus-M/logs/version_0/checkpoints/last.ckpt ddm
janus --batch-size 32 daqh/contact-high-school sample --ckpt logs/daqh/contact-high-school/DDM-Janus-M/logs/version_0/checkpoints/last.ckpt ddm
janus --batch-size 32 daqh/contact-primary-school sample --ckpt logs/daqh/contact-primary-school/DDM-Janus-M/logs/version_0/checkpoints/last.ckpt ddm
janus --batch-size 32 daqh/email-Eu sample --ckpt logs/daqh/email-Eu/DDM-Janus-M/logs/version_0/checkpoints/last.ckpt ddm

# Unconditional

# BVAE
janus --batch-size 32 daqh/email-Enron sample --ckpt logs/daqh/email-Enron/BVAE-Janus-V-S/logs/version_0/checkpoints/last.ckpt bvae
janus --batch-size 32 daqh/email-Eu sample --ckpt logs/daqh/email-Eu/BVAE-Janus-V-M/logs/version_0/checkpoints/last.ckpt bvae
janus --batch-size 32 daqh/contact-high-school sample --ckpt logs/daqh/contact-high-school/BVAE-Janus-V-S/logs/version_0/checkpoints/last.ckpt bvae
janus --batch-size 32 daqh/contact-primary-school sample --ckpt logs/daqh/contact-primary-school/BVAE-Janus-V-S/logs/version_0/checkpoints/last.ckpt bvae
janus --batch-size 32 daqh/NDC-classes sample --ckpt logs/daqh/NDC-classes/BVAE-Janus-V-S/logs/version_0/checkpoints/last.ckpt bvae

# DDM
janus --batch-size 32 daqh/email-Enron sample --ckpt logs/daqh/email-Enron/DDM-Janus-V-M/logs/version_0/checkpoints/last.ckpt ddm
janus --batch-size 32 daqh/NDC-classes sample --ckpt logs/daqh/NDC-classes/DDM-Janus-V-M/logs/version_0/checkpoints/last.ckpt ddm
janus --batch-size 32 daqh/contact-high-school sample --ckpt logs/daqh/contact-high-school/DDM-Janus-V-M/logs/version_0/checkpoints/last.ckpt ddm
janus --batch-size 32 daqh/contact-primary-school sample --ckpt logs/daqh/contact-primary-school/DDM-Janus-V-M/logs/version_0/checkpoints/last.ckpt ddm
janus --batch-size 32 daqh/email-Eu sample --ckpt logs/daqh/email-Eu/DDM-Janus-V-M/logs/version_0/checkpoints/last.ckpt ddm

