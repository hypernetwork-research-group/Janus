# Conditional

# B-VAE
janus --batch-size 32 daqh/email-Enron train --min-epochs 3000 -lr 1e-4 --model-size S bvae
janus --batch-size 32 daqh/NDC-classes train --min-epochs 3000 -lr 1e-4 --model-size S bvae
janus --batch-size 32 daqh/contact-high-school train --min-epochs 3000 -lr 1e-4 --model-size S bvae
janus --batch-size 32 daqh/contact-primary-school train --min-epochs 3000 -lr 1e-4 --model-size S bvae
janus --batch-size 32 daqh/email-Eu train --min-epochs 3000 -lr 1e-4 --model-size S bvae

# DDM
janus --batch-size 32 --val-size 0 daqh/email-Enron train -lr 1e-4 --model-size M ddm --bvae-ckpt logs/daqh/email-Enron/BVAE-Janus-S/logs/version_0/checkpoints/last.ckpt
janus --batch-size 32 --val-size 0 daqh/NDC-classes train -lr 1e-4 --model-size M ddm --bvae-ckpt logs/daqh/NDC-classes/BVAE-Janus-S/logs/version_0/checkpoints/last.ckpt
janus --batch-size 32 --val-size 0 daqh/contact-high-school train -lr 1e-4 --model-size M ddm --bvae-ckpt logs/daqh/contact-high-school/BVAE-Janus-S/logs/version_0/checkpoints/last.ckpt
janus --batch-size 32 --val-size 0 daqh/contact-primary-school train -lr 1e-4 --model-size M ddm --bvae-ckpt logs/daqh/contact-primary-school/BVAE-Janus-S/logs/version_0/checkpoints/last.ckpt
#

# Unconditional

# B-VAE
janus --batch-size 32 daqh/email-Enron train --min-epochs 3000 -lr 1e-4 --model-size S bvae --vertex-encoding
janus --batch-size 32 daqh/NDC-classes train --min-epochs 3000 -lr 1e-4 --model-size S bvae --vertex-encoding
janus --batch-size 32 daqh/contact-high-school train --min-epochs 3000 -lr 1e-4 --model-size S bvae --vertex-encoding
janus --batch-size 32 daqh/contact-primary-school train --min-epochs 3000 -lr 1e-4 --model-size S bvae --vertex-encoding
janus --batch-size 32 daqh/email-Eu train --min-epochs 3000 -lr 1e-4 --model-size S bvae --vertex-encoding

# DDM
janus --batch-size 32 --val-size 0 daqh/email-Enron train -lr 1e-4 --model-size M ddm --bvae-ckpt logs/daqh/email-Enron/BVAE-Janus-S/logs/version_0/checkpoints/last.ckpt
janus --batch-size 32 --val-size 0 daqh/contact-high-school train -lr 1e-4 --model-size M ddm --bvae-ckpt logs/daqh/contact-high-school/BVAE-Janus-S/logs/version_0/checkpoints/last.ckpt
janus --batch-size 32 --val-size 0 daqh/email-Enron train --ckpt logs/daqh/email-Enron/DDM-Janus-V-M/logs/version_0/checkpoints/last.ckpt -lr 1e-4 --model-size M ddm --bvae-ckpt logs/daqh/email-Enron/BVAE-Janus-S/logs/version_0/checkpoints/last.ckpt
janus --batch-size 32 --val-size 0 daqh/contact-primary-school train --min-epochs 3000 -lr 1e-4 --model-size S bvae --vertex-encoding
