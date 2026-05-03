# Conditional

# BVAE
janus --batch-size 32 anonymous-author-1234/email-Enron sample --ckpt logs/anonymous-author-1234/email-Enron/BVAE-Janus-S/logs/version_0/checkpoints/last.ckpt bvae
janus --batch-size 32 anonymous-author-1234/email-Eu sample --ckpt logs/anonymous-author-1234/email-Eu/BVAE-Janus-S/logs/version_0/checkpoints/last.ckpt bvae
janus --batch-size 32 anonymous-author-1234/contact-high-school sample --ckpt logs/anonymous-author-1234/contact-high-school/BVAE-Janus-S/logs/version_0/checkpoints/last.ckpt bvae
janus --batch-size 32 anonymous-author-1234/contact-primary-school sample --ckpt logs/anonymous-author-1234/contact-primary-school/BVAE-Janus-S/logs/version_0/checkpoints/last.ckpt bvae
janus --batch-size 32 anonymous-author-1234/NDC-classes sample --ckpt logs/anonymous-author-1234/NDC-classes/BVAE-Janus-S/logs/version_0/checkpoints/last.ckpt bvae

# DDM
janus --batch-size 32 anonymous-author-1234/email-Enron sample --ckpt logs/anonymous-author-1234/email-Enron/DDM-Janus-M/logs/version_0/checkpoints/last.ckpt ddm
janus --batch-size 32 anonymous-author-1234/NDC-classes sample --ckpt logs/anonymous-author-1234/NDC-classes/DDM-Janus-M/logs/version_0/checkpoints/last.ckpt ddm
janus --batch-size 32 anonymous-author-1234/contact-high-school sample --ckpt logs/anonymous-author-1234/contact-high-school/DDM-Janus-M/logs/version_0/checkpoints/last.ckpt ddm
janus --batch-size 32 anonymous-author-1234/contact-primary-school sample --ckpt logs/anonymous-author-1234/contact-primary-school/DDM-Janus-M/logs/version_0/checkpoints/last.ckpt ddm
janus --batch-size 32 anonymous-author-1234/email-Eu sample --ckpt logs/anonymous-author-1234/email-Eu/DDM-Janus-M/logs/version_0/checkpoints/last.ckpt ddm

# Unconditional

# BVAE
janus --batch-size 32 anonymous-author-1234/email-Enron sample --ckpt logs/anonymous-author-1234/email-Enron/BVAE-Janus-V-S/logs/version_0/checkpoints/last.ckpt bvae
janus --batch-size 32 anonymous-author-1234/email-Eu sample --ckpt logs/anonymous-author-1234/email-Eu/BVAE-Janus-V-M/logs/version_0/checkpoints/last.ckpt bvae
janus --batch-size 32 anonymous-author-1234/contact-high-school sample --ckpt logs/anonymous-author-1234/contact-high-school/BVAE-Janus-V-S/logs/version_0/checkpoints/last.ckpt bvae
janus --batch-size 32 anonymous-author-1234/contact-primary-school sample --ckpt logs/anonymous-author-1234/contact-primary-school/BVAE-Janus-V-S/logs/version_0/checkpoints/last.ckpt bvae
janus --batch-size 32 anonymous-author-1234/NDC-classes sample --ckpt logs/anonymous-author-1234/NDC-classes/BVAE-Janus-V-S/logs/version_0/checkpoints/last.ckpt bvae

# DDM
janus --batch-size 32 anonymous-author-1234/email-Enron sample --ckpt logs/anonymous-author-1234/email-Enron/DDM-Janus-V-M/logs/version_0/checkpoints/last.ckpt ddm
janus --batch-size 32 anonymous-author-1234/NDC-classes sample --ckpt logs/anonymous-author-1234/NDC-classes/DDM-Janus-V-M/logs/version_0/checkpoints/last.ckpt ddm
janus --batch-size 32 anonymous-author-1234/contact-high-school sample --ckpt logs/anonymous-author-1234/contact-high-school/DDM-Janus-V-M/logs/version_0/checkpoints/last.ckpt ddm
janus --batch-size 32 anonymous-author-1234/contact-primary-school sample --ckpt logs/anonymous-author-1234/contact-primary-school/DDM-Janus-V-M/logs/version_0/checkpoints/last.ckpt ddm
janus --batch-size 32 anonymous-author-1234/email-Eu sample --ckpt logs/anonymous-author-1234/email-Eu/DDM-Janus-V-M/logs/version_0/checkpoints/last.ckpt ddm

