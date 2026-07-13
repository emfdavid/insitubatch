"""Train-in-place denoising over real Hubble (WFC3/IR) frames.

A worked cross-domain example: FITS images on MAST's public AWS bucket, indexed as
virtual references (VirtualiZarr -> Icechunk) and streamed by insitubatch with no
reshard. See ``data.py`` for the pipeline and ``train_torch.py`` for the model.
"""
