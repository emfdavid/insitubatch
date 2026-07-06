"""OME-NGFF cell segmentation: one insitu dataset, raw + mask co-batched over Z.

The cross-domain companion to ``examples/advection`` -- same engine, a different
geometry. Here a *sample* is one Z-plane of a confocal stack (``sample_axis=2`` of the
OME-NGFF ``(T,C,Z,Y,X)`` layout), and two variables are gathered at the same anchor: the
2-channel ``raw`` image (chunked one plane deep) and its ``mask`` label (chunked 30 planes
deep, tiled in Y/X). Different physical chunking, different channel count, one sample grid,
no reshard -- the arbitrary-sample-axis + per-variable-chunking unlock on a real store.

``data.py`` builds the store (synthetic cells, or the real IDR image), the dataset, and the
shared eval; ``train_torch.py`` trains a tiny segmentation CNN that beats a global-threshold
(Otsu) baseline by reading spatial context the threshold cannot see.
"""
