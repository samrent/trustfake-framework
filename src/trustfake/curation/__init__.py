"""TB-E3 curation ladder: ingest, embed-once, gates, arms, probe fits.

Spec: documentation/specs/tb-e3-curation-ladder.md. The instrument is a
frozen standard ViT-B/16 (open_clip, laion2B-s34B-b88K); every image is
embedded exactly once and every arm is an index list over the cached
features, so a fit is minutes and the expensive thing (decode + encode)
is paid once per image, ever.
"""
