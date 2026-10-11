# Model checkpoints

The application requires these two checkpoints:

- `segmentation/hsegformer_best.pt`
- `classification/convnext_tiny_4class_candidate_best.pt`

They are intentionally excluded from ordinary Git commits because of their size.
Store them using a suitable model-artifact host and download them during deployment,
or use Git LFS if your hosting and storage limits permit it.

Do not publish datasets or sensitive patient information.
