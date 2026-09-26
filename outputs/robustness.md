# Metamorphic robustness

AP at IoU 0.5, 500 KITTI frames, same ground truth throughout. Strength 0 is the unperturbed baseline for that perturbation.

## brightness (fractional change)

| strength | Car AP | Pedestrian AP | Cyclist AP |
|---|---|---|---|
| +0 | 0.758 | 0.443 | 0.005 |
| +0.3 | 0.760 | 0.442 | 0.005 |
| +0.6 | 0.758 | 0.441 | 0.005 |
| -0.3 | 0.760 | 0.441 | 0.004 |
| -0.6 | 0.756 | 0.433 | 0.004 |

## contrast (fraction removed)

| strength | Car AP | Pedestrian AP | Cyclist AP |
|---|---|---|---|
| +0 | 0.758 | 0.443 | 0.005 |
| +0.3 | 0.743 | 0.437 | 0.009 |
| +0.6 | 0.717 | 0.414 | 0.007 |
| +0.8 | 0.659 | 0.365 | 0.006 |

## blur (gaussian radius px)

| strength | Car AP | Pedestrian AP | Cyclist AP |
|---|---|---|---|
| +0 | 0.758 | 0.443 | 0.005 |
| +1 | 0.753 | 0.440 | 0.004 |
| +2 | 0.733 | 0.405 | 0.003 |
| +4 | 0.634 | 0.366 | 0.004 |

## jpeg (compression, 0 = quality 100)

| strength | Car AP | Pedestrian AP | Cyclist AP |
|---|---|---|---|
| +0 | 0.758 | 0.443 | 0.005 |
| +0.6 | 0.747 | 0.440 | 0.008 |
| +0.85 | 0.731 | 0.413 | 0.004 |
| +0.95 | 0.695 | 0.391 | 0.006 |

## fog (veil opacity)

| strength | Car AP | Pedestrian AP | Cyclist AP |
|---|---|---|---|
| +0 | 0.758 | 0.443 | 0.005 |
| +0.2 | 0.739 | 0.436 | 0.009 |
| +0.4 | 0.722 | 0.430 | 0.007 |
| +0.6 | 0.698 | 0.411 | 0.008 |

