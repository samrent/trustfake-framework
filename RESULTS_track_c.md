# Track C -- auxiliary depth: results

Collated from `/home/samuel-renteria/Desktop/FILES/PROJECTS/trustfake/_runs/logs/track_c_depth`; 102 cells.

Track C is its own table. Phi = fd_auroc (failure detection). `_tr` = transfer attack scoring, `_wb` = white-box (the attack optimises the depth score itself). Every depth-score number carries the caveat that a gradient adaptive attack on the residual is a follow-up.

## Validity gate G2: is the residual just 1 - MSP relabelled? (computed on the SID-Set calib split for every leg)

| arm | abs rho vs 1-MSP | residual mean | residual std | n | verdict |
|---|---:|---:|---:|---:|---|
| pgd_at_depth_l1.0 | 0.2792 | 0.6047 | 0.3197 | 7059 | independent |
| standard_depth_l1.0 | 0.2534 | 0.5139 | 0.2854 | 7059 | independent |

## SID-Set (in-domain)

### Prediction axis: accuracy per condition (msp cells)

| arm | clean | ace_uint8 | corruption_jpeg | pgd | query_overconf | query_underconf |
|---|---:|---:|---:|---:|---:|---:|
| pgd_at | 0.7003 | 0.7003 | 0.6982 | 0.6484 | 0.7003 | 0.7003 |
| pgd_at_depth_l1.0 | 0.6935 | 0.6935 | 0.6945 | 0.6880 | 0.6935 | 0.6935 |
| standard | 0.8249 | 0.8249 | 0.7769 | 0.0992 | 0.8249 | 0.8249 |
| standard_depth_l1.0 | 0.8317 | 0.8317 | 0.7868 | 0.1539 | 0.8317 | 0.8317 |

### Confidence axis: Phi (fd_auroc), three scorings side by side

| arm | cond | msp | combined | combined_tr | depth | depth_tr | depth_wb |
|---|---|---:|---:|---:|---:|---:|---:|
| pgd_at | clean | 0.7422 | -- | -- | -- | -- | -- |
| pgd_at | ace_uint8 | 0.7009 | -- | -- | -- | -- | -- |
| pgd_at | corruption_jpeg | 0.7426 | -- | -- | -- | -- | -- |
| pgd_at | pgd | 0.7363 | -- | -- | -- | -- | -- |
| pgd_at | query_overconf | 0.7418 | -- | -- | -- | -- | -- |
| pgd_at | query_underconf | 0.7413 | -- | -- | -- | -- | -- |
| pgd_at_depth_l1.0 | clean | 0.7665 | 0.7255 | -- | 0.5805 | -- | -- |
| pgd_at_depth_l1.0 | ace_uint8 | 0.7217 | -- | 0.6955 | -- | 0.5792 | 0.5792 |
| pgd_at_depth_l1.0 | corruption_jpeg | 0.7678 | 0.7255 | -- | 0.5752 | -- | -- |
| pgd_at_depth_l1.0 | pgd | 0.7500 | 0.7350 | -- | 0.6208 | -- | -- |
| pgd_at_depth_l1.0 | query_overconf | 0.7689 | -- | 0.7236 | -- | 0.5779 | 0.5716 |
| pgd_at_depth_l1.0 | query_underconf | 0.7625 | -- | 0.7289 | -- | 0.5750 | 0.5947 |
| standard | clean | 0.8156 | -- | -- | -- | -- | -- |
| standard | ace_uint8 | 0.2153 | -- | -- | -- | -- | -- |
| standard | corruption_jpeg | 0.8006 | -- | -- | -- | -- | -- |
| standard | pgd | 0.3630 | -- | -- | -- | -- | -- |
| standard | query_overconf | 0.8018 | -- | -- | -- | -- | -- |
| standard | query_underconf | 0.7365 | -- | -- | -- | -- | -- |
| standard_depth_l1.0 | clean | 0.8393 | 0.7498 | -- | 0.5761 | -- | -- |
| standard_depth_l1.0 | ace_uint8 | 0.1610 | -- | 0.3481 | -- | 0.6070 | 0.6070 |
| standard_depth_l1.0 | corruption_jpeg | 0.7729 | 0.7022 | -- | 0.5598 | -- | -- |
| standard_depth_l1.0 | pgd | 0.4937 | 0.5367 | -- | 0.5361 | -- | -- |
| standard_depth_l1.0 | query_overconf | 0.8083 | -- | 0.7094 | -- | 0.5779 | 0.5591 |
| standard_depth_l1.0 | query_underconf | 0.7123 | -- | 0.6571 | -- | 0.5589 | 0.5808 |

### Selective / moderation columns, per cell

| arm | cond | score | acc | Phi | aurc | n_op | rec_tam | resid_risk_2axis | review_2axis |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| pgd_at | clean | msp | 0.7003 | 0.7422 | 0.1285 | 885 | 0.7690 | 0.0402 | 0.6270 |
| pgd_at | ace_uint8 | msp | 0.7003 | 0.7009 | 0.1425 | 896 | 0.7690 | 0.0296 | 0.6280 |
| pgd_at | corruption_jpeg | msp | 0.6982 | 0.7426 | 0.1294 | 889 | 0.7627 | 0.0402 | 0.6270 |
| pgd_at | pgd | msp | 0.6484 | 0.7363 | 0.1612 | 980 | 0.5665 | 0.0312 | 0.6470 |
| pgd_at | query_overconf | msp | 0.7003 | 0.7418 | 0.1286 | 856 | 0.7690 | 0.0462 | 0.6100 |
| pgd_at | query_underconf | msp | 0.7003 | 0.7413 | 0.1288 | 909 | 0.7690 | 0.0113 | 0.6450 |
| pgd_at_depth_l1.0 | clean | msp | 0.6935 | 0.7665 | 0.1226 | 989 | 0.6709 | 0.0413 | 0.6130 |
| pgd_at_depth_l1.0 | clean | combined | 0.6935 | 0.7255 | 0.1457 | 999 | 0.6709 | 0.0449 | 0.5990 |
| pgd_at_depth_l1.0 | clean | depth | 0.6935 | 0.5805 | 0.2326 | 999 | 0.6709 | 0.0470 | 0.6170 |
| pgd_at_depth_l1.0 | ace_uint8 | msp | 0.6935 | 0.7217 | 0.1370 | 993 | 0.6709 | 0.0522 | 0.6170 |
| pgd_at_depth_l1.0 | ace_uint8 | combined_tr | 0.6935 | 0.6955 | 0.1576 | 999 | 0.6709 | 0.0553 | 0.6020 |
| pgd_at_depth_l1.0 | ace_uint8 | depth_tr | 0.6935 | 0.5792 | 0.2319 | 999 | 0.6709 | 0.0577 | 0.6190 |
| pgd_at_depth_l1.0 | ace_uint8 | depth_wb | 0.6935 | 0.5792 | 0.2319 | 999 | 0.6709 | 0.0577 | 0.6190 |
| pgd_at_depth_l1.0 | corruption_jpeg | msp | 0.6945 | 0.7678 | 0.1217 | 991 | 0.6741 | 0.0415 | 0.6140 |
| pgd_at_depth_l1.0 | corruption_jpeg | combined | 0.6945 | 0.7255 | 0.1447 | 999 | 0.6741 | 0.0449 | 0.5990 |
| pgd_at_depth_l1.0 | corruption_jpeg | depth | 0.6945 | 0.5752 | 0.2310 | 999 | 0.6741 | 0.0466 | 0.6140 |
| pgd_at_depth_l1.0 | pgd | msp | 0.6880 | 0.7500 | 0.1320 | 999 | 0.6266 | 0.0087 | 0.6560 |
| pgd_at_depth_l1.0 | pgd | combined | 0.6880 | 0.7350 | 0.1450 | 999 | 0.6266 | 0.0313 | 0.6170 |
| pgd_at_depth_l1.0 | pgd | depth | 0.6880 | 0.6208 | 0.2255 | 999 | 0.6266 | 0.0450 | 0.6220 |
| pgd_at_depth_l1.0 | query_overconf | msp | 0.6935 | 0.7689 | 0.1217 | 983 | 0.6709 | 0.0529 | 0.6030 |
| pgd_at_depth_l1.0 | query_overconf | combined_tr | 0.6935 | 0.7236 | 0.1466 | 999 | 0.6709 | 0.0516 | 0.5930 |
| pgd_at_depth_l1.0 | query_overconf | depth_tr | 0.6935 | 0.5779 | 0.2318 | 999 | 0.6709 | 0.0492 | 0.6140 |
| pgd_at_depth_l1.0 | query_overconf | depth_wb | 0.6935 | 0.5716 | 0.2362 | 999 | 0.6709 | 0.0473 | 0.5980 |
| pgd_at_depth_l1.0 | query_underconf | msp | 0.6935 | 0.7625 | 0.1240 | 995 | 0.6709 | 0.0349 | 0.6270 |
| pgd_at_depth_l1.0 | query_underconf | combined_tr | 0.6935 | 0.7289 | 0.1440 | 999 | 0.6709 | 0.0504 | 0.6030 |
| pgd_at_depth_l1.0 | query_underconf | depth_tr | 0.6935 | 0.5750 | 0.2326 | 999 | 0.6709 | 0.0571 | 0.6150 |
| pgd_at_depth_l1.0 | query_underconf | depth_wb | 0.6935 | 0.5947 | 0.2365 | 999 | 0.6709 | 0.0366 | 0.6450 |
| standard | clean | msp | 0.8249 | 0.8156 | 0.0511 | 950 | 0.6994 | 0.0436 | 0.4040 |
| standard | ace_uint8 | msp | 0.8249 | 0.2153 | 0.2414 | 1000 | 0.6994 | 0.2299 | 0.3300 |
| standard | corruption_jpeg | msp | 0.7769 | 0.8006 | 0.0745 | 981 | 0.4968 | 0.0509 | 0.4890 |
| standard | pgd | msp | 0.0992 | 0.3630 | 0.9405 | 759 | 0.1234 | 0.7072 | 0.0300 |
| standard | query_overconf | msp | 0.8249 | 0.8018 | 0.0533 | 829 | 0.6994 | 0.1199 | 0.1410 |
| standard | query_underconf | msp | 0.8249 | 0.7365 | 0.0684 | 982 | 0.6994 | 0.0061 | 0.6700 |
| standard_depth_l1.0 | clean | msp | 0.8317 | 0.8393 | 0.0442 | 996 | 0.7532 | 0.0565 | 0.3450 |
| standard_depth_l1.0 | clean | combined | 0.8317 | 0.7498 | 0.0655 | 999 | 0.7532 | 0.0539 | 0.3510 |
| standard_depth_l1.0 | clean | depth | 0.8317 | 0.5761 | 0.1438 | 999 | 0.7532 | 0.0583 | 0.3830 |
| standard_depth_l1.0 | ace_uint8 | msp | 0.8317 | 0.1610 | 0.3078 | 1000 | 0.7532 | 0.2116 | 0.2770 |
| standard_depth_l1.0 | ace_uint8 | combined_tr | 0.8317 | 0.3481 | 0.1973 | 999 | 0.7532 | 0.2076 | 0.2630 |
| standard_depth_l1.0 | ace_uint8 | depth_tr | 0.8317 | 0.6070 | 0.1226 | 999 | 0.7532 | 0.1851 | 0.3140 |
| standard_depth_l1.0 | ace_uint8 | depth_wb | 0.8317 | 0.6070 | 0.1226 | 999 | 0.7532 | 0.1851 | 0.3140 |
| standard_depth_l1.0 | corruption_jpeg | msp | 0.7868 | 0.7729 | 0.0818 | 999 | 0.5633 | 0.0542 | 0.4460 |
| standard_depth_l1.0 | corruption_jpeg | combined | 0.7868 | 0.7022 | 0.1040 | 999 | 0.5633 | 0.0525 | 0.4480 |
| standard_depth_l1.0 | corruption_jpeg | depth | 0.7868 | 0.5598 | 0.1795 | 999 | 0.5633 | 0.0482 | 0.4810 |
| standard_depth_l1.0 | pgd | msp | 0.1539 | 0.4937 | 0.8547 | 504 | 0.2437 | 0.5324 | 0.0420 |
| standard_depth_l1.0 | pgd | combined | 0.1539 | 0.5367 | 0.8387 | 999 | 0.2437 | 0.5324 | 0.0420 |
| standard_depth_l1.0 | pgd | depth | 0.1539 | 0.5361 | 0.8352 | 999 | 0.2437 | 0.5702 | 0.5230 |
| standard_depth_l1.0 | query_overconf | msp | 0.8317 | 0.8083 | 0.0503 | 970 | 0.7532 | 0.1263 | 0.0740 |
| standard_depth_l1.0 | query_overconf | combined_tr | 0.8317 | 0.7094 | 0.0768 | 999 | 0.7532 | 0.1265 | 0.0750 |
| standard_depth_l1.0 | query_overconf | depth_tr | 0.8317 | 0.5779 | 0.1358 | 999 | 0.7532 | 0.1213 | 0.1590 |
| standard_depth_l1.0 | query_overconf | depth_wb | 0.8317 | 0.5591 | 0.1465 | 999 | 0.7532 | 0.0442 | 0.3670 |
| standard_depth_l1.0 | query_underconf | msp | 0.8317 | 0.7123 | 0.0721 | 998 | 0.7532 | 0.0070 | 0.7160 |
| standard_depth_l1.0 | query_underconf | combined_tr | 0.8317 | 0.6571 | 0.0901 | 999 | 0.7532 | 0.0060 | 0.6680 |
| standard_depth_l1.0 | query_underconf | depth_tr | 0.8317 | 0.5589 | 0.1472 | 999 | 0.7532 | 0.0058 | 0.6540 |
| standard_depth_l1.0 | query_underconf | depth_wb | 0.8317 | 0.5808 | 0.1339 | 999 | 0.7532 | 0.0549 | 0.5260 |

## So-Fake-OOD (shift, in-domain thresholds)

### Prediction axis: accuracy per condition (msp cells)

| arm | clean | ace_uint8 | corruption_jpeg | pgd | query_overconf | query_underconf |
|---|---:|---:|---:|---:|---:|---:|
| pgd_at | 0.3241 | 0.3241 | 0.3232 | 0.3209 | 0.3241 | 0.3241 |
| pgd_at_depth_l1.0 | 0.3411 | 0.3411 | 0.3408 | 0.3284 | 0.3411 | 0.3411 |
| standard | 0.3516 | 0.3516 | 0.3514 | 0.3401 | 0.3516 | 0.3516 |
| standard_depth_l1.0 | 0.3454 | 0.3454 | 0.3463 | 0.3284 | 0.3454 | 0.3454 |

### Confidence axis: Phi (fd_auroc), three scorings side by side

| arm | cond | msp | combined | combined_tr | depth | depth_tr | depth_wb |
|---|---|---:|---:|---:|---:|---:|---:|
| pgd_at | clean | 0.5350 | -- | -- | -- | -- | -- |
| pgd_at | ace_uint8 | 0.4538 | -- | -- | -- | -- | -- |
| pgd_at | corruption_jpeg | 0.5333 | -- | -- | -- | -- | -- |
| pgd_at | pgd | 0.5091 | -- | -- | -- | -- | -- |
| pgd_at | query_overconf | 0.5408 | -- | -- | -- | -- | -- |
| pgd_at | query_underconf | 0.5286 | -- | -- | -- | -- | -- |
| pgd_at_depth_l1.0 | clean | 0.5549 | 0.5426 | -- | 0.5171 | -- | -- |
| pgd_at_depth_l1.0 | ace_uint8 | 0.4638 | -- | 0.4947 | -- | 0.5201 | -- |
| pgd_at_depth_l1.0 | corruption_jpeg | 0.5535 | 0.5384 | -- | 0.5140 | -- | -- |
| pgd_at_depth_l1.0 | pgd | 0.5338 | 0.5212 | -- | 0.5087 | -- | -- |
| pgd_at_depth_l1.0 | query_overconf | 0.5663 | -- | 0.5525 | -- | 0.5156 | -- |
| pgd_at_depth_l1.0 | query_underconf | 0.5493 | -- | 0.5444 | -- | 0.5162 | -- |
| standard | clean | 0.4703 | -- | -- | -- | -- | -- |
| standard | ace_uint8 | 0.0222 | -- | -- | -- | -- | -- |
| standard | corruption_jpeg | 0.4554 | -- | -- | -- | -- | -- |
| standard | pgd | 0.5396 | -- | -- | -- | -- | -- |
| standard | query_overconf | 0.4838 | -- | -- | -- | -- | -- |
| standard | query_underconf | 0.5174 | -- | -- | -- | -- | -- |
| standard_depth_l1.0 | clean | 0.5118 | 0.4467 | -- | 0.4428 | -- | -- |
| standard_depth_l1.0 | ace_uint8 | 0.0173 | -- | 0.1371 | -- | 0.5003 | -- |
| standard_depth_l1.0 | corruption_jpeg | 0.4999 | 0.4613 | -- | 0.4582 | -- | -- |
| standard_depth_l1.0 | pgd | 0.5573 | 0.5347 | -- | 0.4932 | -- | -- |
| standard_depth_l1.0 | query_overconf | 0.5557 | -- | 0.4645 | -- | 0.4590 | -- |
| standard_depth_l1.0 | query_underconf | 0.5586 | -- | 0.4407 | -- | 0.4510 | -- |

### Selective / moderation columns, per cell

| arm | cond | score | acc | Phi | aurc | n_op | rec_tam | resid_risk_2axis | review_2axis |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| pgd_at | clean | msp | 0.3241 | 0.5350 | 0.6317 | 997 | 0.7196 | 0.7838 | 0.9630 |
| pgd_at | ace_uint8 | msp | 0.3241 | 0.4538 | 0.7018 | 995 | 0.7196 | 0.7500 | 0.9600 |
| pgd_at | corruption_jpeg | msp | 0.3232 | 0.5333 | 0.6341 | 996 | 0.7169 | 0.7500 | 0.9600 |
| pgd_at | pgd | msp | 0.3209 | 0.5091 | 0.6800 | 996 | 0.5291 | 0.5227 | 0.9560 |
| pgd_at | query_overconf | msp | 0.3241 | 0.5408 | 0.6290 | 996 | 0.7196 | 0.7538 | 0.9350 |
| pgd_at | query_underconf | msp | 0.3241 | 0.5286 | 0.6360 | 997 | 0.7196 | 0.6250 | 0.9840 |
| pgd_at_depth_l1.0 | clean | msp | 0.3411 | 0.5549 | 0.5994 | 1000 | 0.7090 | 0.7632 | 0.9620 |
| pgd_at_depth_l1.0 | clean | combined | 0.3411 | 0.5426 | 0.6037 | 1000 | 0.7090 | 0.5000 | 0.9380 |
| pgd_at_depth_l1.0 | clean | depth | 0.3411 | 0.5171 | 0.6194 | 1000 | 0.7090 | 0.4028 | 0.9280 |
| pgd_at_depth_l1.0 | ace_uint8 | msp | 0.3411 | 0.4638 | 0.6568 | 1000 | 0.7090 | 0.8298 | 0.9530 |
| pgd_at_depth_l1.0 | ace_uint8 | combined_tr | 0.3411 | 0.4947 | 0.6373 | 1000 | 0.7090 | 0.5775 | 0.9290 |
| pgd_at_depth_l1.0 | ace_uint8 | depth_tr | 0.3411 | 0.5201 | 0.6203 | 1000 | 0.7090 | 0.5128 | 0.9220 |
| pgd_at_depth_l1.0 | corruption_jpeg | msp | 0.3408 | 0.5535 | 0.5990 | 999 | 0.7196 | 0.7500 | 0.9640 |
| pgd_at_depth_l1.0 | corruption_jpeg | combined | 0.3408 | 0.5384 | 0.6046 | 1000 | 0.7196 | 0.4531 | 0.9360 |
| pgd_at_depth_l1.0 | corruption_jpeg | depth | 0.3408 | 0.5140 | 0.6226 | 1000 | 0.7196 | 0.3889 | 0.9280 |
| pgd_at_depth_l1.0 | pgd | msp | 0.3284 | 0.5338 | 0.6284 | 997 | 0.5529 | 0.2500 | 0.9680 |
| pgd_at_depth_l1.0 | pgd | combined | 0.3284 | 0.5212 | 0.6343 | 1000 | 0.5529 | 0.1739 | 0.9080 |
| pgd_at_depth_l1.0 | pgd | depth | 0.3284 | 0.5087 | 0.6479 | 1000 | 0.5529 | 0.1963 | 0.8930 |
| pgd_at_depth_l1.0 | query_overconf | msp | 0.3411 | 0.5663 | 0.5923 | 998 | 0.7090 | 0.7547 | 0.9470 |
| pgd_at_depth_l1.0 | query_overconf | combined_tr | 0.3411 | 0.5525 | 0.5952 | 1000 | 0.7090 | 0.6176 | 0.9320 |
| pgd_at_depth_l1.0 | query_overconf | depth_tr | 0.3411 | 0.5156 | 0.6135 | 1000 | 0.7090 | 0.5942 | 0.9310 |
| pgd_at_depth_l1.0 | query_underconf | msp | 0.3411 | 0.5493 | 0.6002 | 1000 | 0.7090 | 0.6667 | 0.9820 |
| pgd_at_depth_l1.0 | query_underconf | combined_tr | 0.3411 | 0.5444 | 0.6005 | 1000 | 0.7090 | 0.2687 | 0.9330 |
| pgd_at_depth_l1.0 | query_underconf | depth_tr | 0.3411 | 0.5162 | 0.6165 | 1000 | 0.7090 | 0.2386 | 0.9120 |
| standard | clean | msp | 0.3516 | 0.4703 | 0.7284 | 999 | 0.2725 | 0.5870 | 0.7530 |
| standard | ace_uint8 | msp | 0.3516 | 0.0222 | 0.9401 | 1000 | 0.2725 | 0.8057 | 0.2690 |
| standard | corruption_jpeg | msp | 0.3514 | 0.4554 | 0.7472 | 1000 | 0.1746 | 0.6853 | 0.7490 |
| standard | pgd | msp | 0.3401 | 0.5396 | 0.6169 | 658 | 0.2804 | 0.3580 | 0.0140 |
| standard | query_overconf | msp | 0.3516 | 0.4838 | 0.7146 | 1000 | 0.2725 | 0.5677 | 0.3060 |
| standard | query_underconf | msp | 0.3516 | 0.5174 | 0.7025 | 998 | 0.2725 | 0.6400 | 0.9750 |
| standard_depth_l1.0 | clean | msp | 0.3454 | 0.5118 | 0.6747 | 1000 | 0.2884 | 0.5193 | 0.7150 |
| standard_depth_l1.0 | clean | combined | 0.3454 | 0.4467 | 0.7280 | 1000 | 0.2884 | 0.5164 | 0.7250 |
| standard_depth_l1.0 | clean | depth | 0.3454 | 0.4428 | 0.7314 | 1000 | 0.2884 | 0.5277 | 0.7650 |
| standard_depth_l1.0 | ace_uint8 | msp | 0.3454 | 0.0173 | 0.9422 | 998 | 0.2884 | 0.7619 | 0.2230 |
| standard_depth_l1.0 | ace_uint8 | combined_tr | 0.3454 | 0.1371 | 0.8958 | 1000 | 0.2884 | 0.7636 | 0.2260 |
| standard_depth_l1.0 | ace_uint8 | depth_tr | 0.3454 | 0.5003 | 0.6824 | 1000 | 0.2884 | 0.7614 | 0.3420 |
| standard_depth_l1.0 | corruption_jpeg | msp | 0.3463 | 0.4999 | 0.7045 | 1000 | 0.1746 | 0.6330 | 0.7030 |
| standard_depth_l1.0 | corruption_jpeg | combined | 0.3463 | 0.4613 | 0.7318 | 1000 | 0.1746 | 0.6382 | 0.7070 |
| standard_depth_l1.0 | corruption_jpeg | depth | 0.3463 | 0.4582 | 0.7323 | 1000 | 0.1746 | 0.6537 | 0.7430 |
| standard_depth_l1.0 | pgd | msp | 0.3284 | 0.5573 | 0.6299 | 424 | 0.7011 | 0.3784 | 0.0090 |
| standard_depth_l1.0 | pgd | combined | 0.3284 | 0.5347 | 0.6296 | 1000 | 0.7011 | 0.3796 | 0.0120 |
| standard_depth_l1.0 | pgd | depth | 0.3284 | 0.4932 | 0.6517 | 1000 | 0.7011 | 0.3868 | 0.4830 |
| standard_depth_l1.0 | query_overconf | msp | 0.3454 | 0.5557 | 0.6444 | 1000 | 0.2884 | 0.5617 | 0.1900 |
| standard_depth_l1.0 | query_overconf | combined_tr | 0.3454 | 0.4645 | 0.7087 | 1000 | 0.2884 | 0.5619 | 0.1920 |
| standard_depth_l1.0 | query_overconf | depth_tr | 0.3454 | 0.4590 | 0.7229 | 1000 | 0.2884 | 0.5846 | 0.2970 |
| standard_depth_l1.0 | query_underconf | msp | 0.3454 | 0.5586 | 0.6642 | 1000 | 0.2884 | 0.4706 | 0.9830 |
| standard_depth_l1.0 | query_underconf | combined_tr | 0.3454 | 0.4407 | 0.7263 | 1000 | 0.2884 | 0.5000 | 0.9820 |
| standard_depth_l1.0 | query_underconf | depth_tr | 0.3454 | 0.4510 | 0.7309 | 1000 | 0.2884 | 0.4615 | 0.9870 |

## Not collated

- standard_depth_l1.0__sid_set__query_underconf__combined_wb: no .done marker (failed or still running)

## What is NOT readable here

- Depth-score robustness against a GRADIENT adaptive attack on the residual (follow-up); `_wb` covers the gradient-free query attacks and the gradient confidence attacks through the teacher only.
- Anything about Track A/B arms: different fits, separate tables.
- A cell whose G2 verdict is DEGENERATE: its depth score is MSP relabelled, whatever its Phi.
