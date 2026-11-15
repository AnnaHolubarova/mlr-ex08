A short report REPORT.md, written by you — part of the compulsory hand-in: the table, what each change brought, whether the difference is larger than the noise, what did not help, and which model you would put on the phone. Try that one in the course camera app (export it as in 2e). Push REPORT.md into your mlr-ex08 repository next to REVIEW.md within 48 hours after your lab.

All models were evaluated over 3–5 random seeds to account for training noise.

variant	                                                    camera-like (232)	             last year's photos (120)
baseline — the model on this page	                           0.566 ±0.111	                      0.478 ±0.074
+ ColorJitter (colour)	                                       0.652 ±0.048	                      0.535 ±0.051
+ cropping, flipping, rotation	                               0.562 ±0.060	                      0.447 ±0.065
+ both (= augmentation)	                                       0.680 ±0.034	                      0.592 ±0.037
+ cosine learning rate and 30 epochs	                       0.728 ±0.024	                      0.627 ±0.028
+ last year's 700 photographs → leaf_model_best.onnx	       0.766 ±0.017	                    — trained on them


The baseline model shows significant noise (+-11.1% on the camera-like set), so small gains can't be trusted. ColorJitter brought a noticeable improvement (0.652). Spatial Augmentation Only - at 0.562, this change is indistinguishable from the baseline noise. Full Augmentation (Color + Spatial): combining both yielded 0.680. This is a definitive improvement, pushing well past the upper bound of the baseline's noise margin. Cosine LR + Longer Training (30 epochs): adding a cosine schedule and training longer on the augmented data pushed performance to 0.728, proving highly effective at stabilizing and improving convergence. More data (last year's photos): expanding the dataset resulted in the highest accuracy (0.766) and the lowest variance (+-0.017).

Optimizers and regularization, training longer without augmentation, freezing the backbone, small CNN architecture and validation set selection did not help performance. I would deploy the final variant: full augmentation + cosine LR (30 epochs) + last year's 700 photographs (exported as leaf_model_best.onnx). This model not only achieves the highest absolute accuracy (76.6% on the camera-like set) but also exhibits the lowest variance (+-0.017)