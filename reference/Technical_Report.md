# Technical Report: The Sharpness Quest

## 1. Introduction & Hypothesis

**Hypothesis:** Adding a discriminator term to the loss function of a Conditional Variational Autoencoder (cVAE) during training increases the sharpness of the reconstructed images.

Standard Variational Autoencoders (VAEs) are known to suffer from a "regression-to-the-mean" problem. The pixel-wise reconstruction loss (MSE) penalises outliers by averaging over plausible outputs, resulting in blurry, washed-out images. This phenomenon has been well-documented in the literature (Kingma & Welling, 2014; Larsen et al., 2016).

Several prior works have attempted to mitigate this blurriness by combining VAEs with adversarial training:
- **Larsen et al. (2016)** proposed the VAE-GAN architecture, demonstrating that replacing the pixel-wise reconstruction loss with a feature-wise loss from a discriminator produces visually sharper images.
- **Rosca et al. (2017)** provided theoretical analysis showing that adversarial objectives can sharpen the implicit data distribution learned by VAEs.

This project investigates whether incorporating an adversarial discriminator loss into the cVAE training pipeline can force the model to generate clearer strokes and sharper edges on a custom 20×20 MNIST dataset.

## 2. Methodology & Architectures

### 2.1 Models

Two models are compared, both sharing identical Encoder and Decoder architectures with a latent dimension of $z = 7$:

1. **Baseline cVAE:** Trained with the standard VAE objective:
$$\mathcal{L}_{VAE} = \text{MSE}(x, \hat{x}_z) + \beta \cdot \text{KL}(z \| p(z))$$

2. **Modified cVAE (with Discriminator Loss):** Extends the baseline with an adversarial discriminator, introducing two additional loss terms as specified in the task:

$$\mathcal{L}_D = \mathbb{E}_x[\log(1 - D(x))] + \mathbb{E}_z[\log(D(G(z)))]$$
$$\mathcal{L}_G = \mathbb{E}_z[\log(1 - D(G(z)))]$$

The total generator loss combines the standard VAE loss with the adversarial term:
$$\mathcal{L}_{total} = \mathcal{L}_{VAE} + \alpha \cdot \mathcal{L}_G$$

where $\alpha = 0.5$ and $\beta = 0.1$.

### 2.2 Training Configuration

| Parameter | Baseline | Modified |
|-----------|----------|----------|
| Optimiser | AdamW (lr=0.001, β=(0.75, 0.99)) | AdamW (lr=0.001, β=(0.75, 0.99)) |
| Discriminator Optimiser | N/A | AdamW (lr=0.0005, β=(0.5, 0.999)) |
| Early Stopping Patience | 5 epochs | 5 epochs |
| Training Convergence | Epoch 30 (Val Loss: 2.60) | Epoch 30 (Val Loss: −6.66) |

The negative validation loss for the modified model is expected, as the adversarial term $\mathcal{L}_G = \log(1 - D(G(z)))$ produces negative values when the generator successfully fools the discriminator.

## 3. Visual Comparison & Qualitative Evaluation

Both models generated a 12×10 grid of digits (12 samples per class, 120 total) using identical random latent vectors $z$, ensuring a direct one-to-one comparison.

**Baseline cVAE Grid**
![Baseline Generated Digits](plot/baseline_grid.png)

**Discriminator-Enhanced cVAE Grid**
![Modified Generated Digits](plot/modified_grid.png)

*Observation:* Both models produce recognisable digits across all 10 classes. Visual differences between the two models are subtle at this resolution, necessitating quantitative evaluation to rigorously assess the hypothesis.

## 4. Quantitative Evaluation

### 4.1 Sharpness Metric: Laplacian Variance

The **Variance of the Laplacian** is a standard image processing metric for edge detection. The Laplacian operator highlights rapid intensity changes (edges), and its variance quantifies overall edge content:
- **Higher variance → sharper images** with stronger edge definition
- **Lower variance → blurrier images** with smoother, washed-out appearance

![Sharpness Comparison](plot/sharpness_comparison.png)

| Model | Laplacian Variance |
|-------|-------------------|
| Baseline cVAE | **0.07** |
| Modified cVAE | **0.07** |

*Result:* Both models achieved identical sharpness scores of 0.07. The discriminator loss did not produce a measurable increase in Laplacian variance in this configuration. This suggests that the adversarial pressure, at the current weight ($\alpha = 0.5$) and with this specific loss formulation, is insufficient to significantly alter the edge sharpness of the generated images.

### 4.2 Semantic Consistency: Classifier Accuracy

A standalone CNN classifier (trained on real MNIST data) evaluated the recognisability of generated digits. This ensures the adversarial loss did not distort digit geometry.

![Class Consistency Comparison](plot/consistency_comparison.png)

| Model | Classification Accuracy |
|-------|------------------------|
| Baseline cVAE | **96.56%** |
| Modified cVAE | **96.38%** |

*Result:* Both models maintain exceptionally high semantic consistency (~96%), confirming that the discriminator loss did not degrade the semantic quality of the generated digits.

### 4.3 Distribution Quality: Fréchet Classifier Distance (FCD)

FCD measures the distributional distance between real and generated images in the classifier's feature space. Lower values indicate generated images whose statistics better match real data.

![FCD Comparison](plot/fcd_comparison.png)

| Model | FCD | Precision | Recall |
|-------|-----|-----------|--------|
| Baseline cVAE | **7.11** | 0.786 | 0.098 |
| Modified cVAE | **7.18** | 0.785 | 0.168 |

*Result:* The FCD scores are nearly identical (7.11 vs 7.18), indicating comparable overall distribution quality. Notably, the modified model shows a significant improvement in **Recall** (0.098 → 0.168, a 71% increase), meaning it covers a substantially larger portion of the real data distribution. This suggests the discriminator loss improves **diversity** of generated samples, even though it did not significantly impact sharpness in this experiment.

## 5. Advanced Latent & Reconstruction Analysis

### 5.1 Reconstruction-based Classification

We passed real test images through each model's encoder-decoder pipeline and classified the reconstructions using the standalone classifier.

![Reconstruction Confusion Matrices](plot/reconstruction_classification.png)

*Result:* Both models faithfully preserve digit identity during reconstruction, with strong diagonal dominance in the confusion matrices.

### 5.2 Latent Space Analysis

**2D Histogram of Latent Means**
![2D Histogram](plot/latent_2d_hist.png)

**Scatterplot of Latent Means (Coloured by Class)**
![2D Scatterplot](plot/latent_2d_scatter.png)

*Observation:* Both models organise the latent space into class-discriminative clusters. The modified model shows slightly different structural arrangements due to adversarial pressure, but both achieve effective separation.

**Kernel Density Estimates (KDE)**
![KDE Analysis](plot/latent_kde_analysis.png)

*Observation:* The KDE distributions of μ and σ demonstrate how closely the encoded variables conform to the standard normal prior. Both models show similar distributional characteristics.

**Latent Vector Norms & Angular Distribution**
![Norms and Angles](plot/latent_norms_angles.png)

*Observation:* The L2 norms and angular distributions are tightly bounded for both models, validating that the adversarial loss did not destabilise the KL-regularised prior.

## 6. Discussion

The experimental results present a nuanced picture:

1. **Sharpness (Laplacian Variance):** Contrary to the hypothesis, the discriminator loss did not produce a measurable increase in image sharpness (both models scored 0.07). This may be attributed to the specific loss formulation $\mathcal{L}_G = \log(1 - D(G(z)))$ which is known to suffer from vanishing gradients when the generator is already producing reasonable outputs (Goodfellow, 2014).

2. **Diversity (Recall):** The modified model showed a notable improvement in recall (0.098 → 0.168), suggesting the adversarial pressure encourages the generator to explore a broader range of the data manifold.

3. **Quality Preservation:** Semantic consistency (~96%) and FCD (~7.1) remained stable, confirming that the discriminator loss did not introduce harmful artefacts.

4. **Limitations:** The adversarial weight $\alpha$, discriminator architecture, and training dynamics (learning rates, alternating update frequency) are all hyperparameters that significantly influence the outcome. Different configurations may yield different sharpness results. Additionally, the 20×20 image resolution may limit the observable differences in edge sharpness.

## 7. Conclusion

The experimental evidence does not strongly support the hypothesis that adding a discriminator term increases the sharpness of generated images in the current configuration. While both models achieve comparable sharpness and semantic consistency, the discriminator-enhanced model demonstrates improved sample diversity (71% higher recall). This suggests that adversarial training may primarily benefit VAEs by encouraging mode coverage rather than per-image sharpness at this scale. Future work could explore alternative loss formulations (e.g., the non-saturating generator loss), higher adversarial weights, or perceptual loss functions to more directly target sharpness improvement.

## References

- Kingma, D.P. & Welling, M. (2014). Auto-Encoding Variational Bayes. *ICLR 2014*.
- Larsen, A.B.L., Sønderby, S.K., Larochelle, H. & Winther, O. (2016). Autoencoding beyond pixels using a learned similarity metric. *ICML 2016*.
- Goodfellow, I.J. et al. (2014). Generative Adversarial Nets. *NeurIPS 2014*.
- Rosca, M., Lakshminarayanan, B., Warde-Farley, D. & Mohamed, S. (2017). Variational Approaches for Auto-Encoding Generative Adversarial Networks. *arXiv:1706.04987*.
