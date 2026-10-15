# Development Log: The Sharpness Quest

## 1. Codebase Refactoring & Centralization
*   **Approach:** The initial notebooks (`cVAE_DiscriminatorLoss.ipynb` and `main_report.ipynb`) contained duplicated model definitions for `cEncoder`, `cDecoder`, `cVAE`, and `cDiscriminator`. 
*   **What Worked:** Moving all PyTorch neural network classes and shared loss functions into a central `models.py` file. This guaranteed structural parity between the training and evaluation phases.
*   **Resolution:** Both notebooks were refactored to simply import from `models.py`, vastly reducing code clutter and preventing mismatched definitions during evaluation.

## 2. Advanced Metrics Implementation
*   **Approach:** Implementing FCD (Fréchet Classifier Distance), along with K-Nearest Neighbors Precision/Recall, to rigorously evaluate mode collapse and distribution quality.
*   **What Didn't Work:** 
    *   FCD calculations triggered `SciPy LinAlgWarning: Matrix is singular` errors due to the inherently low variance of the latent feature distributions. 
    *   OpenCV's `cv2.Laplacian` threw format errors because PyTorch tensors were cast to `float64` but OpenCV expected `float32`.
    *   Evaluating the classifier inside the notebook execution bloated the main report.
*   **Resolution:** 
    *   The classifier training was decoupled into a standalone script (`train_classifier.py`), which pre-trains and saves `classifier_weights.pt` to the `assets/` directory.
    *   `LinAlgWarning` was safely suppressed since taking the principal square root of a singular covariance matrix in FCD is standard practice.
    *   Fixed OpenCV Laplacian by explicitly enforcing `cv2.CV_32F` during numpy conversion.

## 3. Generative Adversarial Mechanics (The VAE-GAN Tradeoff)
*   **Approach:** In an attempt to further smooth out the generated digits and prevent "fractured" strokes caused by the Discriminator clashing with the VAE's MSE loss, we experimented with advanced mechanics: **Feature Matching** and **KL-Divergence Annealing**.
*   **What Didn't Work:** Adding Feature Matching drastically disrupted the semantic conditionality. While the digits became extremely diverse (Recall skyrocketed), they became completely unrecognizable, causing Semantic Consistency (Accuracy) to plummet to ~10%.
*   **Resolution:** Per the user's explicit request, we reverted to the pure, basic loss function: Pixel-wise MSE + BCE Adversarial Loss. The retrained model struck a mathematically perfect balance:
    *   **Sharpness:** Doubled (0.07 -> 0.17).
    *   **Distribution Quality (FCD):** Plunged (10.09 -> ~3.98).
    *   **Diversity (Recall):** Quadrupled (10% -> 40%).
    *   **Consistency:** Held strong at ~94%.

## 4. Visual Layout & Storytelling
*   **Approach:** The initial evaluation report relied on standalone, detached bar graphs to present metrics. The quest requirements also specified exactly 12 generated samples per digit class.
*   **What Didn't Work:** 
    *   Bar graphs lacked context; it was impossible to visually confirm *why* the Sharpness metric doubled.
    *   The `cVAE` reconstruction plots were misaligned with the 20x20 custom MNIST dataset (the y-ticks were centered for 28x28 digits).
*   **Resolution:** 
    *   Developed a `plot_metric_with_samples` utility that generates 1x3 subplots: a grid of Baseline samples, a grid of Modified samples, and the corresponding Bar Graph side-by-side. 
    *   Fixed the `y-ticks` to `[12, 34, 56]` to perfectly align with 20x20 padded images.
    *   Built a unified 10x24 generation grid with a thick vertical separator line explicitly dividing the Baseline and Modified outputs.

## 5. Output Cleanliness
*   **Approach:** Jupyter notebooks executed headlessly via `nbconvert` capture terminal output indiscriminately.
*   **What Didn't Work:** The `tqdm` progress bars spawned thousands of lines of raw text, bloating the notebook's JSON structure and making it unreadable.
*   **Resolution:** Authored `format_outputs.py` to parse and strip redundant `tqdm` spam from the `.ipynb` files, ensuring the final submitted notebooks are perfectly clean and professional.

## 6. Loss Formula Correction (Final Iteration)
*   **Approach:** Cross-referenced the implementation against the exact formulas provided in `image1.png` (Task 2).
*   **What Didn't Work:** The original implementation used the standard GAN convention:
    - `ℒ_D = BCE(D(x), 1) + BCE(D(G(z)), 0)` = `-log(D(x)) - log(1-D(G(z)))`
    - `ℒ_G = BCE(D(G(z)), 1)` = `-log(D(G(z)))` (non-saturating trick)
*   **Issue:** The image specifies different formulas:
    - `ℒ_D = E_x[log(1 - D(x))] + E_z[log(D(G(z)))]`
    - `ℒ_G = E_z[log(1 - D(G(z)))]`
*   **Resolution:** Replaced all BCE-based loss computations with raw log formulas using `eps = 1e-8` for numerical stability. Both training and validation loops were updated consistently.
*   **Impact:** After retraining with the corrected formulas, the sharpness metric (Laplacian Variance) showed identical scores for both models (0.07 vs 0.07), while Recall improved from 0.098 to 0.168. The corrected `log(1-D(G(z)))` generator loss is known to have vanishing gradients when the generator is already performing well, which may explain the lack of sharpness improvement compared to the non-saturating variant.
