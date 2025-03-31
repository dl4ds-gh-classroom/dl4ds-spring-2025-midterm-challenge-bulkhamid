# Midterm Challenge Report: Parts 1–3

## 1. AI Disclosure

**AI Assistance:**
- **ChatGPT:** Used to help generate the initial draft of this report, suggest improvements for structuring text and code comments, and to provide ideas on experiment analysis.
- **GitHub Copilot:** Assisted with autocomplete suggestions during code development.

**Code Contributions:**
- **Written by Me:**
  - Overall project structure and integration of all three parts.
  - Manual implementation of the SimpleCNN model (Part 1).
  - Design of the experiments, hyperparameter tuning, and integration of evaluation functions.
  - Detailed analysis of results and discussion sections.
- **Assisted by AI:**
  - Initial drafts for various sections of the report.
  - Suggestions for organizing experiment tracking summaries and detailed code comments.

**Code Comments:**
- Each Python file includes comprehensive comments explaining data loading, model definitions, training procedures, evaluation, and any modifications made to the pretrained models.

---

## 2. Overview

This project consists of three distinct parts:

- **Part 1: SimpleCNN**
  - A manually defined convolutional neural network built from scratch using PyTorch. The model includes one convolutional layer, a ReLU activation, a pooling layer, and a fully connected layer for CIFAR-100 classification.

- **Part 2: Sophisticated CNN (Pretrained ResNet-18)**
  - Utilizes a pretrained ResNet-18 model from torchvision. The final fully connected layer is replaced to match the 100 classes in CIFAR-100, leveraging transfer learning to improve performance.

- **Part 3: Transfer Learning & Fine-Tuning**
  - Further fine-tunes the pretrained model from EfficientNet_B0. This phase includes advanced data augmentation, early stopping, and learning rate scheduling to achieve superior performance on CIFAR-100.

---

## 3. Model Descriptions

### Part 1 – SimpleCNN
- **Architecture:**
  - **Convolution Layer:** 16 filters with a 3×3 kernel, processing 3-channel input images.
  - **Activation:** ReLU to introduce non-linearity.
  - **Pooling:** Max Pooling to reduce spatial dimensions.
  - **Fully Connected Layer:** Flattens the output and maps it to 100 classes.
- **Justification:**  
  This model establishes a baseline and demonstrates a fundamental understanding of building a CNN from scratch.

### Part 2 – Sophisticated CNN (Pretrained ResNet-18)
- **Architecture:**
  - A ResNet-18 model pretrained on ImageNet is used.
  - The final fully connected layer is replaced with a new linear layer outputting 100 classes.
- **Justification:**  
  - **Transfer Learning:** Leverages robust features learned from a large dataset (ImageNet).
  - **Residual Learning:** Incorporates skip connections to enable the training of deeper networks.
  - **Efficiency:** Balances performance with computational demands.

### Part 3 – Transfer Learning & Fine-Tuning
- **Architecture:**
  - An EfficientNet_B0 model pretrained on ImageNet. The classifier was modified by replacing the last linear layer to match the 100 classes of CIFAR-100.
  - Includes aggressive data augmentation, early stopping, and learning rate scheduling.
- **Design Rationale:**  
  EfficientNet_B0 has shown state-of-the-art performance on various image recognition tasks. Its compound scaling approach helps achieve a good balance between accuracy and computational efficiency. The input images were resized to 224×224 to align with the model’s expectations, and ImageNet normalization was applied

---

## 4. Hyperparameter Tuning

### Part 1 (SimpleCNN)
- **Learning Rate:** 0.1 (baseline value)
- **Batch Size:** 8 (determined via initial experimentation)
- **Epochs:** 5 (for demonstrative purposes)
- **Observations:**  
  The SimpleCNN serves as a proof-of-concept, providing a reference point for more complex models.

### Part 2 (ResNet-18)
- **Learning Rate:** 0.01 with step decay (γ = 0.1 every 5 epochs)
- **Batch Size:** 32
- **Momentum:** 0.9
- **Epochs:** 50 with early stopping (patience of 5 epochs)
- **Observations:**  
  Pretrained weights allow faster convergence; early stopping prevents overfitting.

### Part 3 (Transfer Learning & Fine-Tuning)
- **Modifications:**
  - Continued fine-tuning with the same learning rate schedule.
  - Additional data augmentation and minor hyperparameter adjustments based on validation performance.
- **Observations:**  
  Fine-tuning helps push the test accuracy beyond the baseline, balancing learning rate decay and regularization to mitigate overfitting.

---

## 5. Regularization Techniques

- **Early Stopping:**  
  Training halts if the validation accuracy does not improve for 5 consecutive epochs to prevent overfitting.

- **Data Augmentation:**  
  - **Training Augmentations:** Random cropping (with 4-pixel padding) and random horizontal flipping.
  - **Test Augmentations:** Only normalization is applied.

- **Additional Techniques:**  
  Future work could explore dropout or weight decay for additional regularization.

---

## 6. Data Augmentation Strategy

### Training Transforms:
```python
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```
## 7. Results Analysis

### Part 1 – SimpleCNN
- **Performance:**  
  - Local Test Accuracy: **[21.34%]**
  - Kaggle Leaderboard Score: **[0.19720]**
- **Strengths:**
  - Simple architecture with minimal parameters, making it easy to train and understand.
  - Serves as a clear baseline for further experimentation.
- **Weaknesses:**
  - Limited capacity to capture complex image features.
  - Lower performance on out-of-distribution (OOD) data compared to more sophisticated models.

### Part 2 – Sophisticated CNN (Pretrained ResNet-18)
- **Performance:**  
  - Achieved a test accuracy of approximately **[0.20717]%** on CIFAR-100.
  - Local Test Accuracy: **[42.54%]**
- **Strengths:**
  - Transfer learning leverages robust features learned from ImageNet.
  - Residual connections enable deeper network training and improved convergence.
  - Better generalization than the SimpleCNN baseline.
- **Weaknesses:**
  - Requires higher computational resources.
  - Some signs of overfitting were observed, necessitating careful tuning of regularization and early stopping strategies.
  - Early stopped on **[23]th epoch**
  

### Part 3 – Transfer Learning & Fine-Tuning
- **Performance:**  
  - Achieved a test accuracy of approximately **[Z]%** on CIFAR-100, outperforming the baseline models.
  - Local Test Accuracy: **[79%]**
- **Strengths:**
  - Fine-tuning the pretrained ResNet-18 allows the model to adapt specifically to CIFAR-100.
  - Effective use of data augmentation and early stopping enhances robustness and mitigates overfitting.
  - Improved handling of OOD data compared to Parts 1 and 2.
- **Weaknesses:**
  - Some OOD distortions remain challenging.
  - Further hyperparameter optimization and additional regularization (e.g., dropout or weight decay) may yield incremental gains.

---

## 8. Experiment Tracking Summary

- **Tool:**  
  Experiments were tracked using **Weights & Biases (WandB)**.

- **Metrics Logged:**
  - **Training Metrics:** Loss and accuracy per epoch.
  - **Validation Metrics:** Loss and accuracy per epoch.
  - **Learning Rate Progression:** Monitored via a step scheduler.
  - **Model Checkpoints:** Saved best-performing models based on validation accuracy.

- **Run Summary:**
  - Detailed logs for each run, including hyperparameters such as batch size, learning rate, and epoch counts.
  - The WandB dashboard provides visualizations of training and validation metrics over time.
  - Example link to a run: [WandB Run Link](https://wandb.ai/your_username/your_project/runs/your_run_id) *(replace with actual URL)*.

- **Observations:**
  - Transfer learning in Parts 2 and 3 resulted in faster convergence and higher test accuracy compared to the SimpleCNN.
  - Early stopping was effective in preventing overfitting, as seen by the stabilization of validation loss.
  - Fine-tuning (Part 3) further improved performance, highlighting the benefits of adapting pretrained weights to the specific dataset.

---

## 9. Conclusion

This project explored three approaches for CIFAR-100 classification:

- **Part 1:**  
  A manually implemented SimpleCNN established a baseline.
  
- **Part 2:**  
  A pretrained ResNet-18 model was adapted for CIFAR-100, significantly improving performance through transfer learning.
  
- **Part 3:**  
  Fine-tuning the pretrained ResNet-18 with advanced data augmentation and early stopping achieved the best results.

**Summary of Findings:**
- The best performance was observed in Part 3, demonstrating that combining transfer learning with careful fine-tuning and regularization leads to improved accuracy.
- Regularization techniques like early stopping and data augmentation were key in reducing overfitting.
- Experiment tracking with WandB provided crucial insights into model performance and hyperparameter tuning.

**Future Directions:**
- Investigate additional regularization techniques such as dropout and weight decay.
- Experiment with deeper or alternative architectures (e.g., ResNet-34, DenseNet).
- Conduct ablation studies to further quantify the impact of each augmentation and hyperparameter choice.

