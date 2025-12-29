## Food Freshness Detection using CNN

### Abstract

This project implements a Convolutional Neural Network (CNN) to classify food items as fresh or rotten using image data. Using VGG19 as the base architecture with transfer learning, we address five key research questions about CNN-based food freshness detection, visual feature analysis, preprocessing impacts, environmental robustness, and practical deployment considerations.

### Research Questions

1. **RQ1: Dataset Characteristics** - How is the food freshness dataset structured in terms of class distribution, balance, and visual characteristics of fresh vs. rotten samples?
2. **RQ2: Training Performance** - How does the VGG19 architecture with transfer learning perform during training, and what patterns emerge in the learning curves?
3. **RQ3: Data Augmentation Impact** - How does data preprocessing and augmentation affect model performance in food freshness classification across different food categories?
4. **RQ4: Visual Feature Interpretation** - Which visual features and spatial regions does the CNN prioritize when identifying food spoilage using Grad-CAM analysis?
5. **RQ5: Deployment Readiness** - How effective is the system for practical deployment in terms of test accuracy, per-category performance, inference speed, and single-image classification reliability?

### Dataset

- **Source**: Food Freshness Dataset with 13 food types
- **Structure**:
  - Fresh: Apple, Banana, Bellpepper, Bittergourd, Capsicum, Carrot, Cucumber, Mango, Okra, Orange, Potato, Strawberry, Tomato (~47,345 images)
  - Rotten: Same 13 food types (~23,977 images)
- **Total Classes**: 26 (13 Fresh + 13 Rotten)
- **Total Images**: ~71,322

### Technologies

- Python 3.x
- PyTorch
- torchvision (VGG19 pre-trained model)
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn

### Project Structure

```
food-freshness/
├── Dataset/
│   ├── Fresh/
│   │   ├── FreshApple/
│   │   ├── FreshBanana/
│   │   └── ... (13 folders)
│   └── Rotten/
│       ├── RottenApple/
│       ├── RottenBanana/
│       └── ... (13 folders)
├── model/
│   ├── Food_Freshness_Detection.ipynb
│   ├── cat_to_name.json
│   └── food_freshness_best.pth (generated after training)
├── Figures_Tables/
│   ├── RQ1/ (Dataset distribution, samples, statistics)
│   ├── RQ2/ (Learning curves, training metrics, model config)
│   ├── RQ3/ (Validation metrics, augmentation comparison)
│   ├── RQ4/ (Grad-CAM heatmaps, feature visualizations)
│   └── RQ5/ (Confusion matrix, per-category metrics, deployment analysis, inference examples)
└── README.md
```

### How to Run

#### Prerequisites

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn pillow openpyxl
```

#### Training the Model

1. Ensure dataset is in the correct structure under `Dataset/`
2. Open and run `model/Food_Freshness_Detection.ipynb`
3. The notebook will:
   - Load and preprocess the dataset
   - Train VGG19 model with transfer learning
   - Answer all 5 research questions with experiments
   - Export figures (PDF) and tables (Excel) to `Figures_Tables/`

#### Inference

The trained model can classify new food images as fresh or rotten for any of the 13 supported food types.

### Model Architecture

- **Base**: VGG19 (pre-trained on ImageNet)
- **Modifications**:
  - Frozen convolutional layers (feature extractor)
  - Custom classifier head for 26 classes
  - Dropout for regularization
- **Input Size**: 224×224 pixels
- **Output**: 26-class softmax (Fresh/Rotten for each food type)

### Key Results & Outputs

**RQ1:** Dataset analysis showing class balance, fresh vs. rotten distribution, and visual sample characteristics
**RQ2:** Training/validation curves demonstrating convergence, with loss and accuracy trends across epochs
**RQ3:** Quantitative metrics showing augmentation improves validation accuracy by ~7% over baseline
**RQ4:** Grad-CAM visualizations revealing the model focuses on discoloration, texture changes, and spoilage patterns
**RQ5:** Test accuracy metrics, normalized confusion matrix, per-category precision analysis, inference benchmarks (~10-50ms), and single-image prediction examples

### Team

- Student 1 (Technical Lead): [Azizbek Khushvakov]
- Student 2 (Technical Report): [Mukhriddin Aktamov]
- Student 3 (Presentation): [Manoj Shivalingaiah]

### Submission Contents

1. **Jupyter Notebook** (`Food_Freshness_Detection.ipynb`): Complete implementation with all 5 RQs
2. **GitHub Repository**: Full source code and dataset instructions
3. **Figures_Tables.zip**: Organized outputs (RQ1-RQ5 folders with PDFs and Excel files)

### References

- VGG19: Simonyan & Zisserman (2014)
- Transfer Learning for Image Classification
- Grad-CAM: Visual Explanations from Deep Networks

### License

Academic Project - Pattern Recognition Course