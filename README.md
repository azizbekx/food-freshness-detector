## Food Freshness Detection using CNN

### Abstract

This project implements a Convolutional Neural Network (CNN) to classify food items as fresh or rotten using image data. Using VGG19 as the base architecture with transfer learning, we address five key research questions about CNN-based food freshness detection, visual feature analysis, preprocessing impacts, environmental robustness, and practical deployment considerations.

### Research Questions

1. **Can a CNN accurately distinguish between fresh and spoiled food items?**
2. **Which visual features (color, texture) contribute most to detecting food spoilage?**
3. **How does data preprocessing and augmentation affect performance?**
4. **What is the impact of image quality and environmental factors on accuracy?**
5. **Can this provide a practical, non-invasive solution for food quality monitoring?**

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
│   ├── RQ1/ (Dataset analysis figures & tables)
│   ├── RQ2/ (Model configuration)
│   ├── RQ3/ (Augmentation comparison)
│   ├── RQ4/ (Feature visualization & Grad-CAM)
│   └── RQ5/ (Deployment metrics & Confusion Matrix)
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

### Key Results

_Results will be generated when running the notebook. See Figures_Tables/ for:_

- Classification accuracy and metrics
- Feature importance visualizations (Grad-CAM)
- Augmentation impact analysis
- Per-category performance breakdown
- Inference time and deployment feasibility

### Team

_[Add your group number and team member names/roles here]_

- Student 1 (Technical Lead): [Name]
- Student 2 (Figures/Tables): [Name]
- Student 3 (Report/Storytelling): [Name]

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

---

_This project demonstrates practical application of CNNs for food quality assessment, with potential real-world applications in food safety and waste reduction._
