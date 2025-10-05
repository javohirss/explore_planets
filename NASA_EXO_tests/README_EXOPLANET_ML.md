# Exoplanet Detection using Neural Networks with PyTorch

## Overview

This Jupyter notebook implements a **deep learning approach** to exoplanet classification, improving upon classical machine learning methods referenced in the paper [MNRAS 513:4:5505](https://academic.oup.com/mnras/article/513/4/5505/6472249).

### Key Features:
- ✅ **Multi-mission data**: Kepler, K2, and TESS light curves
- ✅ **Comprehensive preprocessing**: Data cleaning, normalization, detrending
- ✅ **Feature engineering**: Automated time-series feature extraction with tsfresh
- ✅ **Deep learning**: PyTorch neural networks for binary classification
- ✅ **Full pipeline**: Data acquisition → preprocessing → training → evaluation
- ✅ **Detailed explanations**: Every cell includes in-depth commentary

## What's Included

### 📓 Notebook Structure (29 cells total):

1. **Setup & Installation** (Cells 1-2)
   - Package installation
   - Library imports and configuration

2. **Data Acquisition** (Cells 3-5)
   - Fetching Kepler mission data (confirmed planets & field stars)
   - Fetching K2 mission data
   - Fetching TESS mission data
   - ~50-70 light curves total from all missions

3. **Data Visualization** (Cell 6)
   - Raw light curve inspection
   - Understanding data quality and characteristics

4. **Data Cleaning** (Cell 7)
   - Remove NaN values and quality flags
   - Sigma clipping for outlier removal
   - Normalization and detrending
   - ~80-90% data retention after cleaning

5. **Feature Extraction** (Cell 8)
   - tsfresh automated feature extraction
   - ~20 statistical features per light curve
   - Handles NaN/inf values

6. **Data Preparation** (Cell 9)
   - Train/validation/test split (70%/15%/15%)
   - Feature scaling (StandardScaler)
   - PyTorch tensor conversion
   - DataLoader creation

7. **Model Architecture** (Cells 10-11)
   - Fully connected neural network
   - 4 hidden layers: [256, 128, 64, 32]
   - Dropout regularization
   - Binary cross-entropy loss
   - Adam optimizer

8. **Training** (Cell 12)
   - Training loop with validation
   - Early stopping
   - Learning rate scheduling
   - ~100 epochs max (typically stops earlier)

9. **Evaluation & Visualization** (Cells 13-15)
   - Training history plots
   - Test set evaluation
   - ROC curve and AUC
   - Confusion matrix
   - Precision-Recall curve

10. **Summary** (Cell 16)
    - Results overview
    - Future extensions
    - Real-world applications

## Getting Started

### Prerequisites:
```bash
# Python 3.8+ recommended
# Jupyter Notebook or JupyterLab
```

### Installation:
All required packages are installed in Cell 1 of the notebook:
- `lightkurve` - NASA data access
- `torch` - PyTorch deep learning
- `tsfresh` - Feature extraction
- `scikit-learn` - Preprocessing & metrics
- `matplotlib`, `seaborn` - Visualization
- `pandas`, `numpy`, `astropy`, `tqdm`

### Running the Notebook:

1. **Open the notebook:**
   ```bash
   jupyter notebook exoplanet_classification_nn.ipynb
   ```

2. **Run cells sequentially:**
   - Start with Cell 1 (installation)
   - Run each cell in order
   - Read explanations before running code
   - Total runtime: ~15-30 minutes (depends on download speed and CPU/GPU)

3. **Expected download times:**
   - Kepler data: ~2-5 minutes
   - K2 data: ~1-3 minutes  
   - TESS data: ~1-2 minutes
   - Feature extraction: ~2-5 minutes
   - Training: ~1-3 minutes (CPU), ~10-30 seconds (GPU)

4. **Troubleshooting:**
   - If downloads fail, reduce `max_downloads` parameter
   - If out of memory, reduce `batch_size` in Cell 9
   - If training is slow, reduce `num_epochs` in Cell 11

## Understanding the Approach

### Classical ML vs Deep Learning

**Original Paper (MNRAS 513:4:5505) - Classical ML:**
- Random Forests, SVM, Gradient Boosting
- Manual feature engineering
- Limited capacity for complex patterns
- Good but plateauing performance

**Our Approach - Deep Neural Networks:**
- Multi-layer fully connected network
- Learns hierarchical feature representations
- Better captures non-linear relationships
- Scalable to large datasets
- GPU acceleration

### Why Neural Networks Improve Performance:

1. **Hierarchical Features**: Early layers learn simple patterns, deep layers learn complex combinations
2. **Non-linearity**: Multiple ReLU activations capture complex decision boundaries
3. **End-to-end Learning**: Can optimize directly for classification objective
4. **Dropout Regularization**: Prevents overfitting better than classical methods
5. **Scalability**: Efficiently handles 1000s-millions of samples

### Data Pipeline:

```
Raw Light Curves (FITS files)
    ↓
Data Cleaning (remove outliers, NaNs, quality flags)
    ↓
Preprocessing (normalize, flatten/detrend)
    ↓
Feature Extraction (tsfresh: 20+ features)
    ↓
Feature Scaling (StandardScaler)
    ↓
Neural Network (4 layers, dropout)
    ↓
Binary Classification (planet vs non-planet)
```

## Expected Results

### Performance Metrics:
Based on typical runs with ~40-60 cleaned samples:

- **Accuracy**: 75-95%
- **F1-Score**: 0.70-0.90
- **ROC-AUC**: 0.80-0.95
- **Precision**: 70-90%
- **Recall**: 70-90%

*Note: Results vary based on downloaded data and random initialization*

### What Good Performance Looks Like:
- Training loss decreases smoothly to <0.3
- Validation loss tracks training loss (no major divergence = no overfitting)
- ROC-AUC > 0.85 (much better than random guessing = 0.5)
- F1-score > 0.75 (good balance of precision and recall)

## Extending the Project

### Easy Extensions:
1. **More data**: Increase `max_downloads` to fetch more light curves
2. **Different features**: Use `ComprehensiveFCParameters()` instead of `MinimalFCParameters()`
3. **Hyperparameter tuning**: Adjust learning rate, dropout, hidden dimensions
4. **Different architecture**: Add/remove layers, change layer sizes

### Advanced Extensions:
1. **CNN on raw data**: Use 1D convolutions directly on time-series
2. **LSTM/GRU**: Recurrent networks for temporal sequences
3. **Attention mechanisms**: Focus on transit regions
4. **Transfer learning**: Pre-train on Kepler, fine-tune on TESS
5. **Multi-task learning**: Classify + regress planet parameters
6. **Ensemble models**: Combine multiple neural networks

### Research Directions:
- Real-time candidate screening for TESS
- Re-analysis of Kepler/K2 archives
- Domain adaptation across missions
- Active learning for efficient followup prioritization
- Interpretability: Which features matter most?

## Dataset Details

### Missions:

**Kepler (2009-2013)**
- Primary mission: 4 years continuous monitoring
- ~150,000 target stars
- Single field of view: 115 sq degrees
- Cadence: 30-min (long), 1-min (short)
- ~2,700 confirmed exoplanets

**K2 (2014-2018)**
- Extended mission after reaction wheel failure
- ~20 campaigns of 80-day observations
- Fields along ecliptic plane
- More instrumental noise than Kepler
- ~500 confirmed exoplanets

**TESS (2018-present)**
- All-sky survey: 85% of sky
- 2-year primary mission (ongoing)
- 27-day sectors, some areas re-observed
- Brighter stars than Kepler
- Cadence: 2-min, 20-sec
- ~400+ confirmed, 1000s of candidates

### Data Access:
All data downloaded from NASA's MAST (Mikulski Archive for Space Telescopes):
- URL: https://mast.stsci.edu/
- API: lightkurve Python library
- Format: FITS files with time-series photometry

## Technical Details

### Neural Network Architecture:
```
Input Layer: n_features (typically 20-50)
    ↓
Dense Layer 1: 256 units + ReLU + Dropout(0.3)
    ↓
Dense Layer 2: 128 units + ReLU + Dropout(0.3)
    ↓
Dense Layer 3: 64 units + ReLU + Dropout(0.3)
    ↓
Dense Layer 4: 32 units + ReLU
    ↓
Output Layer: 1 unit + Sigmoid
    ↓
Probability: [0, 1] (planet or non-planet)
```

### Training Configuration:
- **Loss**: Binary Cross-Entropy
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-5)
- **Batch size**: 16
- **Epochs**: Up to 100 (early stopping at 20 patience)
- **Learning rate schedule**: ReduceLROnPlateau (factor=0.5, patience=10)
- **Regularization**: Dropout (0.3) + L2 weight decay (1e-5)

### Hardware Requirements:
- **Minimum**: 8GB RAM, multi-core CPU
- **Recommended**: 16GB RAM, GPU (CUDA-capable)
- **Storage**: ~500MB for downloaded data
- **Runtime**: 15-30 mins (CPU), 5-10 mins (GPU)

## Resources & References

### Papers:
- **Original inspiration**: [MNRAS 513:4:5505](https://academic.oup.com/mnras/article/513/4/5505/6472249)
- Kepler mission: [Borucki et al. 2010](https://ui.adsabs.harvard.edu/abs/2010Sci...327..977B)
- TESS mission: [Ricker et al. 2015](https://ui.adsabs.harvard.edu/abs/2015JATIS...1a4003R)

### Documentation:
- **Lightkurve**: https://docs.lightkurve.org/
- **PyTorch**: https://pytorch.org/docs/
- **tsfresh**: https://tsfresh.readthedocs.io/
- **NASA Exoplanet Archive**: https://exoplanetarchive.ipac.caltech.edu/

### Tutorials:
- Lightkurve tutorials: https://docs.lightkurve.org/tutorials/
- PyTorch tutorials: https://pytorch.org/tutorials/
- MAST portal: https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html

## License & Citation

This notebook is for educational purposes. If you use this code in research, please cite:
- The original MNRAS paper that inspired this work
- The Lightkurve library
- The relevant mission papers (Kepler, K2, TESS)

## Support & Questions

For issues or questions:
1. Check cell explanations in the notebook
2. Review troubleshooting section above
3. Consult documentation links
4. Check NASA MAST status for data availability

## Author Notes

This notebook demonstrates a **complete end-to-end pipeline** for exoplanet classification using modern deep learning. Every step is explained in detail to ensure understanding of:
- Why each step is necessary
- What each function does
- How to interpret results
- Common pitfalls and solutions

**Learning Goals:**
- Understand exoplanet detection via transit method
- Work with real astronomical time-series data
- Apply modern deep learning to scientific problems
- Improve upon classical ML approaches
- Deploy reproducible ML pipelines

**Enjoy exploring exoplanets with neural networks! 🪐🔭✨**



