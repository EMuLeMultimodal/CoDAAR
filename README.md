# 📊 Cross-Modal Discrete Alignment And Reconstruction (CoDAAR)

PyTorch implementation for multimodal learning with discrete representations.

---

## 📝 Requirements and Installation

### Getting Started
```bash
git clone [your-repo-url]
cd src_cvpr

# Create environment with Python 3.10.16
conda create -n codaar python=3.10.16
conda activate codar

# Install requirements
pip install -r requirements.txt
```

---

## 🚀 Training and Evaluation

### Pretraining

**Audio-Visual (AV) Setting:**
```bash
cd src_cvpr
sbatch pretrain_novel_AV.sbatch
```

**Audio-Visual-Text (AVT) Setting:**
```bash
cd src_cvpr
sbatch pretrain_novel_AVT.sbatch
```

### Downstream Tasks

**AVE:**
```bash
cd src_cvpr
sbatch ave_novel.sbatch
```

**AVVP:**
```bash
cd src_cvpr
sbatch avvp_novel.sbatch
```

**AVE→AVVP:**
```bash
cd src_cvpr
sbatch ave_avvp_novel.sbatch
```

**UCF-VGGSound:**
```bash
cd src_cvpr
sbatch ucf_vggsound_novel.sbatch
```

**AVS:**
```bash
# Training
cd src_cvpr/AVSBench_downstream/avs_scripts/avs_s4
sbatch train_novel.sbatch

# Testing
cd src_cvpr/AVSBench_downstream/avs_scripts/avs_s4
sbatch test_novel.sbatch
```

---

## 📂 Data Preparation

Dataset CSV files are located in `src_cvpr/data/`

### Dataset Sources:
- **VGGSound90k**: [https://github.com/hche11/VGGSound](https://github.com/hche11/VGGSound)
- **VGGSound40k**: [https://github.com/jasongief/CPSP](https://github.com/jasongief/CPSP)
- **AVE**: [https://github.com/YapengTian/AVE-ECCV18](https://github.com/YapengTian/AVE-ECCV18)
- **AVVP**: [https://github.com/YapengTian/AVVP-ECCV20](https://github.com/YapengTian/AVVP-ECCV20)
- **AVSBench-S4**: [https://github.com/OpenNLPLab/AVSBench](https://github.com/OpenNLPLab/AVSBench)
- **UCF-101**: [https://www.crcv.ucf.edu/data/UCF101.php](https://www.crcv.ucf.edu/data/UCF101.php)

### Feature Extraction
For video and audio feature extraction methods, please refer to [AVE](https://github.com/YapengTian/AVE-ECCV18).

---
