# AMCNet
AMCNet (Autonomous Multi-Resolution Fusion Classification Network) is a classification network for complex traffic scene understanding in autonomous driving. It addresses the challenge of inter-class similarity by extracting distinctive features and establishing correlations between features across highly similar scenes.

📚 About AMCNet
AMCNet is an innovative classification network designed to extract discriminative features from complex traffic scenes and establish relationships between them. The network leverages multi-resolution feature extraction, feature differentiation screening, and feature fusion classification to achieve high accuracy and robust generalization.


🚀 Project Structure

AMCNet/

├── dataset/

│   ├── bdd100k/          # BDD100k dataset

│   ├── bdd114k/          # BDD114k dataset

│   └── proprietary/      # Proprietary dataset

├── models/

│   ├── amcnet.py         # Definition of the AMCNet model

│   ├── mre.py            # Multi-Resolution Extraction module

│   ├── fds.py            # Feature Differentiation Screening module

│   └── afb.py            # Feature Fusion module

├── utils/

│   ├── data_loader.py    # Data loading utilities

│   ├── metrics.py        # Evaluation metric calculations

│   └── visualization.py  # Visualization tools

├── train.py              # Training script

├── test.py               # Testing script

├── README.md             # Project documentation

└── LICENSE               # License file


🛠️ Installation
Before you start, please ensure you have the following dependencies installed. We recommend using Anaconda or Miniconda environments.

# Create and activate the environment

conda create -n amcnet python=3.7

conda activate amcnet

# Install dependencies

pip install torch==1.7.1 torchvision==0.8.2

pip install numpy matplotlib opencv-python scikit-learn


# Hyperparameter search configuration (configs/hparams_search.yml)
hyperparameters:
  learning_rate:
    min: 1e-4
    max: 1e-1
    log_scale: true
  dropout:
    values: [0.1, 0.2, 0.3, 0.5]
  batch_size:
    values: [16, 32, 64]
  optimizer:
    values: ["sgd", "adamw", "radam"]

Learning Rate Optimization:
  Conduct logarithmic sweeps (1e-1 to 1e-4)
  Use 5-epoch warmup followed by cosine annealing
  Final selection: LR=0.1 (initial), decaying to 1e-4
Dropout Rate Selection:
  Tested values: [0.1, 0.2, 0.3, 0.5]
  Evaluated via 5-fold cross-validation
  Optimal: 0.2 (minimal overfitting)
Batch Size Calibration:
  Tested sizes: [16, 32, 64]
  32 provides best memory/accuracy balance
  Gradient accumulation steps=2 when using batch size 16


🚀 Getting Started

Data Preparation

BDD100k Dataset

Download the BDD100k dataset and extract it to the dataset/bdd100k folder.

BDD114k Dataset

If you are using the augmented dataset, please follow the data augmentation methods described in the paper and place the dataset in the dataset/bdd114k folder.

Proprietary Dataset

Place your proprietary dataset in the dataset/proprietary folder and annotate it according to the guidelines in the paper.

Training the Model

Run the following command to start training the model:


python train.py --dataset bdd100k --epochs 50 --batch-size 16

Testing the Model

Run the following command to perform testing:

python test.py --model-path ./checkpoints/best_model.pth --dataset bdd100k


⚙️ Training Protocol
Our optimized training pipeline:

Data Augmentation
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(512),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

Optimization Schedule
Phase	Epochs	Optimizer	Parameters
Initial	1-30	SGD	lr=0.1, momentum=0.9
Mid-training	31-100	AdamW	lr=1e-3, weight_decay=1e-4
Fine-tuning	101-150	RAdam	lr=1e-4, gradient_clip=1.0

Execution Command
python train.py \
  --dataset BDD114K \
  --batch_size 32 \
  --dropout 0.2 \
  --lr_schedule cosine \
  --max_epochs 150 \
  --data_dir ./data/bdd114k
📊 Expected Results
Reproduction variance should be ≤0.3% from reported metrics:

Dataset	Accuracy	95% CI	Expected Range
BDD114k	93.41%	(93.27-93.55)	93.1-93.7%
Proprietary	80.41%	(79.83-80.99)	80.1-80.7%

🧪 Verification Tests
# Run end-to-end verification
pytest tests/ --benchmark

# Validate against Table 7 metrics
python verify_results.py --checkpoint pretrained/amcnet_bdd114k.pth
🚀 Quick Start
Clone repository:
git clone https://github.com/yourname/AMCNet.git
cd AMCNet
Install dependencies:
pip install -r requirements.txt
Download datasets:
python scripts/download_datasets.py
Train model:
python train.py --config configs/bdd114k.yml
Evaluate:
python eval.py --checkpoint outputs/model_best.pth

📈 Experimental Results

AMCNet achieves a classification accuracy of 91.44% on the BDD100k dataset, outperforming the current Top-2 network methods by a significant margin. On the BDD114k dataset, the accuracy further improves to 93.41%. Additionally, AMCNet demonstrates strong performance on the proprietary dataset, with an accuracy of 80.41%.


📈 Performance Highlights

Multi-Resolution Feature Extraction: The Multi-Resolution Extraction (MRE) module enables the network to capture features at different scales, enhancing the understanding of complex scenes.

Feature Differentiation Screening: The Feature Differentiation Screening (FDS) module focuses the network on more discriminative features, reducing the impact of inter-class similarity.

Feature Fusion Classification: The Feature Fusion module (AFB) integrates local details and global semantic information to further improve classification accuracy.


📝 License

This project is licensed under the MIT License.


📝 Contact

For any questions or suggestions, please feel free to contact us via:

Yanji Jiang: jyjvip@126.com

GitHub Issues: Submit an issue on GitHub Issues

