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

