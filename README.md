## Paper overview and Graphical Representation
Drug-drug interactions(DDIs) can lead to unexpected adverse side effects, presenting significant challenges in both research and clinical environments. Although numerous DDIs have been documented, the underlying mechanisms remain inadequately understood. The precise prediction and analysis of DDIs are essential for improving drug safety and ensuring patient health. In this study, we present DDINet, a predictive model that leverages Morgan fingerprints to assess DDIs based on the structural characteristics of drugs. We utilize a range of data-splitting methodologies with a fundamental neural network to evaluate novel drug combinations. The findings from DDINet indicate superior performance compared to the baseline model when tested on the independent hold-out set, and it exhibits robust efficacy on imbalanced datasets, peculiarly in multi-class classification. Additionally, it demonstrates commendable results using the scaffold-splitting approach.

![Data_and_Model_diagram-min_optimized_10000](https://github.com/user-attachments/assets/28dcfe9a-3a26-4443-b175-9457162a0b25)


## Experimental Setup
Our model was developed using Python 3.7.10, TensorFlow 2.4.1, and Keras 2.4.3, running on Spyder 4.2.5. The setup incorporates functionalities from NumPy 1.19.5, Pandas 1.2.4, Scikit-learn 0.24.2, Scikit-image 0.18.1, and RDKit 2023.03.2. All experiments were conducted with CUDA 11.2 on an NVIDIA Titan Xp GPU with 12 GB memory.
