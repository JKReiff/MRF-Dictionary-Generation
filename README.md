# Accelerated Dictionary Generation with Machine Learning for Magnetic Resonance Fingerprinting


# MRF-Dictionary-Generation


Abstract:
This project explores the use of machine learning to accelerate the creation of dictionaries for Magnetic Resonance Fingerprinting (MRF). MRF allows for rapid acquisition of tissue properties in MRI. Traditionally, these dictionaries are created through time-consuming simulations. This project demonstrates how a neural network can significantly reduce this processing time.

Introduction
Magnetic Resonance Fingerprinting (MRF) is a powerful imaging technique used in MRI to obtain multiple tissue properties simultaneously. It achieves this by comparing measured signals with a pre-computed database known as a "dictionary." However, generating these dictionaries through simulations can be a bottleneck, hindering the clinical adoption of MRF.

This project addresses this challenge by leveraging machine learning. It proposes utilizing a neural network to learn the simulation process, leading to significantly faster dictionary 



## Background

Currently, creating this database involves time-consuming simulations. To make MRF clinically viable, this process needs to be significantly accelerated. This project is based on the paper "Machine Learning for Rapid Magnetic Resonance Fingerprinting Tissue Property Quantification" by Hamilton and Seiberlich (DOI: 10.1109/JPROC.2019.2936998), which proposes using neural networks for this purpose.

## Project Goal

The goal of this project was to train a neural network (GRU) that learns the simulation of the MRF database, thereby significantly speeding up its creation. 
![image](https://github.com/user-attachments/assets/d2073aad-3b12-41f0-8b32-f944737f4d41)
![image](https://github.com/user-attachments/assets/65a0c357-ae7b-4ab7-88e1-41ecf01166a9)

## Methodology

The workflow can be divided into the following steps:

1.  **Training Data Generation:** 1000 variations of heart rate (RR intervals) were generated for different age groups, and for each of these variations, 1600 simulated T1 and T2 combinations were generated. The data was normalized, shuffled, and split into training, validation, and test sets.

<img width="985" alt="1" src="https://github.com/user-attachments/assets/7b9b6507-2e2f-4866-8e5e-ff74244ba86c" />


2.  **Neural Network Training:** A specific type of neural network called a Gated Recurrent Unit (GRU) with an attention layer was chosen for this task. It was trained using the generated dataset, allowing it to learn the relationship between heart rate variations and the corresponding dictionary entries.The model's performance was visualized using the mean absolute error (MAE).

<img width="970" alt="2" src="https://github.com/user-attachments/assets/8e0a9955-0dfa-49c7-b9e4-7ca23bf372cd" />

3.  **Integration and Testing with In Vivo Data:** After training, the model was integrated into the image processing pipeline. Its performance was evaluated using real patient data, confirming its effectiveness in real-world scenarios.

![image](https://github.com/user-attachments/assets/e6e83538-4098-4b02-976f-f3e8a191ab10)


## Results

The developed model successfully reproduced the simulation data and showed very small deviations, especially in the myocardium (heart wall) region. Compared to the results in the Hamilton and Seiberlich paper, significantly higher precision and accuracy were achieved. The maximum deviation from the simulation was only 1.75 ms.

Using the neural network enabled a significant acceleration of database creation:

*   **Total time with denormalization:** 6.01 seconds (of which dictionary generation: 2.60 seconds)
*   **Simulation time with EPG:** 240.26 seconds

This corresponds to a reduction of approximately **97.5%** in computation time.

## Conclusion

This project demonstrates the great potential of neural networks for accelerating MRF database creation and paves the way for improved clinical applicability of the MRF technique.


## References

*   Hamilton and Seiberlich. Machine Learning for Rapid Magnetic Resonance Fingerprinting Tissue Property Quantification. DOI: 10.1109/JPROC.2019.2936998























 

# Code Module Descriptions and Usage


## 0_RR_interval_sim
This module generates RR intervals for simulating patient heartbeats.
- **Script**: `RR_Intervals_Var_PhysioNet.m`
  - **Configuration**: Specify the number of "patients" (equivalent to 14 heartbeats) to generate.
  - `msize` tetermines how many RR-Intervals should be generated
**Filter Criteria**:
- Healthy Subjects from Physionet.org
- 1000 RR Interval Rhythms extracted from 24-hour measurements
- Sample size: 10 Persons aged 20-55

**Output**:
- `RR_Matrix_Full.txt`: A text file containing all RR intervals.

![image](https://github.com/user-attachments/assets/ea686945-4e6f-4d23-86fa-51f07e4cafcb)

![image](https://github.com/user-attachments/assets/35b46754-2e13-4c48-af0b-7d9a334f362a)


## 1_Data_Generator
Generates training data by simulating a dictionary (EPG) for each RR interval.
- **Important Inputs**:
  - `Dictlist = 'T1T2_lowfield_invivo'`: Defines the combinations of T1, T2, etc., to be generated. See `simulation_function` folder under `loadDictionarylist_v4`.
  - `RR_matrix = importdata('RR_Matrix_Full.txt')`: Imports the RR Interval Matrix generated by `0_RR_interval_sim`.
- **Outputs**:
  - `T1_T2_Combinations.csv`: A file listing all T1 and T2 combinations used.
  - `RR_1.h5`, `RR_2.h5`, ..., `RR_n.h5`: Each dictionary for each RR interval is saved separately in an H5 file.



![image](https://github.com/user-attachments/assets/7bdcfb21-099f-4955-b16b-255424d6c9d4)


## 2_Train_Model
Trains a GRU model with an attention layer.
- **Inputs**:
  - `my_batch_size = 8`
  - `my_learning_rate = 0.001`
  - `Input_n = 16`
  - `Output_n = 750`
  - `cpu = 16`
  - `first_layer = 256`
  - `second_layer = 128`
  - `saveEpochs = 2`: Model is saved every 2 epochs.

**Note**: A smaller batch size, though resulting in better memory performance and potentially improving model accuracy, may reduce generalization capabilities. A batch size of 16 has also shown good results.

![image](https://github.com/user-attachments/assets/94bc3894-3700-4c3e-b931-b2968707982b)


## 3_Dictionary_Generator
Generates the dictionary by predicting it through the trained GRU model.
- **Inputs**:
  - `model = load_model("GRU_Full_01-28_LR_1E-03_BS_8_att_pp")`: Model used for prediction.
  - **RR Intervals and Ranges**:
    - `RR_Intervals`: Specified list of RR intervals.
    - `t1_ranges` and `t2_ranges`: Specified ranges for T1 and T2 values.
   
 
 
