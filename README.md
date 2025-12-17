## 📋 Student Information

| Field             | Details                                             |
| ----------------- | --------------------------------------------------- |
| **Name**          | Rakshith M P                                         |
| **USN**           | 1CD22AI045                                        |
| **Semester**      | 7th                                                 |
| **Department**    | Artificial Intelligence and Machine Learning (AIML) |
| **Subject**       | Deep Learning and Reinforcement Learning            |
| **Course Code**   | BAI701                                              |
| **Academic Year** | 2025–2026                                           |

----------------------------------------------------

## 📑 Table of Contents

1. [cat_and_dog.py](#1-cat_and_dogpy)
2. [AlexNet](#2-alexnet)
3. [DeepReinforcementLearning](#3-deepreinforcementlearning)
4. [LSTM](#4-lstm)
5. [RNN](#5-rnn)
6. [Tic Tac Toe](#6-tic-tac-toe)



----------------------------------------------------

## **1. cat_and_dog.py**

### **Explanation of Changes and Improvements**

---

### **1️⃣ Dataset Handling**

**Original Code**

* Dataset was manually downloaded and extracted using a local ZIP file
* Required fixed file paths like `E:\DL\cats_and_dogs_filtered`
* Code fails if the file path is wrong, dataset is missing, or run on another system

**Modified Code**

* Used **TensorFlow Datasets (`tfds`)**

  ```python
  tfds.load('cats_vs_dogs', ...)
  ```
* Dataset is automatically downloaded and prepared

**Benefit**

* Code is portable, error-free, and system independent

---

### **2️⃣ Data Preprocessing Pipeline**

**Original Code**

* Used `ImageDataGenerator` with only `rescale = 1./255`
* Image loading through directory structure

**Modified Code**

* Used **`tf.data` pipeline**

  ```python
  ds_train.map().batch().prefetch()
  ```
* Images resized, normalized, and efficiently batched

**Benefit**

* Faster training and better memory utilization

---

### **3️⃣ Model Architecture Improvements**

**Original Code**

* Simple CNN without normalization or regularization

```
Conv → Pool → Conv → Pool → Conv → Pool → Dense
```

**Modified Code**

* Added Batch Normalization and Dropout
* Increased number of convolution filters

```
Conv → BatchNorm → ReLU → Pool → Dropout
```

**Benefit**

* Stable training and reduced overfitting

---

### **4️⃣ Optimizer Change**

**Original Code**

* RMSprop optimizer

**Modified Code**

* Adam optimizer with lower learning rate

**Benefit**

* Faster and more stable convergence

---

### **5️⃣ Training Strategy**

**Original Code**

* Fixed number of epochs

**Modified Code**

* Added EarlyStopping

**Benefit**

* Prevents overfitting and saves best model

---

### **6️⃣ Visualization and Output**

**Original Code**

* Complex visualizations and forced program termination

**Modified Code**

* Clean accuracy and loss plots

**Benefit**

* Clear, professional, and safe output

---

### **7️⃣ Code Simplicity and Maintainability**

**Original Code**

* Long and hard to debug

**Modified Code**

* Shorter, modular, and industry standard

**Benefit**

* Easy to explain and extend

---

### **Performance Comparison (Typical)**

| Metric              | Original | Modified |
| ------------------- | -------- | -------- |
| Training Accuracy   | ~85%     | ~90%     |
| Validation Accuracy | ~80–85%  | ~85–90%  |
| Overfitting         | Higher   | Reduced  |
| Runtime Errors      | Possible | Very low |

---

----------------------------------------------------


## **2. AlexNet**

### **Explanation of Changes and Improvements**

---

### **1️⃣ Batch Normalization Added**

**Original Code**

* No Batch Normalization layers
* Training may be unstable and slower

**Modified Code**

* Added `BatchNormalization()` after each convolution layer

**Benefit**

* Faster convergence
* Stable training
* Improved generalization

---

### **2️⃣ Fully Connected Layer Size Reduced**

**Original Code**

```python
Dense(4096)
Dense(4096)
```

**Modified Code**

```python
Dense(1024)
Dense(1024)
```

**Benefit**

* Reduces number of parameters
* Prevents overfitting
* Faster training and lower memory usage

---

### **3️⃣ Improved Activation Placement**

**Original Code**

* Activation applied directly inside `Conv2D`

**Modified Code**

* Activation applied **after Batch Normalization**

**Benefit**

* Follows modern CNN best practices
* More stable gradient flow

---

### **4️⃣ Optimizer Improvement**

**Original Code**

* Optimizer not explicitly defined

**Modified Code**

* Used **Adam optimizer** with low learning rate

**Benefit**

* Adaptive learning rate
* Faster and smoother convergence

---

### **5️⃣ Regularization Enhancement**

**Original Code**

* Dropout used only for dense layers

**Modified Code**

* Dropout retained with better-balanced architecture

**Benefit**

* Reduced overfitting
* Better validation accuracy

---

### **6️⃣ Model Complexity Optimization**

**Original Code**

* Very high parameter count
* Difficult to train on limited hardware

**Modified Code**

* Reduced parameters while maintaining depth

**Benefit**

* Easier to train
* Suitable for modern systems
* Easy to explain in viva

---

### **Performance Comparison (Typical)**

| Metric             | Original AlexNet | Modified AlexNet |
| ------------------ | ---------------- | ---------------- |
| Parameters         | Very High        | Reduced          |
| Training Stability | Medium           | High             |
| Overfitting        | High             | Reduced          |
| Training Speed     | Slow             | Faster           |

---
----------------------------------------------------


## **3. DeepReinforcementLearning**

### **Explanation of Changes and Improvements**

---

### **1️⃣ Code Structure and Readability**

**Original Code**

* Very long and repetitive
* Same functions (`available_actions`, `update`) defined multiple times
* Difficult to follow execution flow

**Modified Code**

* Removed duplicate functions
* Organized code into clear sections (Environment, Training, Testing)

**Benefit**

* Cleaner structure
* Easier to read, debug, and explain

---

### **2️⃣ Action Selection Strategy**

**Original Code**

* Actions selected randomly from available actions
* No balance between exploration and exploitation

**Modified Code**

* Implemented **epsilon-greedy policy**
* Agent sometimes explores and sometimes exploits best-known action

**Benefit**

* Prevents getting stuck in suboptimal paths
* More realistic reinforcement learning behavior

---

### **3️⃣ Reward and Q-Matrix Handling**

**Original Code**

* Reward matrix and Q-matrix mixed with extra logic
* Reward normalization done repeatedly

**Modified Code**

* Clearly separated **Reward (R)** and **Q-value (Q)** matrices
* Used standard Q-learning update rule

**Benefit**

* Stable learning
* Matches standard reinforcement learning theory

---

### **4️⃣ Learning Process Simplification**

**Original Code**

* Included police and drug trace environment matrices
* Added unnecessary complexity for basic learning task

**Modified Code**

* Removed police/drug environment logic
* Focused on core Q-learning path optimization

**Benefit**

* Easier to understand
* Suitable for academic and exam purposes

---

### **5️⃣ Training Strategy**

**Original Code**

* Random initial states
* Learning progress difficult to interpret

**Modified Code**

* Fixed number of episodes
* Stored cumulative rewards per episode

**Benefit**

* Clear learning curve
* Easy to visualize convergence

---

### **6️⃣ Optimal Path Extraction**

**Original Code**

* Multiple condition checks and randomness during testing

**Modified Code**

* Used greedy policy after training to extract optimal path

**Benefit**

* Consistent and correct shortest path to goal

---

### **7️⃣ Visualization Improvements**

**Original Code**

* Multiple plots with unclear meaning

**Modified Code**

* Single, clear plot showing learning progress (Q-values vs episodes)

**Benefit**

* Better visualization
* Professional and meaningful output

---

### **📊 Performance Comparison (Typical)**

| Aspect             | Original  | Modified       |
| ------------------ | --------- | -------------- |
| Code Length        | Very Long | Short & Clean  |
| Readability        | Low       | High           |
| Action Strategy    | Random    | Epsilon-Greedy |
| Learning Stability | Medium    | High           |
| Exam/Viva Friendly | ❌         | ✅              |

---

----------------------------------------------------

## **4. LSTM**

### **Explanation of Changes and Improvements**

---

### **1️⃣ Dataset Handling**

**Original Code**

* Dataset loaded using a **fixed local path**

  ```
  D:\Antony2\Acad\Lec\DLRL\Datasets\international-airline-passengers.csv
  ```
* Code fails if the file is missing or run on another system.

**Modified Code**

* Dataset is **automatically downloaded from GitHub** using a URL.
* No dependency on local file paths.

**Benefit**

* Code becomes **portable and reusable**.
* Runs on any system or cloud environment without errors.

---

### **2️⃣ Data Preparation & Sequence Creation**

**Original Code**

* Sequence creation logic written **twice** (for train and test).
* Time step fixed at `10`.

**Modified Code**

* Introduced a **single reusable function** `create_sequences()`.
* Time steps increased to `12` (monthly seasonality).

**Benefit**

* Cleaner and modular code.
* Better learning of yearly patterns in time-series data.

---

### **3️⃣ Model Architecture**

**Original Code**

* Single LSTM layer with **10 units**.
* Shallow network.

```
LSTM → Dense
```

**Modified Code**

* Used **Bidirectional LSTM**.
* Added **multiple stacked LSTM layers**.
* Included Dropout layers.

```
BiLSTM → Dropout → BiLSTM → Dropout → Dense
```

**Benefit**

* Learns patterns from both past and future context.
* Improved prediction accuracy.
* Reduced overfitting.

---

### **4️⃣ Regularization Techniques**

**Original Code**

* No explicit regularization.

**Modified Code**

* Added **Dropout layers**.
* Used **EarlyStopping** callback.

**Benefit**

* Prevents overfitting.
* Stops training when validation loss stops improving.

---

### **5️⃣ Training Strategy**

**Original Code**

* Batch size = 1.
* Fixed epochs (50).
* No validation monitoring.

**Modified Code**

* Increased batch size (8).
* Added validation split.
* Early stopping used.

**Benefit**

* Faster and more stable training.
* Better generalization.

---

### **6️⃣ Evaluation Metrics**

**Original Code**

* Used only **RMSE**.

**Modified Code**

* Used **RMSE + MAE**.

**Benefit**

* More comprehensive evaluation of model performance.
* Better error interpretation.

---

### **7️⃣ Visualization Improvements**

**Original Code**

* Basic plots without structure.
* No separation of training and testing predictions.

**Modified Code**

* Clear plots with:

  * Actual data
  * Training predictions
  * Testing predictions
* Proper labels and titles.

**Benefit**

* Professional and easy-to-understand output.
* Suitable for reports and presentations.

---

### **8️⃣ Model Saving**

**Original Code**

* Model not saved explicitly.

**Modified Code**

* Model saved as `.h5` file.

**Benefit**

* Model can be reused or deployed later.

---

### **📊 Performance Comparison (Typical)**

| Aspect           | Original LSTM | Modified LSTM |
| ---------------- | ------------- | ------------- |
| Model Depth      | Shallow       | Deep (BiLSTM) |
| Context Learning | One-direction | Bidirectional |
| Overfitting      | Higher        | Reduced       |
| Metrics Used     | RMSE          | RMSE + MAE    |
| Portability      | Low           | High          |

---

----------------------------------------------------

## **5. RNN**

### **Explanation of Changes and Improvements**

---

### **1️⃣ Input Sequence Handling**

**Original Code**

* Used `seq_length = 5`
* Very short context window for character prediction

**Modified Code**

* Increased sequence length to `6`

**Benefit**

* Better context understanding
* Improves next-character prediction accuracy

---

### **2️⃣ Model Architecture**

**Original Code**

* Single `SimpleRNN` layer
* No regularization

```
SimpleRNN → Dense
```

**Modified Code**

* Added **Dropout layer**
* Improved hidden units

```
SimpleRNN → Dropout → Dense
```

**Benefit**

* Reduces overfitting
* Improves model generalization

---

### **3️⃣ Activation Function**

**Original Code**

* Used **ReLU** activation in RNN layer

**Modified Code**

* Used **tanh** activation (default for RNNs)

**Benefit**

* Prevents exploding gradients
* More stable recurrent learning

---

### **4️⃣ Optimizer Configuration**

**Original Code**

* Used default Adam settings

**Modified Code**

* Explicit Adam optimizer with controlled learning rate

**Benefit**

* Stable convergence
* Better training control

---

### **5️⃣ Training Strategy**

**Original Code**

* Trained for 100 epochs
* No regularization during training

**Modified Code**

* Reduced epochs to 60
* Improved architecture compensates for fewer epochs

**Benefit**

* Faster training
* Similar or better performance

---

### **6️⃣ Text Generation Logic**

**Original Code**

* Mixed commented input sequences
* Less structured generation loop

**Modified Code**

* Clean and consistent text generation pipeline

**Benefit**

* Easier to demonstrate
* Cleaner output

---

### **7️⃣ Code Readability & Maintainability**

**Original Code**

* Minimal comments
* Harder to extend

**Modified Code**

* Cleaner structure
* Clearly separated sections

**Benefit**

* Easy to explain in viva
* Easy to modify

---

### **📊 Performance Comparison (Typical)**

| Aspect         | Original RNN | Modified RNN |
| -------------- | ------------ | ------------ |
| Context Length | Short        | Improved     |
| Overfitting    | Higher       | Reduced      |
| Stability      | Medium       | High         |
| Readability    | Medium       | High         |

---

====================================================


## **6. Tic Tac Toe**

---

## 🔹 Original Code (Before)

* Used **Reinforcement Learning (Q-learning)**
* Required **training phase (50,000 games)** before play
* Stored policies using **pickle files**
* Board positions represented as **(row, col)** tuples
* Human had to input **row and column separately**
* Execution was **slow** due to:

  * Training loop
  * State-value updates
* Hard to debug and explain in viva

---

## 🔹 Modified Code (After)

* **Removed training completely**
* Implemented **rule-based AI logic**:

  * Win if possible
  * Block opponent if needed
  * Otherwise choose random move
* Board positions represented as **1–9 numbering**
* Human gives **single number input**
* No files, no policies, no reinforcement learning
* Instant execution (no delay)

---

## 1️⃣ Gameplay Interaction

### Original

* Input format:

  ```text
  row = 0, col = 2
  ```
* Not user-friendly

### Modified

* Input format:

  ```text
  1 2 3
  4 5 6
  7 8 9
  ```
* Simple and intuitive

✅ **Benefit:** Easy for users and exam demo

---

## 2️⃣ AI Logic

### Original

* Learns by:

  * Exploring states
  * Updating rewards
  * Storing Q-values

### Modified

* Uses **deterministic logic**:

  * Try to win
  * Block opponent
  * Random fallback

✅ **Benefit:**

* Fast
* Predictable
* No training required

---

## 3️⃣ Performance

| Feature         | Original  | Modified |
| --------------- | --------- | -------- |
| Training time   | Very high | None     |
| Execution speed | Slow      | Instant  |
| Files required  | Yes       | No       |
| Runtime errors  | Possible  | Very low |

---

## 4️⃣ Code Complexity

### Original

* Long code
* Multiple classes
* Reinforcement learning concepts
* Difficult to explain

### Modified

* Short and clean
* No ML dependency
* Easy to understand logic

✅ **Benefit:**
Perfect for **assignments, lab exams, and viva**

---

## 5️⃣ Replay Feature

### Original

* Replay handled with manual loops

### Modified

```python
again = input("Play again? (y/n): ")
```

✅ **Benefit:**
User-controlled replay, clean exit

---

## ✅ Final Outcome

* Same **Tic Tac Toe functionality**
* Much **faster**
* **Human vs AI playable**
* Uses **1–9 input**
* Ideal for **academic submission**

---
----------------------------------------------------
