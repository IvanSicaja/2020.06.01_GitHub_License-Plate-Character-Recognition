**🧾 🎯 Project Title: LICENSE PLATE CHARACTER RECOGNITION  
📅 Project Timeline:** June 2020 – September 2020  
🎥 YouTube Demo: Not available  
📦 GitHub Source Code: <https://github.com/IvanSicaja/2020.06.01_GitHub_License-Plate-Character-Recognition>  
\----------------------------------------------------------------------------------------------------------------

🏷️ My Personal Profiles: ⬇︎  
🎥 Video Portfolio: To be added  
📦 GitHub Profile: <https://github.com/IvanSicaja>  
🔗 LinkedIn: <https://www.linkedin.com/in/ivan-si%C4%8Daja-832682222>  
🎥 YouTube: <https://www.youtube.com/@ivan_sicaja>  
\----------------------------------------------------------------------------------------------------------------

### 📚🔍 Project description: ⬇︎⬇︎⬇︎

### 💡 App Purpose

Efficient and accurate **extraction of characters from car license plates** for applications in e.g. **traffic monitoring**, **vehicle identification**, and **smart parking systems**.

### 🧠 How It Works

In this project, a **Convolutional Neural Network (CNN)** is trained on **400,000+ image examples**.  
Images are converted into **.CSV files** for **faster processing**.  
The input image is filtered with different filters (**Grayscale, Gaussian Blur, Threshold, Binary, Dilatation**) in order to:

- **Speed up** image processing (replace three color channels with one channel, RGB → Grayscale)
- **Reduce noise** (dust, reflections on license plates)
- **Enhance edges** for sharper and more accurate **character recognition**

The trained model achieved **77.54% accuracy**.

PROJECT WORKFLOW:

• **Data Collection & Preprocessing**  
 ◦ **Combined** and adjusted a few **publicly available datasets** into one **.CSV dataset.**  
 ◦ Each **image** stored as **28×28 normalized values (0–1)** with corresponding **labels encoded**.

• **Noise Reduction & Filtering**  
 ◦ Applied **Grayscale**, **Gaussian Blur**, **Thresholding**, **Binary conversion**, and **Dilatation**.  
 ◦ Improved **edge clarity** for robust **character segmentation**.

• **Data Augmentation**  
 ◦ Introduced **rotation**, **scaling**, and **shifting**.  
 ◦ Expanded **dataset variety** to improve **generalization** and reduce **overfitting**.

• **Model Architecture**  
 ◦ Layers: **Conv2D → MaxPooling2D → Dropout → Flatten → Dense**.  
 ◦ **Dropout layers** added for **regularization**.

• **Training Strategy**  
 ◦ **Optimizer**: **Adam** with **adaptive learning rate**.  
 ◦ Trained over **60 epochs**, **batch size 512**.  
 ◦ **Early stopping** and **checkpointing** used to avoid **overtraining**.

• **Evaluation & Validation**  
 ◦ **Accuracy** monitored on a **validation set** after each epoch.  
 ◦ Final performance tested using a **confusion matrix** to analyze **class-level errors**.  
 ◦ Ensured balance between **accuracy**, **robustness**, and **scalability**.

### ⚠️ Note

None.

### 🔧 Tech Stack

**Python, OpenCV, TensorFlow, Keras, Pandas, scikit-learn, Seaborn, Matplotlib, Git, Linux, Visual Studio Code**

### 📸 Project Snapshot

Not available.

### 🎥 Video Demonstration

Not available.

---

### 📸 Project Snapshot

<p align="center">
  <img src="https://github.com/IvanSicaja/2020.06.01_GitHub_License-plate-character-recognition/raw/main/0.1_GitHub/1.0_Description_4_media_key_messages_and_captions/2.0_Thumbnail_1.png" 
       alt="App Preview" 
       width="640" 
       height="360">
</p>

---

### 🎥 Video Demonstration

Not available.

---


### 📣 Hashtags Section

**\# #LicensePlateRecognition #CNN #ComputerVision #Python #OpenCV #TensorFlow #Keras #MachineLearning #DataScience #DeepLearning #ImageProcessing #AI #Git #Linux**
