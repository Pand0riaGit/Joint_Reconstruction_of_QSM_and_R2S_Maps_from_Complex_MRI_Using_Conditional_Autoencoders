# QSM & R2* Autoencoder Framework

This repository provides a complete pipeline for generating artificial brain MRI data, training complex-valued autoencoder models, and evaluating their performance in **Quantitative Susceptibility Mapping (QSM)** and **R2*** estimation.  
It combines synthetic data generation, ghost model processing, model training, and visualization utilities.

---

## 📂 Repository Structure

### 🔹 Data Generation
- **`dataset.py`**  
  Creates artificial data using real tissue values (white matter, gray matter, CSF, etc.).

- **`ghost.py`**  
  Generates **Magnitude**, **Phase**, and **R2*** images from an artificial QSM brain image located in the `Ghost Models` folder.  
  Maps voxel values to the closest possible tissue values and stores the generated files in `Ghost Models`.

- **`save_training_data.py`**  
  - Uses `dataset.py` to create artificial **Magnitude**, **Phase**, **QSM**, and **R2*** images.  
  - Runs a trained autoencoder model to reconstruct QSM and R2* from magnitude + phase input.  
  - Saves results plus **difference maps** in the `Created Artificial Data` folder.

### 🔹 Ghost Model Processing
- **`ghost_to_model.py`**  
  Takes generated ghost model images (magnitude + phase), runs them through the autoencoder, and produces **QSM** and **R2*** ghost reconstructions.  
  Also saves **difference maps** in the `Ghost Models` folder.

### 🔹 Models
- **`model.py`**  
  Complex-valued autoencoder (**No-Add version**): input and output are not added in the forward pass.  

- **`model_add.py`**  
  Complex-valued autoencoder (**Add version**): input and output are added at the end of the forward pass.  

- **`initializer.py`**  
  Utility to initialize complex weights for the models.

### 🔹 Training
- **`training.py`**  
  Trains the **No-Add** autoencoder. Saves model weights after each epoch and the checkpoint with the lowest validation loss.  

- **`training_add.py`**  
  Trains the **Add** autoencoder (can be run in parallel with `training.py` to compare both versions).  

- Both training scripts also store an array of loss values per epoch for later visualization.

### 🔹 Visualization
- **`loss_graph.py`**  
  Plots training losses over epochs:  
  - Total loss  
  - Magnitude loss  
  - Angle loss  

- **`subplots.py`**  
  Creates comparison subplots for artificial data reconstructions stored in `Created Artificial Data`.  
  Saves results in the `Subplots` folder.

- **`subplots_ghost.py`**  
  Creates comparison subplots for ghost model reconstructions (created, reference, and difference).  
  Saves results in the `Subplots` folder.

---

## 🔄 Workflow

The typical pipeline looks like this:

1. **Train a model**  
   python training.py        # Train No-Add autoencoder
   python training_add.py    # Train Add autoencoder
2. **Build ghost models**  
   python ghost.py
3. **Generate artificial data with the trained AE**  
   python save_training_data.py
4. **Apply AE to ghost models**  
   python ghost_to_model.py
5. **Visualize results**
   python subplots.py        # Artificial data comparison
   python subplots_ghost.py  # Ghost model comparison
   python loss_graph.py      # Training loss curves

---

## 📁 Output Folders
-Ghost Models – Contains generated magnitude, phase, QSM, R2*, and difference maps.

-Created Artificial Data – Contains artificial dataset reconstructions and their difference maps.

-Subplots – Stores subplot visualizations for both artificial and ghost data.

-Models – Stores model weights and loss arrays from training.



