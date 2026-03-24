# MammoTwin: An Open Source Digital Twin Framework for Protocol Optimization via MRI-to-Mammography Synthesis

**MammoTwin** is an open-source software framework designed to generate synthetic mammograms from Magnetic Resonance Imaging (MRI), validated through standard clinical Quality Assurance (QA) metrics. Exploring the promising pathway of digital twin simulation, it enables protocol optimization without biological risk, offering a tool to estimate patient-specific radiation dose and provide high-quality images at low radiation levels.

## 📖 Overview

The proposed system integrates a comprehensive five-phase pipeline:
1. **Acquisition and Normalization**: Processing diverse patient data with MR volume normalization.
2. **Hybrid Semantic Segmentation**: Utilizing state-of-the-art AI models (**nnU-Net** and **BreastSegNet**) to precisely segment internal structures and surfaces.
3. **Pseudo-Elastic Compression Model**: A biomechanical compression algorithm that strictly preserves biological tissue volume.
4. **Physics-Based X-Ray Simulation**: Fast deterministic ray-casting based on NIST XCOM attenuation cross-sections, including full detector sensitometry emulation (H&D curve equivalents).
5. **QA Validation**: Automated assessment of clinical quality metrics, such as Signal-to-Noise Ratio (SNR), Contrast-to-Noise Ratio (CNR), Dynamic Range, and Field Non-Uniformity.

## 👥 Authors

- **Daniel Lozano Gutiérrez** - Tecnológico de Monterrey
- **Juan Salvador Toledo Rios** - Tecnológico de Monterrey
- **Diana Sofía M. Rosales Gurmendi** - Tecnológico de Monterrey
- **José Gerardo Tamez Peña** - Tecnológico de Monterrey

## 📁 Project Structure

```text
SOFTW_SIMULATE_COMPRESSION/
│
├── src/                           # Main source code
│   ├── ui/                        # User interfaces (PySide6/Qt)
│   ├── physics/                   # Physics engine and simulation (NIST)
│   ├── utils/                     # 3D geometric transformations and helpers
│   └── config/                    # Global configurations
│
├── tests/                         # Tests and clinical validation scripts
├── docker/                        # Containerization setup
├── models/                        # AI models (nnU-Net and BreastSegNet weights)
├── patient_data/                  # Patient data directory (DICOM/NIfTI)
├── main.py                        # 🚀 Main entry point
└── requirements.txt               # Python dependencies
```

## 🚀 Installation and Usage

### Prerequisites

```bash
python -m venv medical_3d_stable
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
.\medical_3d_stable\Scripts\activate
pip install -r requirements.txt
```

### Run the Application

```bash
python main.py
```

## 🐳 Docker

```bash
cd docker
docker build -t breast-sim .
docker run -it --rm -p 6080:6080 -v "D:\patient_data:/patient_data" breast-sim
```

## 🧪 Tests

```bash
# MLO/CC transformation tests
python tests/test_mlo_fix.py

# Physical realism validation
python tests/verify_realism.py
```

## 🔬 Technologies

- **UI**: PySide6 (Qt)
- **3D**: PyVista, VTK, Trimesh
- **AI**: nnU-Net, PyTorch
- **Medical**: DICOM, NIfTI, NIST XCOM
- **Physics**: NumPy, OpenCV

## 📝 License

Released under the **MIT License**.
