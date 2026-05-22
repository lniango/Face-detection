# Face Detection — Method Comparison

A computer vision project exploring and comparing different face detection techniques using Python and OpenCV. Each method is implemented as a standalone module and tested on real-time webcam input.

## Goal

Implement multiple face detection approaches, then benchmark and compare their performance in terms of accuracy, speed, and robustness.

---

## Methods Implemented

### 1. Haar Cascade (`face_detection_haar.py`)
A classical detection approach based on **Haar-like features** and a boosted classifier trained by Viola & Jones (2001).

- Uses OpenCV's pre-trained `haarcascade_frontalface_default.xml`
- Operates on **grayscale** images
- Fast and lightweight, suitable for real-time use
- Sensitive to lighting conditions and face orientation

### 2. HOG + SVM (`face_detection_hog.py`)
Detection based on **Histogram of Oriented Gradients** combined with a linear SVM classifier, via the `dlib` library.

- Uses `dlib.get_frontal_face_detector()` (pre-trained HOG + SVM model)
- Operates on **RGB** images
- More robust than Haar to lighting variations
- Slightly slower but generally more accurate on frontal faces

---

## Requirements

```bash
pip install opencv-python dlib imutils scikit-image numpy matplotlib
```

> You will also need `haarcascade_frontalface_default.xml` (bundled with OpenCV) and a `helpers.py` file providing the `convert_and_trim_bb` utility for the HOG module.

---

## Usage

Run either script directly to start detection on your webcam (device index `1` by default — change to `0` if needed):

```bash
python face_detection_haar.py
python face_detection_hog.py
```

Press `q` to quit.

---

## Planned Methods

- CNN-based detection (dlib `cnn_face_detection_model_v1`)
- Deep learning with OpenCV DNN module
- MTCNN / RetinaFace

