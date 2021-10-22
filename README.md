# Representation Learning of breaking waves

Repository for code to reproduce the results of the pre-print Representation Learning of breaking waves.

---

## Contents

[Representation Learning of breaking waves](#representation-learning-of-breaking-waves)

- [1. Environment](#1-environment)
- [2. Data](#2-data)
    - [2.1. Sources](#21-sources)
    - [2.2. Folder Structure](#22-folder-structure)
- [3. Models](#3-models)
- [4. Training](#4-training)
- [5. Evaluation](#5-evaluation)
    - [5.1. Evaluating](#51-evaluating)
    - [5.2. Results](#52-results)

---

## 1. Environment

Creating a virtual environment to install the dependencies packages is recommended, this can be done with either conda or the python3 venv.

### Using [`conda`](https://docs.conda.io/en/latest/)

```bash
conda env create -f environment.yml
conda activate rlwaves
```

#### exporting dependencies to yml file

```bash
conda env export > environment.yml
```

### Using [`pip`](https://pypi.org/project/pip/)

```bash
python3 -m venv rlwaves
source rlwaves/bin/activate
pip install -r requirements.txt
```

#### exporting dependencies to a txt file

```bash
pip freeze >> requirements.txt
```

<details>
    <summary>Dependencies list</summary>

    - h5py
    - keras
    - keras
    - matplotlib
    - numpy
    - pickleshare
    - pillow
    - scikit-learn
    - scipy
    - seaborn
    - tensorflow
    - albumentations

</details>

---

## 2. Data

### 2.1. Sources

### 2.2. Folder Structure

---

## 3. Models

---

## 4. Training

---

## 5. Evaluation

### 5.1. Evaluating

### 5.2. Results

---

## [License](LICENSE)

<details>
<summary>MIT License</summary>

Copyright (c) 2021 Ryan Smith

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
</details>
