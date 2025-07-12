# 亚马逊高温事件识别与温度预测项目

## 🌍 项目概述

本项目旨在使用人工神经网络模型，对亚马逊热带雨林的月度高温事件进行识别（分类任务）以及温度进行预测（回归任务）。该区域受四大气候模态（ENSO、NAO、TSA、TNA）的影响显著，近年来极端高温事件频发，对生态系统和全球气候稳定性造成潜在威胁。

本项目基于 TensorFlow/Keras 框架，完成以下两个主要任务：

- **任务 A – 分类任务：** 预测某月是否发生高温事件（Hot）。
- **任务 B – 回归任务：** 预测该月的实际温度值。

---

## 📁 数据说明

项目提供了两份主要数据集：

- `Amazon_temperature_student.csv`：包含1982年至2022年亚马逊地区的月度温度数据及4种气候模态指数（ENSO、NAO、TSA、TNA）。
- `threshold.csv`：提供每个月份的高温判定阈值（threshold）。

气候模态指数定义如下：

| 指数 | 含义                                 | 范围     |
|------|--------------------------------------|----------|
| ENSO | 厄尔尼诺/拉尼娜 (Nino 3.4区海温异常) | -3 到 3  |
| NAO  | 亚速尔群岛与冰岛间的气压差（归一化） | -4 到 4  |
| TSA  | 热带南大西洋海温异常                  | -1 到 1  |
| TNA  | 热带北大西洋海温异常                  | -1 到 1  |

---

## 🧠 任务 A – 高温事件分类

### 🔧 数据预处理

- 构造二值标签 `Hot`：若该月温度超过对应月份阈值，则记为1，否则为0。
- 数据集按比例随机划分为训练集、验证集和测试集。
- 使用 `StandardScaler` 对特征进行标准化。
- 对月份使用**周期性编码**，以反映时间连续性（即 12 月与 1 月应相近）。

### 🏗️ 模型结构

- 多层全连接神经网络，使用 ReLU 激活函数。
- 输出层：1 个神经元，Sigmoid 激活函数。
- 损失函数：Binary Crossentropy（二元交叉熵）。
- 优化器：Adam。
- 超参数配置：
  - 批大小：32
  - 训练轮数：50
  - 学习率：0.001

### 📊 模型评估

- 测试集**平衡准确率（Balanced Accuracy）**：**87%**
- 敏感度（Sensitivity / True Positive Rate）：**84%**
- 特异度（Specificity / True Negative Rate）：由混淆矩阵计算获得
- 可视化混淆矩阵
- 绘制训练集与验证集的准确率变化曲线

---

## 🌡️ 任务 B – 月度温度回归预测

### 🔧 数据预处理

- 输入特征使用 `StandardScaler` 进行标准化。
- **随机划分模式下**：目标值（temperature）不进行缩放。
- **逐年划分模式下**：
  - 将完整年份划分为训练/验证/测试，确保每年只出现在一个子集中。
  - 对目标值（temperature）使用 `MinMaxScaler` 进行缩放，仅在训练集拟合并用于验证集和测试集。
  - 特征缩放策略与随机划分保持一致。

### 🏗️ 模型结构

- 全连接神经网络用于回归预测。
- 输出层：1 个神经元，线性激活函数。
- 损失函数：均方误差（MSE）。
- 优化器：Adam。
- 超参数配置：
  - 批大小：32
  - 训练轮数：100
  - 学习率：0.001

### 📊 模型评估

- **随机划分结果**：
  - 平均绝对误差（MAE）：**小于 0.27°C**
  - Pearson 相关系数：**0.93**
  - 绘制真实值与预测值的散点图

- **逐年划分结果**：
  - 使用相同网络结构与超参重新训练
  - 预测结果经 MinMaxScaler 反标准化处理
  - 使用 MAE 与 Pearson r 进行评估

---

## 🔁 模型部署与复现性保障

- 固定随机种子以确保实验可复现。
- 保存所有模型与预处理器：
  - Hot 分类器 + 特征缩放器
  - 随机划分回归模型 + 特征缩放器
  - 逐年划分回归模型 + 特征缩放器 + 目标缩放器
- Notebook 包含：
  - 一键加载隐藏测试集及所有模型
  - 自动评估并输出混淆矩阵、回归散点图、各项评估指标

---

## 📌 附加说明

- 月份使用以下公式进行周期性编码：  
  `month_norm = 2π × (month - 1) / 12`  
  以确保 12 月与 1 月在特征空间中距离接近。

- 模型总参数量控制在样本数的 10% 以下，避免过拟合。

---




# Amazon Hot Event Detection and Temperature Forecasting

## 🌍 Project Overview

This project focuses on predicting monthly temperature and identifying high-temperature events (hot events) in the Amazon rainforest using artificial neural networks. The region of study—highlighted in red on the map provided in the assignment—is particularly vulnerable to climate variability, especially due to the influence of four major oceanic climate modes: ENSO, NAO, TSA, and TNA.

The project addresses two main tasks using TensorFlow/Keras:

- **Task A – Classification:** Predict whether a hot event occurs in a given month.
- **Task B – Regression:** Predict the actual temperature for a given month.

---

## 📁 Data Description

Two main datasets are provided:

- `Amazon_temperature_student.csv`: Contains monthly temperature records (1982–2022) and climate mode indices (ENSO, NAO, TSA, TNA).
- `threshold.csv`: Provides the hot event temperature threshold for each calendar month.

The climate mode indices have the following interpretations:

| Index | Meaning                                    | Range   |
|-------|---------------------------------------------|---------|
| ENSO  | El Niño/La Niña (Nino 3.4 SST anomaly)      | -3 to 3 |
| NAO   | Pressure differential (Azores - Iceland)    | -4 to 4 |
| TSA   | SST anomaly (Tropical South Atlantic)       | -1 to 1 |
| TNA   | SST anomaly (Tropical North Atlantic)       | -1 to 1 |

---

## 🧠 Task A – Hot Event Classification

### 🔧 Preprocessing

- A binary `Hot` label is defined: 1 if temperature exceeds the monthly threshold; 0 otherwise.
- Dataset is randomly split into training/validation/test sets.
- Features are normalized using `StandardScaler`.
- `month` is cyclically encoded to preserve temporal continuity.

### 🏗️ Model Architecture

- Fully-connected neural network with ReLU activations.
- Output layer: 1 unit with sigmoid activation.
- Loss: Binary Crossentropy.
- Optimizer: Adam.
- Hyperparameters:
  - Batch size: 32
  - Epochs: 50
  - Learning rate: 0.001

### 📊 Evaluation

- Balanced Accuracy on test set: **87%**
- Sensitivity (TPR): **84%**
- Specificity (TNR): Calculated from confusion matrix
- Confusion matrix plotted for visualization
- Training and validation accuracy plotted across epochs

---

## 🌡️ Task B – Temperature Regression

### 🔧 Preprocessing

- Inputs normalized using `StandardScaler`.
- Target (`temperature`) not transformed in random split.
- For year-wise split:
  - Years partitioned such that each appears in only one of training/validation/test.
  - Target (`temperature`) scaled using `MinMaxScaler`, fitted only on the training set.

### 🏗️ Model Architecture

- Fully-connected neural network for regression.
- Output: 1 unit (linear activation).
- Loss: Mean Squared Error.
- Optimizer: Adam.
- Hyperparameters:
  - Batch size: 32
  - Epochs: 100
  - Learning rate: 0.001

### 📊 Evaluation

- **Random Split:**
  - MAE: **< 0.27°C**
  - Pearson r: **0.93**
  - True vs Predicted scatter plot visualized

- **Year-wise Split:**
  - Same architecture re-used
  - Inverse transform applied to scaled predictions
  - Evaluation via MAE and Pearson r

---

## 🔁 Deployment & Reproducibility

- Random seed fixed to ensure reproducibility
- All scalers and models saved:
  - Classifier + feature scaler
  - Random-split regressor + feature scaler
  - Year-wise regressor + feature scaler + target scaler
- Notebook includes:
  - Single cell to load hidden test dataset
  - Restore all saved models and scalers
  - Run evaluations & generate required plots and metrics

---

## 📌 Notes

- `month` was cyclically encoded as:  
  `month_norm = 2π × (month - 1) / 12`  
  This ensures that December and January are close in feature space.

- Model complexity was carefully kept under 10% of training sample size.

---
