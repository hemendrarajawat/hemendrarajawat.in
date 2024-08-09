---
title: First End-to-End Machine Learning Project
description: This blog will work through an example project end to end to illustrate the main steps of a machine learning project.
date: 2024-08-08T20:06:06.902Z
draft: true
tags:
    - machine learning
categories:
    - Machine Learning
lastmod: 2024-08-09T11:18:45.788Z
cover:
    image: /blog/first-end-to-end-machine-learning-project/first_end_to_end_machine_learning_project.jpg
slug: first-end-to-end-machine-learning-project
---

## Introduction
In this blog, you will work through an example project end to end, pretending to be a recently hired data scientist at a real estate company. This example is fictitious; the goal is to illustrate the main steps of a machine learning project, not to learn anything about the real estate business. Here are the main steps we will walk through
1. Look at the big picture
2. Get the data
3. Explore and visualize the data to gain insights
4. Prepare the data for machine learning algorithms
5. Select a model and train it
6. Fine-tune your model
7. Present your solutions

We'll use the [California Housing Prices](https://github.com/ageron/handson-ml/tree/master/datasets/housing) dataset. This dataset is a modified version of the California Housing dataset available from Luís Torgo's page (University of Porto). Luís Torgo obtained it from the StatLib repository (which is closed now).

> The original dataset appeared in a 1997 paper titled "Sparse Spatial Autoregressions" by Pace, R. Kelley and Ronald Barry, published in the Statistics and Probability Letters journal. They built it using the 1990 California census data.

## Look at the Big Picture
Your task is to use the data to build a model of housing prices in the state. This data includes metrics such as the population, median income, and median housing price for each block group in Califonia. Your model should learn from this data and be able to predict the median housing price in any district, given all the other metrics.

### Frame the Problem
The first to ask is what exactly the business objective is. Building a model is probably not the end goal. Knowing the objective is important because it will determine how to from the problem, which algorithms to select, which performance measure to use to evaluate your model, and how much effort to spend on tweaking it.

The next question to ask is what the current solution looks like (if any). The current solution will often give you a reference for performance, as well as insights on how to solve the problem. 


It have been provided to you that, your model's output (a prediction of a district's median housing price) will be fed to another machien learning system, along with many other signals. This downstream system will determine whether it is worth incesting in a given area (Figure 1). Currently, the district housing prices are estimated manually by experts, or they estimate it using compex rules when expets cannot estimate.

![A machine learning pipeline for real estate investment](/blog/first-end-to-end-machine-learning-project/machine_learning_pipeline.png)
<figcaption align="center"> Figure 1: A machine learning pipeline for real estate investment</figcaption>

<br />

***
**PIPELINES**

A sequence of data processing components is called a data pipeline. Pipelines are very common in machine learning systems, since there is a lot of data to manipulate and many data transformations to apply. 

Components typically run asynchronously. Each components pulls a large amount of data, process it, and splits out the result in another data source. Then, some time later, the next component in the pipeline pulls in this data and spits out its own output.
***

With all this information, you are now ready to start designing your system. This is clearly a typical supervised learning, multiple regression, batch learning task. 

### Select a Performance Measure
Your next step is to select a performance measure. A typical performance measure for regression problems is the *root mean square error* (RMSE). It gives an idea of how much error the system typically makes in its predictions, with a higher weight given to large errors. Below equation how the mathematical formula to calcuate the RMSE:

$$
\begin{multline} \\
\small \text{RMSE(X, h)} = \sqrt{ \dfrac{1}{m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)})^2 } \\
\shoveleft{\small \text{where:}} \\
\end{multline}
$$

$$
\begin{split}
m &= \scriptsize \text{the number of instances in the dataset you are measuring the RMSE on.} \\
x^{(i)} &= \scriptsize \text{a vector of all the feature values of the } i^{th} \scriptsize \text{ instance in the dataet, and } y^{(i)} \scriptsize \text{ is its label.} \\
X &= \scriptsize \text{a matrix containing all the feature values of all the instances in the dataset.} \\
h &= \scriptsize \text{your system's prediction function, also called a hypothesis.} \\
\scriptsize \text{RMSE(X, h)} &= \scriptsize \text{the cost function measured on the set of examples using your hypothesis } h.
\end{split}
$$

Although the RMSE is generally the preferred performance measure for regression tasks, in some contexts you may prefer to use another function. For eg., if there are many outliers districts. In that case, you may consider using the *mean absolute error* (MAE):

$$\small \text{MAE(X, h)} = \dfrac{1}{m} \sum_{i=1}^{m} \big | h(x^{(i)}) - y^{(i)} \big | \\$$


## Get the Data

### Download the Data
In typical environments your data would be available in a relational database or some other common data store, and spread across multiple tables/documents/files. However, things are much simpler: you will just download a single compressed file, housing.tgz, which contains a comma-seperated values (CSV) file called housing.csv with all the data.

*Rather than manually downloading and decompressing the data, it's usually preferrable to write a function that does it for you.*

```python
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
        return pd.read_csv(Path("datasets/housing/housing.csv"))
    
housing = load_housing_data()
```

### Take a Quick Look at the Data Structure
```python
housing.head()
```
![Top five rows in the dataset](/blog/first-end-to-end-machine-learning-project/pd_data_head.png)

Each data represent one district and there are 10 attributes. The `info()` method is useful to get a quick description of the data, in particular the total number of rows, each attribute's type, and the number of non-null values.

```python
housing.info()
```
```output
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20640 entries, 0 to 20639
Data columns (total 10 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   longitude           20640 non-null  float64
 1   latitude            20640 non-null  float64
 2   housing_median_age  20640 non-null  float64
 3   total_rooms         20640 non-null  float64
 4   total_bedrooms      20433 non-null  float64
 5   population          20640 non-null  float64
 6   households          20640 non-null  float64
 7   median_income       20640 non-null  float64
 8   median_house_value  20640 non-null  float64
 9   ocean_proximity     20640 non-null  object 
dtypes: float64(9), object(1)
memory usage: 1.6+ MB
```

There are 20,640 instances in the dataset, which means that it is farily small by machine learning standards, but it's perfert to get started. The `total_bedrooms` attribute has only 20,433 non-null values, meaning that 207 districts are missing this feature. You will need to take care of this later.

All the attributes are numerical, except for `ocean_proximity`. You can find out what categories exist and how many districts belong to each category by using the `value_counts()` method:

```python
housing["ocean_proximity"].value_counts()
```
```output
<1H OCEAN     9136
INLAND        6551
NEAR OCEAN    2658
NEAR BAY      2290
ISLAND           5
Name: ocean_proximity, dtype: int64
```

Let's look at the other fields. The `describe()` method shows a summary of the numerical attributes.

```python
housing.describe()
```
![Summary of each numerical attribute](/blog/first-end-to-end-machine-learning-project/pd_data_describe.png)

Another quick way to get a feel of the type of data you are dealing with is to plot a histogram for each numerical attribute.

```python
%matplotlib inline
import matplotlib.pyplot as plt

housing.hist(bins=50, figsize=(20,15))
save_fig("attribute_histogram_plots")
plt.show()
```
![A histogram for each numerical attribute](/blog/first-end-to-end-machine-learning-project/pd_data_hist.png)

### Create a Test Set



## References
1. Géron, A. (2022). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems. Japan: O'Reilly Media.

