---
title: Machine Learning Landscape
description: This blog introduces a lot of fundamental concepts (and jargons) that every
  data scientist should know by heart.
date: 2023-03-11T12:14:48.699Z
draft: false
categories: machine-learning
lastmod: 2023-03-11T18:58:15.205Z
cover:
  image: /blog/machine-learning-landscape/machine_learning_landscape_cover.png
slug: machine-learning-landscape
tags:
  - machine-learning
---

The first ML application that really became mainstream, improving the lives of hundereds of millions of people, took over the world back in 1990s: the spam filter.

## What is Machine Learning?
Machine learning is the science (and art) of programming computers so they can learn from data.

More general definition:
> [Machine learning is the] field of study that gives that ability to learn without being explicity programmed.
>
> -- <cite>Arthur Samuel, 1959</cite>

The examples that the system uses to learn are called the training set. Each training example is called a training instance (or sample). The part of a machine learning that learns and makes predictions is called a model. Neural networks and random forests are example of models.

## Traditional vs Machine Learning Approach
In traditional programs, a developer designs logic or algorithms to solve a problem (Figure 1). The program applies this logic to input and computes the output.

![Traditional Approach](/blog/machine-learning-landscape/traditional_approach.png)
<figcaption align="center">Figure 1. The traditional approach</figcaption>

<br/>

But in machine learning, a model is built from the data, and that model is the logic (Figure 2). ML programs have two distict phases: Training and Inference.

![Machine Learning Approach](/blog/machine-learning-landscape/machine_learning_approach.png)
<figcaption align="center">Figure 2. The machine learning approach</figcaption>

## Why use Machine Learning?
Machine learning is great for:
- Problems for which existing solutions require a lot of fine-tuining or long lists of rules
- Complex problems for which using a traditional approach yields no good solution
- Fluctuating evironments (a machine learning system can easily be retrained on new data, always keeping it up to date)
- Getting insights about complex problems and large amounts of data

*Digging into large amounts of data to discover hidden patterns is called data mining, and machine learning excels at it.*

## Few Examples of Applications
Let's look at some concrete examples of machine learning tasks, along with the techniques that can tackle them. This will give you a sense of the incredible breadth and complexity of the tasks that machine learning can easily tackle.

- *Analzing images of products on a production* <br/>
    This is image classificaiton, typically performed using CNNs or sometime transformers
- *Detecting tumors in brain scans* <br/>
    This is semantic image segmentation, using CNNs or transformers
- *Automatically flagging offensive comments on discussion fourms* <br/>
    This is natural language processing (NLP), and more specifically text classification, which can be tackled using RNNs and CNNs
- *Making your app react to voice commands* <br/>
    This is speech recognition, which requires processing audio samples: since they are long and complex sequences, they are typically processed using RNNs, CNNs, or transformers
- *Detecting credit card fraud* <br/>
    This is anomly detection, which can be tackled using isolation forests, Gaussian mixture models, or autoencoders
- *Recommending a product that a client may be interested in, based on past purchases* <br/>
    This is a recommender system
- *Building an intelligent bot for a game* <br/>
    This is often tackled using reinforcement learning, which is a branch of machine learning that trains agent to pick the actions that will maximize the rewards over time, within a given environment

## Types of Machine Learning Systems
There are so many differents types of machine learning systems that it is useful to classify them in broad categories, based on the following criteria:
- How they are supervised during training (training supervision)
- Whether or not they can learn incrementally on the fly (online vs batch learning)
- Whether they can work by simply comparing new data points to known data points, or instead by detecting a predictive model (instance-based vs model-based learning)

These criteria are not exclusive, you can combine them in any way you like. For example, a state-of-the-art spam filter can be an online, model-based, supervised learning system.

Let's look at each one of these categoies a bit more closely.

### Training Supervision
ML systems can be classifed according to the amount and type of supervision they get during training. There are many categories, but we will discuss the main ones:

#### Supervised learning
The training set you feed to the algorithm includes the desired solutions, called labels. A typicla supervised learning task is classification.

#### Unsupervised learning
The training data is unlabeled and the system tried to learn without a teacher. Visualization alogrithms, anomaly detection, and dimensionality reduction are few examples of unsupervised learning.

#### Semi-supervised learning
Since labeling data is usually time-consuming and costly, you will often have plenty of unlabeled instances, and few labeled instances. Some algorithms can deal with data that's partially labeled. Most semi-supervised learning algorithms are combinations of unsupervised and supervised algorithms. 

Some photo-hosting services, such as Google Photos, are good examples of this. Once you upload all your family photos, it automatically recognizes people in multiple photos, and group them, so you can add one label per person and it is able to name everyone in every photo.

#### Self-supervised learning
Another approach to machine learning involves actually generating a fully labeled dataset from a fully unlabeled one. Again, once the whole dataset is labeled, any supervised learning can be used. This appraoch is called self-supervised learning.

*Transferring knowledge from one task to another is called transfer learning.*

#### Reinforcement learning
The learning system, called an agent in this context, can observe the environment, select and perform actions, get rewards or penalties in return. It must then learn by itself what is the best strategy, called a policy, to get the most reward over time. A policy defines what action the agent should choose when it is in a given situation.

### Batch vs Online Learning
#### Batch learning
In batch learning, the system is incapable of learning incrementally: it must be trained using all the available data. This will generally take a lot of time and computing resources, so it is typically done offline.

First the system is trained, and then it is launced into production and runs without learning anymore; it just applies what it has learned.

#### Online learning
In online learning, you train the system incrementally by feeding it data instances sequentially, either individually or in small groups called mini-batches. Each learning step is fast and cheap, so the system can learn about new data on the fly, as it arrives.

### Instance-Based vs Model-Based Learning
#### Instance-based learning
The system learns the examples by heart, then generalizes to new cases by using a similarity measure to compare them to the learned examples.

#### Model-based learning
Another way to generalize from a set of examples is to build a model of these examples and then use that model to make predictions.

Model selection consists in choosing the type of model and fully specifying its architecture. Training a model means running an algorithm to find the model parameters, that will make it best fit the training data, and hopefully make good predictions on new data.

## Summary
Machine learning is about making machines get better at some task by learning from data, instead of having to explicity code rules.
There are so many different types of ML systems: supervised or not, batch or online, instance-based or model-based.
