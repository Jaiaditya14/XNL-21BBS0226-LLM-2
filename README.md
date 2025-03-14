# XNL-21BBS0226-LLM-2
Fintech LLM Fine-Tuning &amp; Optimization in collaboration with XNL-Innovation
Project Overview

This project involves fine-tuning a large language model (LLM) on a fintech dataset to enhance its understanding of financial queries, transactions, and support tickets.

The goal is to optimize the model for latency, efficiency, and accuracy while maintaining high performance for fintech applications.

The project involved three major phases: Data Preparation & Preprocessing, Model Fine-Tuning, and Optimization & Benchmarking.

In the first phase, we curated a simulated fintech dataset containing financial queries, transaction descriptions, and support tickets. Using libraries like Pandas, NLTK, and NumPy, we cleaned and preprocessed the data, ensuring normalization and tokenization of financial terms. The dataset was then split into training, validation, and test sets. A data pipeline was implemented to automate these preprocessing steps, and statistical insights were documented for reference.

For model fine-tuning, we set up a Python environment with PyTorch/TensorFlow, Hugging Face Transformers, and necessary dataset libraries. A pretrained LLM (such as OPT, LLaMA, or GPT) was loaded, and prompt engineering strategies were applied to enhance performance. Hyperparameters, including learning rate, batch size, and number of epochs, were carefully tuned based on validation metrics like perplexity, accuracy, and F1 score. Experiment tracking was integrated using TensorBoard, ensuring that all model checkpoints and hyperparameter configurations were logged for reproducibility.

Performance Metrics:

![image](https://github.com/user-attachments/assets/ddfb703b-4f3e-47d1-9b14-b4998ac72221)

 Example Model Predictions (Before & After Fine-Tuning):
 ![image](https://github.com/user-attachments/assets/e2b2acf0-f773-45b1-9542-65ef68e8a54c)


Finally, in the optimization and benchmarking phase, we measured inference latency, throughput, and memory usage on the test dataset. Techniques such as model quantization and optimized batching were applied to reduce latency, while knowledge distillation was used to minimize model size. Each optimization step was followed by re-benchmarking to assess improvements. Extensive evaluations were conducted to strike a balance between accuracy, efficiency, and speed, comparing the optimized models against baseline performance.

Optimization & Benchmarking Outputs:
![image](https://github.com/user-attachments/assets/ba41d6bd-af18-4040-afc0-0e4dd0ffff1c)


Architecture & Design Diagrams (if applicable) illustrating the data pipeline and fine‑tuning workflow:
![image](https://github.com/user-attachments/assets/ae306719-211e-4713-9bb6-c1cae19ff3a9)

