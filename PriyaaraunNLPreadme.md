Project NLP | Automated Customer Reviews: README


Project Goal:

This project aims to develop a product review system powered by NLP models that aggregate customer feedback from different sources. The key tasks include classifying reviews, clustering product categories, and using generative AI to summarize reviews into recommendation articles.

Problem Statement:

With thousands of reviews available across multiple platforms, manually analyzing them is inefficient. This project seeks to automate the process using NLP models to extract insights and provide users with valuable product recommendations.
Here is a step-by-step guide to the project, outlining the methods and tools used, and providing a clear explanation for each choice.


Project NLP | Automated Customer Review System


Overview

This project provides an end-to-end solution for analyzing and summarizing customer reviews. By leveraging state-of-the-art NLP models, we transform raw, unstructured review data into actionable business intelligence. The system includes:
Sentiment Classification: A model that automatically determines if a review is positive, neutral, or negative.
Category Clustering: A method to intelligently group diverse product categories into more cohesive "meta-categories."
Generative AI Summarization: A powerful AI that synthesizes review insights into well-structured, human-readable recommendation articles.
Interactive Web App: A Gradio-based application for seamless user interaction and demonstration.

What I Used: A Step-by-Step Breakdown

This project follows a clear, modular pipeline, with each step building upon the last.

Step 1: Data Preprocessing and Preparation

Data Sources: We began by merging three different Amazon review datasets (1429_1.csv, Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv, and Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv) into a single, unified dataset using pandas.
Data Cleaning: Key fields like reviews.rating and reviews.text were identified, and any rows with missing values were removed to ensure data quality for downstream tasks.
Sentiment Labeling: Star ratings were mapped to sentiment labels (1-2 to Negative, 3 to Neutral, and 4-5to Positive), which served as the ground truth for our classification model.
Handling Imbalance: The raw dataset exhibited a significant class imbalance, with a large number of positive reviews. We addressed this crucial issue by resampling the data, down-sampling the majority class and up-sampling the minority classes to create a perfectly balanced dataset for robust model training.

Step 2: Sentiment Classification Model

The core of our sentiment analysis is a transformer-based model.
Model Choice: We chose distilbert-base-uncased, a lightweight yet powerful pre-trained model from Hugging Face. Its efficiency and strong performance on general-purpose language tasks made it an ideal choice for this project, allowing us to build a high-quality classifier without the need for extensive computational resources.
Training: The model was fine-tuned on our balanced dataset using the Hugging Face Trainer. This process adapts the pre-trained model's knowledge to our specific sentiment classification task.
Evaluation: The model's performance was evaluated using standard metrics including Accuracy, Precision, Recall, and F1-score, as well as a Confusion Matrix to provide a detailed view of its performance across all three sentiment classes.

Step 3: Product Category Clustering

To make the vast product catalog understandable, we created meaningful meta-categories.
Methodology: We used the sentence-transformers library with the all-MiniLM-L6-v2 model to generate high-dimensional embeddings for each unique product category string. These embeddings capture the semantic meaning of the categories.
Clustering: We applied the K-Means algorithm to these embeddings, grouping similar product categories based on their vector representations. This approach allowed us to reduce the number of categories from dozens to a handful, making the data more accessible for high-level analysis.

Step 4: Review Summarization

This generative component provides the final, valuable output of the system.
Model Choice: The facebook/bart-large-cnn model was selected for this task due to its proven effectiveness in abstractive text summarization.
Prompt Engineering: Instead of feeding the raw reviews, we developed a structured approach to generate articles. For each cluster, we first identified the top 3 products, their key strengths (from positive review keywords), top complaints (from negative review keywords), and the single worst product with its corresponding negative keywords. This organized data was then used as a prompt to guide the generative model, ensuring the summaries are relevant and focused.

Why MLOps Tools Were Used

In a machine learning project, managing experiments and ensuring reproducibility are critical. We used MLflow and Weights & Biases (W&B) to address these challenges.
MLflow: MLflow served as our primary experiment tracking tool. Its purpose is to create a complete and transparent record of each training run. We logged crucial information such as:
Hyperparameters: The fine-tuning parameters like num_train_epochs and learning_rate.
Metrics: The performance results (accuracy, loss, etc.) on the training and validation sets.
Artifacts: This allows us to save model checkpoints and other output files, providing an audit trail for our work. By logging this data, we can easily compare different runs, debug regressions, and ensure that our results are fully reproducible—meaning anyone can recreate the exact model and outcome.
Weights & Biases (W&B): While MLflow provides a solid foundation, W&B was used for enhanced visualization and collaboration. It offers a rich, web-based dashboard that automatically tracks metrics in real-time. This allowed us to monitor the fine-tuning process as it happened, visualize trends in loss and accuracy across epochs, and perform deep comparisons between different models. This level of insight is invaluable for fine-tuning and hyperparameter optimization, helping us quickly identify the most promising model configurations.

Deployment with Gradio and Hugging Face

The project is designed for deployment as a user-friendly web application.
Gradio: We utilized Gradio to create an intuitive interface. It allows users to interact with our key NLP components directly through a web browser. The app is structured with two tabs, providing a clean separation between the sentiment analysis and the recommendation article generator.
Hugging Face Spaces: This platform serves as a host for our Gradio application. By logging in to Hugging Face and pushing our trained model and tokenizer to a repository, we can then deploy the Gradio app to a free Hugging Face Space. This provides a simple, scalable, and shareable platform for others to test and interact with our model without any local setup.

