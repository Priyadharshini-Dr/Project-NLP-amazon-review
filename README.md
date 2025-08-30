# Project-NLP-amazon-review
Amazon review week 6 individual project 
%%writefile README.md
# Automated Customer Reviews Analysis and Recommendation System

## Project Overview

This project develops an NLP-powered system to automate the analysis of customer reviews and generate product recommendations. By classifying sentiment, clustering product categories, and summarizing insights using generative AI, the system aims to provide valuable information for businesses and consumers.

## Problem Statement

Analyzing the vast volume of customer reviews available across various platforms is a time-consuming and manual process. This project addresses this challenge by automating the extraction of insights to improve products and services and provide users with valuable product recommendations.

## Project Goals

- Classify customer reviews into sentiment categories (Positive, Neutral, Negative).
- Cluster product categories into a smaller, more manageable set of meta-categories.
- Summarize product reviews within each category into recommendation articles using generative AI.

## Dataset

The primary dataset used is the **Amazon Product Reviews** dataset, loaded from `/content/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv`.

Key columns used include:
- `reviews.text`: The text content of the customer review.
- `reviews.rating`: The star rating given by the customer (1-5).
- `categories`: Comma-separated product categories associated with the review.
- `name`: The name of the product.
- `asins`: The ASIN (Amazon Standard Identification Number) of the product.

Challenges were encountered during data loading due to CSV formatting issues, which were addressed by using specific `pandas.read_csv` parameters (`engine='python'`, `quotechar='"`, `escapechar='\\'`, `on_bad_lines='skip'`). Missing values in critical columns were handled by dropping corresponding rows.

## Main Tasks

### 1. Review Classification

**Objective:** To determine the sentiment (Positive, Neutral, Negative) expressed in a review.

**Approach:**
- Star ratings were mapped to sentiment labels: 1-2 (Negative), 3 (Neutral), 4-5 (Positive).
- Review text was cleaned by converting to lowercase, removing punctuation and numbers, and handling whitespace.
- A pre-trained transformer model (`distilbert-base-uncased`) was fine-tuned using the cleaned review text and sentiment labels.
- The model was evaluated using metrics like Accuracy, Precision, Recall, F1-score, and a Confusion Matrix.

**Code Reference:** See the code cells related to "Train and evaluate review classification model".

### 2. Product Category Clustering

**Objective:** To group detailed product categories into a smaller number of broader meta-categories (aiming for 4-6).

**Approach:**
- The `categories` column was cleaned and preprocessed.
- TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert the cleaned category strings into numerical feature vectors.
- K-Means clustering was applied to the TF-IDF matrix to group similar categories.
- The resulting clusters were analyzed qualitatively based on the dominant terms within each cluster.

**Findings:** Due to limited diversity in the 'categories' column in the dataset, the clustering resulted in fewer distinct clusters than the target 4-6, primarily separating entries related to 'electronics' and 'hardware'.

**Code Reference:** See the code cells related to "Data preprocessing for clustering" and "Develop and evaluate clustering model".

### 3. Review Summarization Using Generative AI

**Objective:** To generate a short article for each product cluster, recommending top products and highlighting key information.

**Approach:**
- Data from the classification and clustering tasks was merged with product information.
- Within each cluster, top products were identified based on average star rating and total reviews, and the worst product was identified based on low average rating.
- Sample positive and negative reviews were extracted for top products, and negative reviews for the worst product.
- This structured information (product names, metrics, sample reviews) was used to construct prompts for a pre-trained generative AI model (`t5-small`).
- The T5 model was used to generate a summary article for each cluster, intended to cover the top products, their differences (implicitly from highlights/complaints), common complaints, and reasons to avoid the worst product.

**Code Reference:** See the code cells related to "Data preprocessing for summarization" and "Develop and evaluate summarization model".

## How to Run the Code

The project was developed in a Jupyter Notebook environment (like Google Colab).

1.  **Open the Notebook:** Access the provided Jupyter Notebook file.
2.  **Run Cells Sequentially:** Execute the code cells in order from top to bottom. The notebook is structured to perform data loading, preprocessing, model training/development, and data preparation for the web application.
3.  **Ensure Dependencies:** The necessary libraries (`pandas`, `scikit-learn`, `transformers`, `tensorflow`, `matplotlib`, `seaborn`, `gradio`) should be installed in the environment. The first Gradio code cell includes `%pip install gradio`. Other libraries are standard.
4.  **Data File:** Ensure the `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv` file is available at the specified path (`/content/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv`) or update the path in the code.
5.  **Model Weights:** The notebook assumes the transformer models (DistilBERT, T5) are loaded directly from Hugging Face. Internet access is required for this. If running in an environment without internet or with strict access, models might need to be downloaded and loaded from a local path (not implemented in this notebook).

## Web Application

A web application was developed using Gradio to showcase the project's components.

**How to Access:**
- Run the code cell that launches the Gradio application (`demo.launch()`).
- The output of the cell will provide a local URL and, if `share=True` was successful, a public shareable URL (valid for 72 hours).

**How to Interact:**
- **Sentiment Analysis Tab:** Enter a review text in the textbox and click "Analyze Sentiment" to see the predicted sentiment. *Note: Due to model loading issues encountered during development, this tab might display an error.*
- **Product Cluster Analysis Tab:** Select a cluster label from the dropdown menu. The page will update to show:
    - **Top Products:** List of top products in that cluster with metrics and sample reviews.
    - **Worst Product to Avoid:** The product with the lowest average rating in that cluster with metrics and sample negative reviews.
    - **Generated Recommendation Article:** A summary article generated by the T5 model for that cluster.

## Deliverables

- **Source Code:** The Jupyter Notebook containing all the code.
- **README:** This file explaining the project.
- **PDF Report Outline:** `PDF_Report_Outline.txt` outlining the structure of the project report.
- **PPT Presentation Outline:** `PPT_Presentation_Outline.txt` outlining the structure of the project presentation.
- **Generated Summaries:** The `generated_summaries` dictionary containing the text of the generated articles (available in the notebook environment after execution).
- **Deployed App:** Accessible via the Gradio share link provided upon launching the application.

## Evaluation

The project is evaluated based on the implementation and results of data preprocessing, review classification, product category clustering, review summarization, and the deployed web application, as detailed in the project description.


