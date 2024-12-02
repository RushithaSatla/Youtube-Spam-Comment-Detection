# Youtube-Spam-Comment-Detection

This project is a machine learning-based system to detect spam comments on YouTube using a **Naive Bayes classifier**. It processes comment text, trains a classification model, and provides functionalities to evaluate the model and clean datasets by removing spam comments.

---

## **Features**  
- **Text Preprocessing**: Converts text to lowercase and removes special characters for cleaner input.  
- **Class Distribution Visualization**: Visualizes the distribution of spam and non-spam comments in the dataset.  
- **Naive Bayes Model Training**: Utilizes `MultinomialNB` to classify YouTube comments as spam or not.  
- **Model Evaluation**: Provides accuracy, confusion matrix, and classification report for performance analysis.  
- **Cross-Validation**: Ensures reliability using k-fold cross-validation.  
- **Spam Removal**: Separates spam and non-spam comments and saves non-spam comments to a file.  

---

## **Installation**  
### Prerequisites  
- Python 3.x  
- Libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`  

Install the below required libraries:  
pandas==1.5.3
numpy==1.24.3
matplotlib==3.7.2
scikit-learn==1.2.2



### Clone the Repository  
```bash
git clone https://github.com/RushithaSatla/youtube-spam-detection.git  
cd youtube-spam-detection
```  

---

## **Usage**  
1. **Train and Evaluate the Model**:  
   Run the script to load data, preprocess it, train the model, and evaluate performance.  
   ```bash
   python youtube_spam_detection.py
   ```  

2. **Predict a Single Comment**:  
   Input a comment in the terminal to classify it as spam or not.  

3. **Batch Prediction**:  
   Test a batch of comments with the script’s built-in `batch_predict_comments()` function.

4. **Remove Spam from CSV Files**:  
   Use the `remove_spam_comments()` function to clean a dataset and save only non-spam comments to a new CSV file.  

---

## **Code Overview**  
### Main Functions  
- **`load_data()`**: Loads and concatenates datasets from multiple CSV files.  
- **`preprocess_text()`**: Preprocesses comment text for cleaner input.  
- **`train_model()`**: Trains a Naive Bayes model on the preprocessed data.  
- **`evaluate_model()`**: Evaluates the model using accuracy, confusion matrix, and classification report.  
- **`remove_spam_comments()`**: Removes spam comments and saves non-spam comments to a new file.  

### Example Outputs  
- **Accuracy**: ~90% depending on dataset and preprocessing.  
- **Confusion Matrix**: Visualizes model performance on spam vs. non-spam predictions.  

---

## **Project Structure**  
```
youtube-spam-comment-detection/  
├── youtube_spam_detection.py  
├── datasets/                 
├── non_spam_comments.csv     
└── README.md                 

```  

---


---  

## **License**  
This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).  

--- 

