# International Students Psychological Issues Intent Recognition

## Updates
- *Future updates*: Duplicate value processing, text content updates, model training code
- *Initial release*: Basic preprocessing pipeline with data loading, statistical analysis, and missing values analysis

## Project Overview

This project focuses on identifying psychological issues intent recognition for international students. The collected data aims to classify student concerns into five main categories: Studying, Socializing, Dining, Commuting, and Other.

### Data Collection

The dataset was collected through an online questionnaire survey conducted over a period of more than six months. This comprehensive approach allowed us to gather a diverse range of psychological concerns and challenges faced by international students in their daily lives.

## Project Structure

```
.
├── data/                    # Contains the dataset files
├── data_exploration/        # Outputs from data analysis
├── preprocessing/           # Data preprocessing modules
└── main_preprocessing.py    # Main preprocessing script
```

## Intent Categories

The project classifies psychological issues of international students into the following categories:

1. **Studying**: Issues related to academic challenges, course workload, examinations, etc.
2. **Socializing**: Difficulties in making friends, cultural adaptation, social interactions, etc.
3. **Dining**: Problems with food availability, dietary restrictions, food preferences, etc.
4. **Commuting**: Transportation issues, navigating new places, commuting challenges, etc.
5. **Other**: Miscellaneous concerns that don't fall into the above categories.

## Current Implementation

The project currently includes a preprocessing pipeline that performs:

1. **Data Loading**: Loads data from JSONL format
2. **Basic Statistical Analysis**: Provides fundamental insights into the dataset
3. **Missing Values Analysis**: Identifies and analyzes missing data points

## Future Work

The project will be expanded to include:

- Duplicate value processing
- Text content updates and normalization
- Model training for intent classification
- Evaluation metrics and performance analysis
- Deployment options for practical use

## How to Use

To run the preprocessing pipeline:

```bash
python main_preprocessing.py
```

This will execute the current preprocessing steps and save the analysis results in the `data_exploration` directory.

## Project Impact & Significance

This project represents an important contribution to understanding and addressing the psychological challenges faced by international students. By systematically categorizing and analyzing these concerns, we aim to:

- **Improve Support Systems**: Enable universities and support organizations to better tailor their services to address the specific needs of international students.
- **Early Intervention**: Facilitate early identification of students who may need additional support, potentially preventing more serious psychological issues.
- **Resource Allocation**: Help institutions allocate resources more effectively by understanding the most common and pressing concerns.
- **Research Contribution**: Provide valuable data for researchers studying cross-cultural psychology and student well-being.
- **Policy Development**: Inform policy decisions related to international student programs and support services.

The technical expertise demonstrated in this project spans data collection, preprocessing, analysis, and machine learning application to real-world psychological challenges, showcasing a comprehensive approach to solving complex social issues through technology.


