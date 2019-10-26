## Intracranial Hemorrhage Detection

This blog post is about the challenge that is hosted on kaggle on [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection). 

This post is divided into following parts

1. Overview
2. Basic EDA
3. Data Visualization & Preprocessing
4. Deep Learning Model

### 1. Overview

**What is Intracranial Hemorrhage?**

An intracranial hemorrhage is a type of bleeding that occurs inside the skull. Symptoms include sudden tingling, weakness, numbness, paralysis, severe headache, difficulty with swallowing or vision, loss of balance or coordination, difficulty understanding, speaking , reading, or writing, and a change in level of consciousness or alertness, marked by stupor, lethargy, sleepiness, or coma. Any type of bleeding inside the skull or brain is a medical emergency. It is important to get the person to a hospital emergency room immediately to determine the cause of the bleeding and begin medical treatment. It rquires highly trained specialists review medical images of the patient’s cranium to look for the presence, location and type of hemorrhage. The process is complicated and often time consuming. So as part of this we will be deep learning techniques to detect acute intracranial hemorrhage and its subtypes.

**What am i predicting?**

In this competition our goal is to predict intracranial hemorrhage and its subtypes. Given an image the we need to predict probablity of each subtype. This indicates its a multilabel classification problem.


### 2. Basic EDA 

Lets look at the [data](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data) that is provided.

We have a train.csv containing file names and label indicating whether hemorrhage is present or not and train images folder which is set of [Dicom](https://www.dicomstandard.org/) files (Medical images are stored in dicom formats) and test images folder containing test dicom files.

```python
# load the csv file
train_df = pd.read_csv(input_folder + 'stage_1_train.csv')
train_df.head()
```



Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text
### References

https://my.clevelandclinic.org/health/diseases/14480-intracranial-hemorrhage-cerebral-hemorrhage-and-hemorrhagic-stroke


[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/suryachintu/RSNA-Intracranial-Hemorrhage-Detection/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
