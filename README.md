## Intracranial Hemorrhage Detection

This blog post is about the challenge that is hosted on kaggle on [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection). 

This post is divided into following parts

1. Overview
2. Basic EDA
3. Data Visualization & Preprocessing
4. Deep Learning Model

### 1. Overview

##### What is Intracranial Hemorrhage?

An intracranial hemorrhage is a type of bleeding that occurs inside the skull. Symptoms include sudden tingling, weakness, numbness, paralysis, severe headache, difficulty with swallowing or vision, loss of balance or coordination, difficulty understanding, speaking , reading, or writing, and a change in level of consciousness or alertness, marked by stupor, lethargy, sleepiness, or coma. Any type of bleeding inside the skull or brain is a medical emergency. It is important to get the person to a hospital emergency room immediately to determine the cause of the bleeding and begin medical treatment. It rquires highly trained specialists review medical images of the patient’s cranium to look for the presence, location and type of hemorrhage. The process is complicated and often time consuming. So as part of this we will be deep learning techniques to detect acute intracranial hemorrhage and its subtypes.

Hemorrhage Types

1. Epidural
2. Intraparenchymal    
3. Intraventricular
4. Subarachnoid 
5. Subdural
6. Any

##### What am i predicting?

In this competition our goal is to predict intracranial hemorrhage and its subtypes. Given an image the we need to predict probablity of each subtype. This indicates its a multilabel classification problem.


### 2. Basic EDA 

Lets look at the [data](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data) that is provided.

We have a train.csv containing file names and label indicating whether hemorrhage is present or not and train images folder which is set of [Dicom](https://www.dicomstandard.org/) files (Medical images are stored in dicom formats) and test images folder containing test dicom files.

```python
# load the csv file
train_df = pd.read_csv(input_folder + 'stage_1_train.csv')
train_df.head()
```
It consists of two columns ID and Label. ID has a format FILE_ID_SUB_TYPE for example ID_63eb1e259_epidural so ID_63eb1e259 is file id and epidural is subtype and Label indicating whether subtype hemorrhage is present or not.

Lets seperate file names and subtypes

```python
# extract subtype
train_df['sub_type'] = train_df['ID'].apply(lambda x: x.split('_')[-1])
# extract filename
train_df['file_name'] = train_df['ID'].apply(lambda x: '_'.join(x.split('_')[:2]) + '.dcm')
train_df.head()
```
```python
train_df.shape
````
Output : (4045572, 4)

```python
print("Number of train images availabe:", len(os.listdir(path_train_img)))
```
Output : Number of train images availabe: 674258

The csv file has a shape of (4045572, 4). For every file(dicom file) present in the train folder has 6 entries in csv indicating possible 6 subtype hemorrhages.

Lets check the files available for each subtype

```python
plt.figure(figsize=(16, 6))
graph = sns.countplot(x="sub_type", hue="Label", data=(train_df))
graph.set_xticklabels(graph.get_xticklabels(),rotation=90)
plt.show()
```
Lets check the counts for each subtype

##### Epidural

```python
train_df[train_df['sub_type'] == 'epidural']['Label'].value_counts()
```
Output: 

0    671501

1      2761

Name: Label, dtype: int64

For epidural sub type we have 6,71,501 images labeled as 0 and 2,761 labelled as 1.

##### Intraparenchymal

```python
train_df[train_df['sub_type'] == 'intraparenchymal']['Label'].value_counts()
```
Output: <br/>
0    641698<br/>
1     32564<br/>
Name: Label, dtype: int64

For intraparenchymal sub type we have 6,41,698 images labeled as 0 and 32,564 labelled as 1.


##### Intraparenchymal

```python
train_df[train_df['sub_type'] == 'intraparenchymal']['Label'].value_counts()
```
Output: <br/>
0    650496<br/>
1     23766<br/>
Name: Label, dtype: int64

For intraparenchymal sub type we have 6,50,496 images labeled as 0 and 23,766 labelled as 1.

##### Subarachnoid

```python
train_df[train_df['sub_type'] == 'subarachnoid']['Label'].value_counts()
```
Output: <br/>
0    642140<br/>
1     32122<br/>
Name: Label, dtype: int64

For subarachnoid sub type we have 6,42,140 images labeled as 0 and 32,122 labelled as 1.


##### Subdural

```python
train_df[train_df['sub_type'] == 'subdural']['Label'].value_counts()
```
Output: <br/>
0    631766<br/>
1     42496<br/>
Name: Label, dtype: int64

For Subdural sub type we have 6,31,766 images labeled as 0 and 42,496 labelled as 1.


##### Any

```python
train_df[train_df['sub_type'] == 'any']['Label'].value_counts()
```
Output: <br/>
0    577159<br/>
1     97103<br/>
Name: Label, dtype: int64

For any sub type we have 5,77,159 images labeled as 0 and 97,103 labelled as 1.




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
