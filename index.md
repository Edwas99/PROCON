## Machine Learning project PROCON
This is a project in the course TNM108 at Linköping University. The name PROCON comes from the main idea of the project, to create PROS&CONS lists based on customer reviews.
The Idea is to give the user a good overview of a product based on what other people think without having to read through an ocean of good and bad reviews. I think that good written reviews can give good insights about a product. A product can for example have really good specs but if you read the reviews about the product you get to know that it has a lot of bugs and the specs doesn't really matter because of this. Without further explanation lets dive in to how I tried to do this:
You can use the [editor on GitHub](https://github.com/Edwas99/PROCON/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### The dataset:
The project could be applied for any product, since you can create a PROS&CONS list for anything, however I decided to work with phones in this project. This is because I found a good dataset on Kaggle, which contained one file containing the items (721 phones) and one file which contained 68 000 different reviews  for these phones. For more details about the dataset see [Amazon Cellphone Reviews](https://www.kaggle.com/grikomsn/amazon-cell-phones-reviews/code).



```python
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List
#import the datasets
pd.set_option('display.max_columns', None)
reviews=pd.read_csv("20191226-reviews.csv", encoding='utf-8')
print(reviews.info())
items=pd.read_csv("20191226-items.csv", encoding="utf-8")
print(items.info())



**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```
So this is what the two datsets contained:
<img width="293" alt="datasets_start" src="https://user-images.githubusercontent.com/42933199/145963860-faecf510-d814-4d6e-a023-09d2dc511758.PNG">

To be easier to work with I merged the datasets, changed some names that were the same, droped some unnecesary columns and changed the Amazon product code to a number using labelEncoder.
```python
#Merging and cleaning the datasets
reviews.rename(columns={'rating': 'review_rating'},
          inplace=True, errors='raise')


#changing the amazon product code to a number
labelEncoder = LabelEncoder()
labelEncoder.fit(reviews['asin'])
reviews = pd.merge(reviews, items, how="left", left_on="asin", right_on="asin")
reviews.drop(['name', 'date', 'image'], axis = 1, inplace = True)
reviews['asin'] = labelEncoder.transform(reviews['asin'])
print(reviews.info())
```
This resulted in the following dataset
![image](https://user-images.githubusercontent.com/42933199/145965597-ae30b689-0ba4-4910-9e36-25573216676f.png)
After this, two new features were created. One that told us if a review was helpful by checking how many helfull votes it had. the other feauture checked if the review was positive or not depending on the amount of stars that the review had.


For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/Edwas99/PROCON/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
