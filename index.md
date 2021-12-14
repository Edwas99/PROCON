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

![image](https://user-images.githubusercontent.com/42933199/145967637-ba564915-73f3-46c7-a159-423a1400466d.png)

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

```python
#Creating two new features, Positivity and helpful
reviews["positivity"] = reviews["review_rating"].apply(lambda x: 1 if x>3 else(0 if x==3 else -1))
reviews["helpful"] = reviews["helpfulVotes"].apply(lambda x: 2 if x>10 else(1 if x>5 else 0))
print(reviews.sort_values(by=["totalReviews"]) )
```
This gave us an easy way to see if a review was positive or negative and how helpful people thought it was.
![image](https://user-images.githubusercontent.com/42933199/145968448-f88fa622-af94-4951-8348-f9cc28d42c1f.png)

The next step was to give the "user" a posibility to search for a phone, Here I vectorized the titles for the phones and used SGDClassifier (SVM) to classify the searched phone. I tried to look for the right phone using cos similarity aswell, this often resulted in the same phone as the classifier.

```python
# Vectorize the titles
docs_train, docs_test, y_train, y_test = train_test_split(reviews['title_y'], reviews["asin"], test_size = 0.20, random_state = 12)
vectorizer= CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, max_features=3000)
reviews_counts = vectorizer.fit_transform(docs_train)

Tfmer = TfidfTransformer()
reviews_tfidf = Tfmer.fit_transform(reviews_counts)

searched_phone = [
    'Samsung Galaxy Note 10+ Plus Factory Unlocked Cell Phone with 512GB (U.S. Warranty)',
 ]  

phone_vector=vectorizer.transform(searched_phone)
phone_tfidf=Tfmer.transform(phone_vector)

#Classify the searched phone using SVM
clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42
,max_iter=5, tol=None)

clf.fit(reviews_tfidf,y_train)

docs_test_counts = vectorizer.transform(docs_test)
docs_test_tfidf = Tfmer.transform(docs_test_counts)

y_pred = clf.predict(docs_test_tfidf)
#print(sklearn.metrics.accuracy_score(y_test, y_pred))
pred = clf.predict(phone_tfidf)

#Check cos similarity aswell
cos_similarity = cosine_similarity(reviews_tfidf, phone_tfidf[0])
index_max = max(range(len(cos_similarity)), key=cos_similarity.__getitem__)
```

I got over 92 percent correct when classifying this way, but this was when testing with the same data but split up. So When a real user search for "Iphone 8" The program has a hard time knowing which IPhone it is and can many times take the wrong phone. To get the phone that we want we have to be very wpecific and type it very simliar to what the title is on the Amazon page.
For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/Edwas99/PROCON/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
