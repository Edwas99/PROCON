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

After this the most helpfull reviews for the phone that was searched for are fetched and split into positive and negative ones, this is done by using the features we added earlier (helpful and positivity).
```python
#Getting the reviews fo the searched phone (pred)
reviews = reviews[reviews["asin"]==pred[0]].sort_values(by=["helpfulVotes"])

#getting the most helpfull reviews:
helpful_reviews=reviews[reviews["helpful"]==2].sort_values(by=["asin"])

#splitting the data up in positive and negative.
positive_reviews=helpful_reviews[helpful_reviews["positivity"]==1].sort_values(by=["helpfulVotes"])
negative_reviews = helpful_reviews[helpful_reviews["positivity"]==-1].sort_values(by=["helpfulVotes"])

```
The series of positive and negative reviews were then converted into text.

```python
#making the series of reviews to text
def make_fluent_text(serie):
    pd.set_option('display.max_colwidth', None)
    return " ".join(serie.to_string().split())

#positive reviews
pos_out=make_fluent_text(positive_reviews['body'])

#negative reviews
neg_out=make_fluent_text(negative_reviews['body'])

```

Two new functions are then created to clean the text and to calculate the word frequency.
```python
#Clean the text and delete stopwords and keywords
stop = set(stopwords.words('english'))
punc = set(string.punctuation)
keywords1 = reviews["brand"].apply(lambda x: str(x).lower()).unique().tolist()

keywords1.extend(["like", "new", "work", "good","phone", "arrived", "iphone", "one", "seller", "love", "buy", "work", "back","fully", "advertised", "sent", "first", "one", "got"
                  , "really", "came", "however", "device", "great", "going", "use", "look", "problem", "go", "note", "·", "get", "see", "need", "still", "want",
                  "even", "also", "much", "i've", "seem", "two", "mate","10", "2", "better", "way", "issue", "pro", "day", "way", "issue", "galaxy", "never", "3"            
                  ])

lemma = WordNetLemmatizer()
def clean_text(text):
    # Convert the text into lowercase
    text = str(text).lower()
    # Split into list
    wordList = text.split()
    # Remove punctuation
    wordList = ["".join(x for x in word if (x=="'")|(x not in punc)) for word in wordList]
    # Remove stopwords
    wordList = [word for word in wordList if word not in stop]
    # Remove other keywords
    wordList = [word for word in wordList if word not in keywords1]
    # Lemmatisation
    wordList = [lemma.lemmatize(word) for word in wordList]
    return " ".join(wordList)


# Calculate the word frequency
def word_freq_dict(text):
    # Convert text into word list
    wordList = str(text).split()
    # Generate word freq dictionary
    wordFreqDict = {word: wordList.count(word) for word in wordList}
    return wordFreqDict

```
The last function was a function to create the list of the most common keywords, either good or bad ones and a summary of what people have written about this under each keyword. Before implementing this function I tried out many different extractive summary algorithms but in my case the ones that worked the best was KL-sum and LexRank. These ones gave very good summaries but was pretty different. This is because they work in totally different ways, in short one can say that LexRank is a type of Graph based algorithm that is based on the page rank algorithm. KL-sum on the other hand is a algorithm that tries to create a summary with as small KL-divergence as posible, in this case the KL-divergence is the difference between the unigram distribution of the text and the summary. Down below is the code for the function when KL-sum is used. The only difference when using LexRank is that KL-sum is changed to LexRank in the code. This is because I use the sumy library which contains a lot of different sumarizing algorithms.


```python
def write_out_list(text):
    x=word_freq_dict(clean_text(text))
    important_keywords = dict(sorted(x.items(), key=lambda item: item[1], reverse=True)[:5])
    
    for x in range(5):
        important_txt=[sentence + '.' for sentence in str(text).lower().split('.') if list(important_keywords)[x] in sentence]
        my_parser = PlaintextParser.from_string(important_txt,Tokenizer('english'))
        kl_summarizer=KLSummarizer()
        kl_summary=kl_summarizer(my_parser.document,sentences_count=3)
        print ("\n -" + list(important_keywords)[x]+ ":\n")
        for sentence in kl_summary:
            print(sentence)

```

To write out the PROS and CONS for the choosen phone the write_out_list function was called for both the positive and negative reviews.
```python
print("\n PROS and CONS of " + reviews[0:1]["title_y"].to_string())
print("---------------------------------------------------------------------------------------------------------\n")
text=pos_out
print("\n PROS:")
write_out_list(text)
print("----------------------------------------------------------------------------------------------------------\n")
text=neg_out
print("\n CONS:")
write_out_list(text)
```

This resulted in the following PROS & CONS list for the searched phone, in this case the Samsung Galaxy Note 10+
![image](https://user-images.githubusercontent.com/42933199/145978198-60db80bc-308d-4e33-b4d2-06f570fcbe8f.png)
![image](https://user-images.githubusercontent.com/42933199/145978342-c5fd41fd-1847-4fec-bdc6-0fd08b7edb6c.png)



For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/Edwas99/PROCON/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
