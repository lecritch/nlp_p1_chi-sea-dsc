
Agenda today:
1. Overview of NLP
2. Pre-Prosessing for NLP 
    - Tokenization
    - Stopwords removal
    - Lexicon normalization: lemmatization and stemming
3. Feature Engineering for NLP
    - Bag-of-Words
    - Term frequency-Inverse Document Frequency (tf-idf)


## Activation discussion:

Unmute yourself and share your thoughts about:
 - What makes text mining hard?
 - How is it different than other types of analysis we've done?
 - What are some applications of text mining you've heard of? (or are _interested_ in?)
 
 <img src = 'https://media.giphy.com/media/WqLmcthJ7AgQKwYJbb/giphy.gif' alt="Drawing" style="width: 300px;"  float = 'right'> </img>
 

## 1. Overview of NLP
NLP allows computers to interact with text data in a structured and sensible way. In this section, we will discuss some steps and approaches to common text data analytic procedures. In other words, with NLP, computers are taught to understand human language, its meaning and sentiments. Some of the applications of natural language processing are:
- Chatbots 
- Classifying documents 
- Speech recognition and audio processing 

In this section, we will introduce you to the preprocessing steps, feature engineering, and other steps you need to take in order to format text data for machine learning tasks. 

# NLP process 
<img src="img/nlp_process.png" style="width:1000px;">

# 2. Preprocessing for NLP


```python
#!pip install nltk
#!pip install wordcloud
```

We will be working with a dataset which includes both satirical (The Onion) and real news articles. 

We refer to the entire set of articles as the **corpus**.  


```python
import pandas as pd

corpus = pd.read_csv('data/satire_nosatire.csv')
corpus.shape
```




    (1000, 2)



Each article in the corpus is refered to as a **document**.

It is a balanced dataset with 500 documents of each category. 


```python
corpus.target.value_counts()
```




    1    500
    0    500
    Name: target, dtype: int64



### Tokenization 

In order to convert the texts into data suitable for machine learning, we need to break down the documents into smaller parts. 

The first step in doing that is tokenization.

Tokenization is the process of splitting documents into units of observations. We usually represent the tokens as __n-gram__, where n represent the consecutive words occuring in a document. In the case of unigram (one word token), the sentence "David works here" can be tokenized into?

"David", "works", "here"
"David works", "works here"

Let's consider the first document in our corpus:


```python
first_document = corpus.iloc[0].body
```

There are many ways to tokenize our document. 

It is a long string, so the first way we might consider is to split it by spaces.


```python
first_document.split()[:30]
```




    ['Noting',
     'that',
     'the',
     'resignation',
     'of',
     'James',
     'Mattis',
     'as',
     'Secretary',
     'of',
     'Defense',
     'marked',
     'the',
     'ouster',
     'of',
     'the',
     'third',
     'top',
     'administration',
     'official',
     'in',
     'less',
     'than',
     'three',
     'weeks,',
     'a',
     'worried',
     'populace',
     'told',
     'reporters']



In creating tokens in this way, what problems do we see?

#### Chat out some problems (don't look down)

<img src="https://media.giphy.com/media/ZaiC2DYDRiqhQ269nz/giphy.gif" style="width:1500px;">

We are trying to create a set of tokens with high semantic value.  In other words, we want to isolate text which best represents the meaning in each document.  


## Common text cleaning tasks:  
  1. remove capitalization  
  2. remove punctuation  
  3. remove stopwords  
  4. remove numbers

We could manually perform all of these tasks with string operations

## Capitalization

When we create our matrix of words associated with our corpus, capital letters will mess things up.  The semantic value of a word used at the beginning of a sentence is the same as that same word in the middle of the sentence.  In the two sentences:

sentenceb_one =  "Excessive gerrymandering in small counties suppresses turnout." 
sentence_two =  "Turnout is suppressed in small counties by excessive gerrymandering."

Excessive has the same semantic value, but will be treated as two separate tokens because of capitals.


```python
sentence_one =  "Excessive gerrymandering in small counties suppresses turnout." 
sentence_two =  "Turnout is suppressed in small counties by excessive gerrymandering."

excessive = sentence_one.split(' ')[0]
Excessive = sentence_two.split(' ')[-2]
print(excessive, Excessive)
excessive == Excessive
```

    Excessive excessive





    False




```python
## Manual removal of capitals

manual_cleanup = [token.lower() for token in first_document.split(' ')]
manual_cleanup[:25]
```




    ['noting',
     'that',
     'the',
     'resignation',
     'of',
     'james',
     'mattis',
     'as',
     'secretary',
     'of',
     'defense',
     'marked',
     'the',
     'ouster',
     'of',
     'the',
     'third',
     'top',
     'administration',
     'official',
     'in',
     'less',
     'than',
     'three',
     'weeks,']




```python
print(f"Our initial token set for our document is {len(manual_cleanup)} words long")
```

    Our initial token set for our document is 154 words long


## Punctuation

Like capitals, splitting on white space will create tokens which include punctuation that will muck up our semantics.  

Returning to the above example, 'gerrymandering' and 'gerrymandering.' will be treated as different tokens.


```python
no_punct = sentence_one.split(' ')[1]
punct = sentence_two.split(' ')[-1]
print(no_punct, punct)
no_punct == punct
```

    gerrymandering gerrymandering.





    False




```python
## Manual removal of punctuation
# string library!
import string

string.punctuation

manual_cleanup = [''.join(ch for ch in s if ch not in string.punctuation) for s in manual_cleanup]
manual_cleanup[:25]
```




    ['noting',
     'that',
     'the',
     'resignation',
     'of',
     'james',
     'mattis',
     'as',
     'secretary',
     'of',
     'defense',
     'marked',
     'the',
     'ouster',
     'of',
     'the',
     'third',
     'top',
     'administration',
     'official',
     'in',
     'less',
     'than',
     'three',
     'weeks']



### Stopwords

Stopwords are the filler words in a language: prepositions, articles, conjunctions. They have low semantic value, and almost always need to be removed.  

Luckily, NLTK has lists of stopwords ready for our use.


```python
from nltk.corpus import stopwords
stopwords.__dict__
```




    {'_fileids': ['arabic',
      'azerbaijani',
      'danish',
      'dutch',
      'english',
      'finnish',
      'french',
      'german',
      'greek',
      'hungarian',
      'indonesian',
      'italian',
      'kazakh',
      'nepali',
      'norwegian',
      'portuguese',
      'romanian',
      'russian',
      'slovene',
      'spanish',
      'swedish',
      'tajik',
      'turkish'],
     '_root': FileSystemPathPointer('/Users/johnmaxbarry/nltk_data/corpora/stopwords'),
     '_encoding': 'utf8',
     '_tagset': None,
     '_unload': <bound method LazyCorpusLoader.__load.<locals>._unload of <WordListCorpusReader in '/Users/johnmaxbarry/nltk_data/corpora/stopwords'>>}




```python
stopwords.words('english')[:10]
```




    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"]



Let's see which stopwords are present in our first document.


```python
stops = [token for token in manual_cleanup if token in stopwords.words('english')]
stops[:10]
```




    ['that', 'the', 'of', 'as', 'of', 'the', 'of', 'the', 'in', 'than']




```python
print(f'There are {len(stops)} stopwords in the first document')
```

    There are 63 stopwords in the first document



```python
print(f'That is {len(stops)/len(manual_cleanup): .2%} of our text')
```

    That is  40.91% of our text


Let's also use the FreqDist tool to look at the makeup of our text before and after removal


```python
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
```


```python
fdist = FreqDist(manual_cleanup)
plt.figure(figsize=(10,10))
fdist.plot(30)

```


![png](index_files/index_41_0.png)





    <matplotlib.axes._subplots.AxesSubplot at 0x1a1fb10cf8>




```python
manual_cleanup = [token for token in manual_cleanup if token not in stopwords.words('english')]
```


```python
# We can also customize our stopwords list

custom_sw = stopwords.words('english')
custom_sw.extend(["i'd","say"] )
custom_sw[-10:]
```




    ['wasn',
     "wasn't",
     'weren',
     "weren't",
     'won',
     "won't",
     'wouldn',
     "wouldn't",
     "i'd",
     'say']




```python
manual_cleanup = [token for token in manual_cleanup if token not in custom_sw]
len(manual_cleanup)
```




    90




```python
fdist = FreqDist(manual_cleanup)
plt.figure(figsize=(10,10))
fdist.plot(30)
```


![png](index_files/index_45_0.png)





    <matplotlib.axes._subplots.AxesSubplot at 0x1a20365e48>



#### Numbers

Numbers also usually have low semantic value. Their removal can help improve our models. 

To remove them, we will use regular expressions, a powerful tool which you may already have some familiarity with.

Regex allows us to match strings based on a pattern.  This pattern comes from a language of identifiers, which we can begin exploring on the cheatsheet found here:
  -   https://regexr.com/

Other helpful resources:
  - https://regexcrossword.com/
  - https://www.regular-expressions.info/tutorial.html

We can use regex to isolate numbers




```python
import re

# 1 or more digits
pattern = '\d+'

number = re.findall( pattern, first_document)
number
```




    ['53', '323']




```python
first_document
```




    'Noting that the resignation of James Mattis as Secretary of Defense marked the ouster of the third top administration official in less than three weeks, a worried populace told reporters Friday that it was unsure how many former Trump staffers it could safely reabsorb. “Jesus, we can’t just take back these assholes all at once—we need time to process one before we get the next,” said 53-year-old Gregory Birch of Naperville, IL echoing the concerns of 323 million Americans in also noting that the country was only now truly beginning to reintegrate former national security advisor Michael Flynn. “This is just not sustainable. I’d say we can handle maybe one or two more former members of Trump’s inner circle over the remainder of the year, but that’s it. This country has its limits.” The U.S. populace confirmed that they could not handle all of these pieces of shit trying to rejoin society at once.'




```python
number
```




    ['53', '323']



Sklearn and NLTK provide us with a suite of tokenizers for our text preprocessing convenience.


```python
import nltk
from nltk.tokenize import regexp_tokenize, word_tokenize
from sklearn.feature_extraction.text import tokenize
```


```python
first_document
```




    'Noting that the resignation of James Mattis as Secretary of Defense marked the ouster of the third top administration official in less than three weeks, a worried populace told reporters Friday that it was unsure how many former Trump staffers it could safely reabsorb. “Jesus, we can’t just take back these assholes all at once—we need time to process one before we get the next,” said 53-year-old Gregory Birch of Naperville, IL echoing the concerns of 323 million Americans in also noting that the country was only now truly beginning to reintegrate former national security advisor Michael Flynn. “This is just not sustainable. I’d say we can handle maybe one or two more former members of Trump’s inner circle over the remainder of the year, but that’s it. This country has its limits.” The U.S. populace confirmed that they could not handle all of these pieces of shit trying to rejoin society at once.'




```python
import re
re.findall(r"([a-zA-Z]+(?:'[a-z]+)?)" , "I'd")
```




    ["I'd"]




```python
pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
tokenizer = RegexpTokenizer(r"([a-zA-Z]+(?:[’'][a-z]+)?)")
first_doc = tokenizer.tokenize(first_document)
first_doc = [token.lower() for token in first_doc]
first_doc = [token for token in first_doc if token not in custom_sw]
first_doc
```




    ['noting',
     'resignation',
     'james',
     'mattis',
     'secretary',
     'defense',
     'marked',
     'ouster',
     'third',
     'top',
     'administration',
     'official',
     'less',
     'three',
     'weeks',
     'worried',
     'populace',
     'told',
     'reporters',
     'friday',
     'unsure',
     'many',
     'former',
     'trump',
     'staffers',
     'could',
     'safely',
     'reabsorb',
     'jesus',
     'can’t',
     'take',
     'back',
     'assholes',
     'need',
     'time',
     'process',
     'one',
     'get',
     'next',
     'said',
     'year',
     'old',
     'gregory',
     'birch',
     'naperville',
     'il',
     'echoing',
     'concerns',
     'million',
     'americans',
     'also',
     'noting',
     'country',
     'truly',
     'beginning',
     'reintegrate',
     'former',
     'national',
     'security',
     'advisor',
     'michael',
     'flynn',
     'sustainable',
     'i’d',
     'handle',
     'maybe',
     'one',
     'two',
     'former',
     'members',
     'trump’s',
     'inner',
     'circle',
     'remainder',
     'year',
     'that’s',
     'country',
     'limits',
     'u',
     'populace',
     'confirmed',
     'could',
     'handle',
     'pieces',
     'shit',
     'trying',
     'rejoin',
     'society']



# Stemming

Most of the semantic meaning of a word is held in the root, which is usually the beginning of a word.  Conjugations and plurality do not change the semantic meaning. "eat", "eats", and "eating" all have essentially the same meaning packed into eat.   

Stemmers consolidate similar words by chopping off the ends of the words.

![stemmer](img/stemmer.png)

There are different stemmers available.  The two we will use here are the **Porter** and **Snowball** stemmers.  A main difference between the two is how agressively it stems, Porter being less agressive.


```python
from nltk.stem import *

p_stemmer = PorterStemmer()
s_stemmer = SnowballStemmer(language="english")
```


```python
p_stemmer.stem(first_doc[0])
```




    'note'




```python
s_stemmer.stem(first_doc[0])
```




    'note'




```python
for word in first_doc:
    p_word = p_stemmer.stem(word)
    s_word = s_stemmer.stem(word)
    
    if p_word != s_word:
        print(p_word, s_word)
    
```

    jesu jesus
    can’t can't
    napervil napervill
    i’d i'd
    trump’ trump
    that’ that



```python
first_doc = [p_stemmer.stem(word) for word in first_doc]
```


```python
fdist = FreqDist(first_doc)
plt.figure(figsize=(10,10))
fdist.plot(30)
```


![png](index_files/index_66_0.png)





    <matplotlib.axes._subplots.AxesSubplot at 0x1a2039b390>



# Lemming

Lemming is a bit more sophisticated that the stem choppers.  Lemming uses part of speech tagging to determine how to transform a word.  In that 
Lemmatization returns real words. For example, instead of returning "movi" like Porter stemmer would, "movie" will be returned by the lemmatizer.

- Unlike Stemming, Lemmatization reduces the inflected words properly ensuring that the root word belongs to the language.  It can handle words such as "mouse", whose plural "mice" the stemmers would not lump together with the original. 

- In Lemmatization, the root word is called Lemma. 

- A lemma (plural lemmas or lemmata) is the canonical form, dictionary form, or citation form of a set of words.

![lemmer](img/lemmer.png)



```python
from nltk.stem import WordNetLemmatizer 
  
lemmatizer = WordNetLemmatizer() 

```




    'mouse'




```python
print(f'Mice becomes: {lemmatizer.lemmatize("mice")}')
print(f'Noting becomes: {lemmatizer.lemmatize(first_doc[0])}')
```

    Mice becomes: mouse
    Noting becomes: note



```python
# However, look at the output below:
    
sentence = "He saw the trees down sawed down"
lemmed_sentence = [lemmatizer.lemmatize(token) for token in sentence.split(' ')]
lemmed_sentence
```




    ['He', 'saw', 'the', 'tree', 'down', 'sawed', 'down']




```python
# What should have changed form but didn't?
```

Lemmatizers depend on POS tagging, and defaults to noun.

With a little bit of work, we can POS tag our text.


```python
pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
tokenizer = RegexpTokenizer(r"([a-zA-Z]+(?:[’'][a-z]+)?)")
first_doc = tokenizer.tokenize(first_document)
first_doc = [token.lower() for token in first_doc]
first_doc = [token for token in first_doc if token not in custom_sw]

```


```python
from nltk import pos_tag
# Use nltk's pos_tag to tag our words
first_doc_tagged = pos_tag(first_doc)
first_doc_tagged
```




    [('noting', 'VBG'),
     ('resignation', 'NN'),
     ('james', 'NNS'),
     ('mattis', 'VBP'),
     ('secretary', 'NN'),
     ('defense', 'NN'),
     ('marked', 'VBD'),
     ('ouster', 'JJ'),
     ('third', 'JJ'),
     ('top', 'JJ'),
     ('administration', 'NN'),
     ('official', 'NN'),
     ('less', 'JJR'),
     ('three', 'CD'),
     ('weeks', 'NNS'),
     ('worried', 'VBD'),
     ('populace', 'NN'),
     ('told', 'VBD'),
     ('reporters', 'NNS'),
     ('friday', 'JJ'),
     ('unsure', 'JJ'),
     ('many', 'JJ'),
     ('former', 'JJ'),
     ('trump', 'NN'),
     ('staffers', 'NNS'),
     ('could', 'MD'),
     ('safely', 'RB'),
     ('reabsorb', 'VB'),
     ('jesus', 'NN'),
     ('can’t', 'NNS'),
     ('take', 'VBP'),
     ('back', 'RP'),
     ('assholes', 'NNS'),
     ('need', 'VBP'),
     ('time', 'NN'),
     ('process', 'NN'),
     ('one', 'CD'),
     ('get', 'NN'),
     ('next', 'IN'),
     ('said', 'VBD'),
     ('year', 'NN'),
     ('old', 'JJ'),
     ('gregory', 'NN'),
     ('birch', 'NN'),
     ('naperville', 'FW'),
     ('il', 'NN'),
     ('echoing', 'VBG'),
     ('concerns', 'NNS'),
     ('million', 'CD'),
     ('americans', 'NNS'),
     ('also', 'RB'),
     ('noting', 'VBG'),
     ('country', 'NN'),
     ('truly', 'RB'),
     ('beginning', 'VBG'),
     ('reintegrate', 'VB'),
     ('former', 'JJ'),
     ('national', 'JJ'),
     ('security', 'NN'),
     ('advisor', 'NN'),
     ('michael', 'NN'),
     ('flynn', 'VBP'),
     ('sustainable', 'JJ'),
     ('i’d', 'NN'),
     ('handle', 'NN'),
     ('maybe', 'RB'),
     ('one', 'CD'),
     ('two', 'CD'),
     ('former', 'JJ'),
     ('members', 'NNS'),
     ('trump’s', 'VBP'),
     ('inner', 'JJ'),
     ('circle', 'NN'),
     ('remainder', 'NN'),
     ('year', 'NN'),
     ('that’s', 'JJ'),
     ('country', 'NN'),
     ('limits', 'NNS'),
     ('u', 'JJ'),
     ('populace', 'NN'),
     ('confirmed', 'VBD'),
     ('could', 'MD'),
     ('handle', 'VB'),
     ('pieces', 'NNS'),
     ('shit', 'VB'),
     ('trying', 'VBG'),
     ('rejoin', 'JJ'),
     ('society', 'NN')]




```python
# Then transform the tags into the tags of our lemmatizers
from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):
    '''
    Translate nltk POS to wordnet tags
    '''
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
```


```python
first_doc_tagged = [(token[0], get_wordnet_pos(token[1]))
             for token in first_doc_tagged]
```


```python
first_doc_lemmed = [lemmatizer.lemmatize(token[0], token[1]) for token in first_doc_tagged]
```


```python
first_doc_lemmed[:10]
```




    ['note',
     'resignation',
     'james',
     'mattis',
     'secretary',
     'defense',
     'mark',
     'ouster',
     'third',
     'top']



## Part 3. Feature Engineering for NLP 
The machine learning algorithms we have encountered so far represent features as the variables that take on different value for each observation. For example, we represent individual with distinct education level, income, and such. However, in NLP, features are represented in very different way. In order to pass text data to machine learning algorithm and perform classification, we need to represent the features in a sensible way. One such method is called Bag-of-words (BoW). 

A bag-of-words model, or BoW for short, is a way of extracting features from text for use in modeling. A bag-of-words is a representation of text that describes the occurrence of words within a document. It involves two things:

- A vocabulary of known words.
- A measure of the presence of known words.

It is called a “bag” of words, because any information about the order or structure of words in the document is discarded. The model is only concerned with whether known words occur in the document, not where in the document. The intuition behind BoW is that a document is similar to another if they have similar contents. Bag of Words method can be represented as **Document Term Matrix**, or Term Document Matrix, in which each column is an unique vocabulary, each observation is a document. For example:

- Document 1: "I love dogs"
- Document 2: "I love cats"
- Document 3: "I love all animals"
- Document 4: "I hate dogs"


Can be represented as:

![document term matrix](img/document_term_matrix.png)


```python
# implementing it in python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
# Convert a collection of text documents to a matrix of token counts

vec = CountVectorizer()
X = vec.fit_transform([" ".join(first_doc_lemmed)])


df = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>administration</th>
      <th>advisor</th>
      <th>also</th>
      <th>american</th>
      <th>asshole</th>
      <th>back</th>
      <th>begin</th>
      <th>birch</th>
      <th>can</th>
      <th>circle</th>
      <th>...</th>
      <th>time</th>
      <th>top</th>
      <th>truly</th>
      <th>trump</th>
      <th>try</th>
      <th>two</th>
      <th>unsure</th>
      <th>week</th>
      <th>worry</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 76 columns</p>
</div>



That is not very exciting for one document. The idea is to make a document term matrix for all of the words in our corpus.


```python
corpus
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>body</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Noting that the resignation of James Mattis as...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Desperate to unwind after months of nonstop wo...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nearly halfway through his presidential term, ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Attempting to make amends for gross abuses of ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Decrying the Senate’s resolution blaming the c...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>Britain’s opposition leader Jeremy Corbyn wou...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>996</th>
      <td>Turkey will take over the fight against Islam...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>997</th>
      <td>Malaysia is seeking $7.5 billion in reparatio...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>998</th>
      <td>An Israeli court sentenced a Palestinian to 1...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>999</th>
      <td>At least 22 people have died due to landslide...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 2 columns</p>
</div>




```python
vec = CountVectorizer(token_pattern=r"([a-zA-Z]+(?:'[a-z]+)?)", stop_words=custom_sw)
X = vec.fit_transform(corpus.body[0:2])

df = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>adding</th>
      <th>administration</th>
      <th>advisor</th>
      <th>also</th>
      <th>americans</th>
      <th>another</th>
      <th>assholes</th>
      <th>back</th>
      <th>bank</th>
      <th>beginning</th>
      <th>...</th>
      <th>want</th>
      <th>wants</th>
      <th>weeks</th>
      <th>whether</th>
      <th>whole</th>
      <th>witnesses</th>
      <th>work</th>
      <th>worried</th>
      <th>year</th>
      <th>yet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 176 columns</p>
</div>




```python
vec = CountVectorizer(token_pattern=r"([a-zA-Z]+(?:'[a-z]+)?)", stop_words=custom_sw, ngram_range=[1,2])
X = vec.fit_transform(corpus.body[0:2])

df = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>adding</th>
      <th>adding wants</th>
      <th>administration</th>
      <th>administration official</th>
      <th>advisor</th>
      <th>advisor michael</th>
      <th>also</th>
      <th>also noting</th>
      <th>americans</th>
      <th>americans also</th>
      <th>...</th>
      <th>witnesses want</th>
      <th>work</th>
      <th>work investigating</th>
      <th>worried</th>
      <th>worried populace</th>
      <th>year</th>
      <th>year country</th>
      <th>year old</th>
      <th>yet</th>
      <th>yet another</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 378 columns</p>
</div>



Our document term matrix gets bigger and bigger, with more and more zeros, becoming sparser and sparser.


```python
vec = CountVectorizer(token_pattern=r"([a-zA-Z]+(?:'[a-z]+)?)", stop_words=custom_sw, ngram_range=[1,2])
X = vec.fit_transform(corpus.body)

df = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>aa</th>
      <th>aa united</th>
      <th>aaaaaaah</th>
      <th>aaaaaaah aaaaaah</th>
      <th>aaaaaah</th>
      <th>aaaaaah deal</th>
      <th>aaaaargh</th>
      <th>aaaaargh falls</th>
      <th>aaaah</th>
      <th>aaaah internet</th>
      <th>...</th>
      <th>zuercher kantonalbank</th>
      <th>zverev</th>
      <th>zverev two</th>
      <th>zych</th>
      <th>zych mother</th>
      <th>zych whose</th>
      <th>zzouss</th>
      <th>zzzzzst</th>
      <th>zzzzzst compilation</th>
      <th>zzzzzst shut</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 201647 columns</p>
</div>



We can set upper and lower limits to the word frequency.


```python
corpus.body
```




    0      Noting that the resignation of James Mattis as...
    1      Desperate to unwind after months of nonstop wo...
    2      Nearly halfway through his presidential term, ...
    3      Attempting to make amends for gross abuses of ...
    4      Decrying the Senate’s resolution blaming the c...
                                 ...                        
    995     Britain’s opposition leader Jeremy Corbyn wou...
    996     Turkey will take over the fight against Islam...
    997     Malaysia is seeking $7.5 billion in reparatio...
    998     An Israeli court sentenced a Palestinian to 1...
    999     At least 22 people have died due to landslide...
    Name: body, Length: 1000, dtype: object




```python
vec = CountVectorizer(token_pattern=r"([a-zA-Z]+(?:'[a-z]+)?)", stop_words=custom_sw, ngram_range=[1,2], min_df=2, max_df=25)
X = vec.fit_transform(corpus.body)

df = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>aapl</th>
      <th>aaron</th>
      <th>aaron ross</th>
      <th>ab</th>
      <th>abandon</th>
      <th>abandon conservatives</th>
      <th>abandoned</th>
      <th>abandoned grassroots</th>
      <th>abandoning</th>
      <th>abandoning quarter</th>
      <th>...</th>
      <th>zone</th>
      <th>zone eu</th>
      <th>zones</th>
      <th>zoo</th>
      <th>zoo closed</th>
      <th>zooming</th>
      <th>zor</th>
      <th>zte</th>
      <th>zte corp</th>
      <th>zuckerberg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 31350 columns</p>
</div>



### TF-IDF 
There are many schemas for determining the values of each entry in a document term matrix, and one of the most common schema is called the TF-IDF -- term frequency-inverse document frequency. Essentially, tf-idf *normalizes* the raw count of the document term matrix. And it represents how important a word is in the given document. 

- TF (Term Frequency)
term frequency is simply the frequency of words in a document, and it can be represented as the number of times a term shows up in a document. 

- IDF (inverse document frequency)
IDF represents the measure of how much information the word provides, i.e., if it's common or rare across all documents. It is the logarithmically scaled inverse fraction of the documents that contain the word (obtained by dividing the total number of documents by the number of documents containing the term, and then taking the logarithm of that quotient):

$$idf(w) = log (\frac{number\ of\ documents}{num\ of\ documents\ containing\ w})$$

tf-idf is the product of term frequency and inverse document frequency, or tf * idf. 


```python
tf_vec = TfidfVectorizer(token_pattern=r"([a-zA-Z]+(?:'[a-z]+)?)", stop_words=custom_sw)
X = tf_vec.fit_transform(corpus.body)

df = pd.DataFrame(X.toarray(), columns = tf_vec.get_feature_names())
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>aa</th>
      <th>aaaaaaah</th>
      <th>aaaaaah</th>
      <th>aaaaargh</th>
      <th>aaaah</th>
      <th>aaah</th>
      <th>aaargh</th>
      <th>aah</th>
      <th>aahing</th>
      <th>aap</th>
      <th>...</th>
      <th>zoos</th>
      <th>zor</th>
      <th>zozovitch</th>
      <th>zte</th>
      <th>zuckerberg</th>
      <th>zuercher</th>
      <th>zverev</th>
      <th>zych</th>
      <th>zzouss</th>
      <th>zzzzzst</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23457 columns</p>
</div>




```python
df['zuckerberg'].value_counts()
```




    0.000000    997
    0.062215      1
    0.098663      1
    0.055719      1
    Name: zuckerberg, dtype: int64




```python
tf_vec.vocabulary_
```




    {'noting': 14228,
     'resignation': 17453,
     'james': 11083,
     'mattis': 12876,
     'secretary': 18492,
     'defense': 5328,
     'marked': 12753,
     'ouster': 14711,
     'third': 21061,
     'top': 21290,
     'administration': 293,
     'official': 14445,
     'less': 12042,
     'three': 21095,
     'weeks': 22858,
     'worried': 23185,
     'populace': 15784,
     'told': 21256,
     'reporters': 17365,
     'friday': 8389,
     'unsure': 22125,
     'many': 12705,
     'former': 8261,
     'trump': 21632,
     'staffers': 19831,
     'could': 4654,
     'safely': 18072,
     'reabsorb': 16838,
     'jesus': 11159,
     'take': 20692,
     'back': 1540,
     'assholes': 1254,
     'need': 13936,
     'time': 21176,
     'process': 16146,
     'one': 14516,
     'get': 8762,
     'next': 14045,
     'said': 18088,
     'year': 23300,
     'old': 14477,
     'gregory': 9106,
     'birch': 2136,
     'naperville': 13830,
     'il': 10220,
     'echoing': 6555,
     'concerns': 4203,
     'million': 13240,
     'americans': 723,
     'also': 658,
     'country': 4674,
     'truly': 21631,
     'beginning': 1905,
     'reintegrate': 17182,
     'national': 13870,
     'security': 18510,
     'advisor': 365,
     'michael': 13158,
     'flynn': 8141,
     'sustainable': 20527,
     'handle': 9392,
     'maybe': 12900,
     'two': 21762,
     'members': 13034,
     'inner': 10655,
     'circle': 3622,
     'remainder': 17248,
     'limits': 12176,
     'u': 21782,
     'confirmed': 4262,
     'pieces': 15466,
     'shit': 18877,
     'trying': 21649,
     'rejoin': 17197,
     'society': 19402,
     'desperate': 5613,
     'unwind': 22153,
     'months': 13509,
     'nonstop': 14171,
     'work': 23164,
     'investigating': 10907,
     'russian': 18013,
     'influence': 10582,
     'election': 6661,
     'visibly': 22539,
     'exhausted': 7345,
     'special': 19602,
     'counsel': 4657,
     'robert': 17792,
     'mueller': 13661,
     'powered': 15893,
     'phone': 15415,
     'order': 14631,
     'give': 8824,
     'break': 2565,
     'news': 14036,
     'concerning': 4202,
     'probe': 16133,
     'holiday': 9845,
     'last': 11836,
     'thing': 21049,
     'want': 22724,
     'spending': 19651,
     'family': 7625,
     'cascade': 3151,
     'push': 16491,
     'notifications': 14225,
     'telling': 20886,
     'yet': 23327,
     'another': 885,
     'oligarch': 14488,
     'political': 15735,
     'operative': 14567,
     'highly': 9751,
     'placed': 15559,
     'socialite': 19398,
     'used': 22233,
     'deutsche': 5689,
     'bank': 1676,
     'channels': 3382,
     'funnel': 8496,
     'money': 13473,
     'campaign': 2971,
     'fbi': 7737,
     'director': 5871,
     'firmly': 7979,
     'holding': 9839,
     'power': 15892,
     'button': 2865,
     'adding': 256,
     'wants': 22727,
     'completely': 4131,
     'present': 16004,
     'moment': 13455,
     'celebrating': 3264,
     'loved': 12395,
     'ones': 14518,
     'ruminating': 17978,
     'met': 13111,
     'diplomat': 5856,
     'whether': 22937,
     'someone': 19464,
     'using': 22244,
     'social': 19392,
     'media': 12975,
     'tamper': 20729,
     'witnesses': 23116,
     'calm': 2949,
     'even': 7226,
     'think': 21053,
     'individual': 10507,
     'name': 13816,
     'wait': 22677,
     'hear': 9574,
     'important': 10341,
     'developments': 5704,
     'january': 11102,
     'since': 19085,
     'know': 11618,
     'second': 18483,
     'read': 16854,
     'something': 19467,
     'eric': 7092,
     'involved': 10933,
     'deeply': 5297,
     'previously': 16054,
     'suspected': 20513,
     'pulled': 16404,
     'ruin': 17961,
     'whole': 22977,
     'vacation': 22276,
     'press': 16020,
     'reactivated': 16850,
     'check': 3449,
     'real': 16871,
     'quick': 16605,
     'nearly': 13921,
     'halfway': 9341,
     'presidential': 16018,
     'term': 20940,
     'donald': 6217,
     'continued': 4456,
     'exist': 7356,
     'perpetual': 15309,
     'state': 19913,
     'controversy': 4495,
     'provided': 16336,
     'shortage': 18914,
     'outrageous': 14751,
     'moments': 13458,
     'onion': 14523,
     'looks': 12345,
     'significant': 19039,
     'events': 7231,
     'presidency': 16016,
     'attempting': 1341,
     'make': 12593,
     'amends': 719,
     'gross': 9152,
     'abuses': 101,
     'interior': 10820,
     'department': 5521,
     'unusually': 22143,
     'contrite': 4486,
     'ryan': 18024,
     'zinke': 23433,
     'apologized': 957,
     'monday': 13468,
     'misusing': 13358,
     'government': 8992,
     'funds': 8491,
     'sending': 18589,
     'ethics': 7173,
     'committee': 4045,
     'vase': 22335,
     'change': 3377,
     'anything': 931,
     'exploited': 7424,
     'cabinet': 2896,
     'position': 15832,
     'hope': 9923,
     'accept': 121,
     'beautiful': 1863,
     'example': 7280,
     'qing': 16543,
     'dynasty': 6492,
     'porcelain': 15798,
     'small': 19288,
     'token': 21253,
     'regret': 17136,
     'acknowledging': 198,
     'gift': 8793,
     'spent': 19653,
     'taxpayer': 20806,
     'renovate': 17308,
     'office': 14441,
     'doors': 6245,
     'hoped': 9924,
     'would': 23203,
     'consider': 4344,
     'sincere': 19086,
     'gesture': 8758,
     'apology': 958,
     'wrong': 23238,
     'advantage': 337,
     'lustrous': 12464,
     'glazing': 8844,
     'firing': 7975,
     'evident': 7250,
     'piece': 15464,
     'move': 13612,
     'forgive': 8243,
     'human': 10048,
     'failings': 7581,
     'please': 15621,
     'remember': 17263,
     'man': 12632,
     'detail': 5643,
     'turkey': 21703,
     'violated': 22507,
     'hatch': 9504,
     'act': 218,
     'acted': 219,
     'pawn': 15150,
     'oil': 14466,
     'gas': 8620,
     'industry': 10535,
     'rather': 16801,
     'eyes': 7521,
     'happen': 9420,
     'fall': 7606,
     'unique': 22011,
     'kaolin': 11395,
     'clay': 3717,
     'bought': 2470,
     'mercedes': 13071,
     'benz': 2000,
     'sedans': 18512,
     'find': 7939,
     'parking': 15027,
     'lot': 12376,
     'leave': 11943,
     'today': 21242,
     'plans': 15585,
     'apologize': 956,
     'person': 15324,
     'member': 13033,
     'visiting': 22544,
     'homes': 9871,
     'helicopter': 9642,
     'decrying': 5282,
     'senate': 18585,
     'resolution': 17465,
     'blaming': 2196,
     'crown': 4882,
     'prince': 16086,
     'brutal': 2707,
     'torture': 21322,
     'murder': 13714,
     'journalist': 11260,
     'jamal': 11082,
     'khashoggi': 11497,
     'cruel': 4892,
     'inhumane': 10628,
     'unprecedented': 22065,
     'interference': 10816,
     'sovereign': 19540,
     'kingdom': 11552,
     'internal': 10829,
     'affairs': 385,
     'launched': 11861,
     'rights': 17710,
     'investigation': 10908,
     'harsh': 9479,
     'treatment': 21524,
     'saudi': 18234,
     'ruler': 17970,
     'mohammad': 13437,
     'bin': 2114,
     'salman': 18122,
     'looking': 12343,
     'troubling': 21614,
     'accusations': 175,
     'united': 22018,
     'states': 19918,
     'chosen': 3574,
     'willfully': 23026,
     'knowingly': 11621,
     'place': 15558,
     'fault': 7717,
     'dissident': 6076,
     'president': 16017,
     'claiming': 3666,
     'despot': 5621,
     'made': 12517,
     'endure': 6887,
     'loss': 12373,
     'military': 13222,
     'funding': 8489,
     'ongoing': 14522,
     'war': 22729,
     'yemen': 23317,
     'left': 11961,
     'millions': 13242,
     'homeless': 9869,
     'starving': 19910,
     'matter': 12871,
     'whose': 22987,
     'dismemberment': 6010,
     'may': 12898,
     'ordered': 14632,
     'facing': 7551,
     'criticism': 4846,
     'like': 12153,
     'international': 10831,
     'stage': 19835,
     'powerful': 15894,
     'leader': 11911,
     'basically': 1775,
     'kind': 11543,
     'mistreatment': 13352,
     'seriously': 18661,
     'treat': 21520,
     'authoritarian': 1414,
     'regimes': 17122,
     'purchase': 16456,
     'weapons': 22824,
     'without': 23112,
     'billions': 2109,
     'dollars': 6201,
     'aid': 476,
     'regime': 17121,
     'supposed': 20450,
     'maintain': 12575,
     'basic': 1774,
     'standard': 19872,
     'living': 12262,
     'expected': 7379,
     'charge': 3406,
     'american': 722,
     'senators': 18587,
     'crimes': 4817,
     'humanity': 10052,
     'role': 17833,
     'responsible': 17500,
     'actions': 222,
     'following': 8174,
     'sentencing': 18618,
     'hush': 10105,
     'scandal': 18273,
     'cohen': 3891,
     'granted': 9046,
     'prison': 16108,
     'release': 17215,
     'new': 14026,
     'job': 11202,
     'sources': 19523,
     'wednesday': 22850,
     'confident': 4254,
     'engaging': 6904,
     'honest': 9890,
     'help': 9654,
     'mr': 13632,
     'rehabilitation': 17155,
     'warden': 22732,
     'pete': 15361,
     'clements': 3743,
     'opportunity': 14586,
     'serving': 18674,
     'see': 18518,
     'error': 7110,
     'past': 15089,
     'behaviors': 1915,
     'arrives': 1148,
     'march': 12723,
     'bused': 2836,
     'penitentiary': 15238,
     'manhattan': 12665,
     'eight': 6632,
     'hour': 9991,
     'day': 5170,
     'returning': 17576,
     'night': 14097,
     'strict': 20157,
     'supervision': 20431,
     'furloughs': 8508,
     'allow': 623,
     'use': 22232,
     'skills': 19165,
     'betterment': 2048,
     'community': 4070,
     'chance': 3372,
     'added': 251,
     'request': 17401,
     'rnc': 17772,
     'deputy': 5565,
     'finance': 7931,
     'chairman': 3344,
     'denied': 5483,
     'environment': 7028,
     'easy': 6532,
     'backslide': 1565,
     'criminality': 4821,
     'grimacing': 9130,
     'clutching': 3828,
     'shoulder': 18930,
     'fox': 8308,
     'nfl': 14047,
     'announcer': 869,
     'joe': 11213,
     'buck': 2725,
     'tore': 21305,
     'rotator': 17896,
     'cuff': 4941,
     'awkward': 1495,
     'throw': 21112,
     'sideline': 19002,
     'quarter': 16573,
     'buccaneers': 2722,
     'vs': 22642,
     'cowboys': 4723,
     'game': 8585,
     'hate': 9506,
     'go': 8898,
     'especially': 7134,
     'routine': 17916,
     'erin': 7095,
     'field': 7873,
     'conditions': 4236,
     'thousand': 21087,
     'times': 21184,
     'commentator': 4022,
     'troy': 21621,
     'aikman': 484,
     'went': 22895,
     'hard': 9439,
     'stumbling': 20218,
     'first': 7981,
     'words': 23161,
     'sentence': 18615,
     'still': 20007,
     'ground': 9158,
     'writhing': 23235,
     'pain': 14896,
     'update': 22159,
     'mouth': 13607,
     'look': 12341,
     'right': 17707,
     'twisted': 21757,
     'awkwardly': 1496,
     'shock': 18887,
     'crossed': 4868,
     'face': 7536,
     'bad': 1578,
     'saw': 18253,
     'al': 521,
     'michaels': 13159,
     'tear': 20825,
     'acl': 199,
     'touchdown': 21341,
     'call': 2942,
     'way': 22805,
     'going': 8926,
     'announcing': 872,
     'least': 11942,
     'month': 13507,
     'treated': 21521,
     'concussion': 4225,
     'analyze': 790,
     'play': 15603,
     'conversion': 4515,
     'categorically': 3204,
     'denying': 5517,
     'allegations': 599,
     'tactic': 20666,
     'unconstitutional': 21861,
     'unfairly': 21954,
     'targeted': 20761,
     'players': 15608,
     'protested': 16317,
     'anthem': 900,
     'commissioner': 4036,
     'roger': 17824,
     'goodell': 8950,
     'released': 17216,
     'statement': 19916,
     'sunday': 20389,
     'defending': 5326,
     'subject': 20244,
     'panthers': 14959,
     'safety': 18076,
     'reid': 17160,
     'random': 16743,
     'stop': 20060,
     'frisk': 8409,
     'searches': 18459,
     'simply': 19078,
     'keep': 11442,
     'clean': 3720,
     'provide': 16335,
     'safe': 18067,
     'benefits': 1985,
     'case': 3153,
     'received': 16936,
     'anonymous': 882,
     'tip': 21205,
     'suspicious': 20523,
     'mask': 12808,
     'obscuring': 14362,
     'acting': 220,
     'aggressively': 443,
     'towards': 21365,
     'decided': 5243,
     'inform': 10589,
     'proper': 16257,
     'authorities': 1417,
     'conference': 4243,
     'advised': 360,
     'loitering': 12320,
     'line': 12185,
     'scrimmage': 18416,
     'sensitive': 18606,
     'areas': 1080,
     'avoid': 1468,
     'similar': 19063,
     'incidents': 10416,
     'moving': 13620,
     'forward': 8288,
     'described': 5586,
     'unidentified': 21989,
     'object': 14340,
     'hands': 9403,
     'description': 5589,
     'prompted': 16238,
     'officials': 14449,
     'detain': 5647,
     'perform': 15271,
     'thorough': 21078,
     'strip': 20172,
     'search': 18457,
     'relieved': 17232,
     'discover': 5939,
     'football': 8191,
     'single': 19097,
     'player': 15607,
     'code': 3871,
     'conduct': 4238,
     'teammates': 20823,
     'currently': 4984,
     'held': 9639,
     'questioning': 16596,
     'suspicion': 20521,
     'gang': 8592,
     'related': 17200,
     'activity': 230,
     'eyewitnesses': 7524,
     'observed': 14367,
     'wearing': 22827,
     'clothes': 3807,
     'bearing': 1850,
     'colors': 3965,
     'threatening': 21092,
     'logo': 12315,
     'quashing': 16578,
     'rumors': 17980,
     'team': 20822,
     'early': 6506,
     'exit': 7363,
     'las': 11830,
     'vegas': 22348,
     'oakland': 14322,
     'raiders': 16687,
     'announced': 866,
     'entirety': 7000,
     'home': 9865,
     'schedule': 18310,
     'head': 9546,
     'coach': 3837,
     'jon': 11239,
     'gruden': 9177,
     'backyard': 1573,
     'really': 16886,
     'perfect': 15269,
     'venue': 22382,
     'fact': 7552,
     'playing': 15611,
     'yard': 23293,
     'nowhere': 14254,
     'else': 6710,
     'league': 11921,
     'proposed': 16272,
     'half': 9340,
     'acre': 213,
     'plot': 15646,
     'nestled': 13991,
     'bay': 1823,
     'area': 1079,
     'suburbs': 20294,
     'boasted': 2309,
     'natural': 13890,
     'surface': 20466,
     'enough': 6950,
     'improvised': 10380,
     'seating': 18474,
     'accommodate': 137,
     'dozens': 6311,
     'hardcore': 9440,
     'faithful': 7597,
     'mistake': 13349,
     'rocking': 17814,
     'mean': 12943,
     'derek': 5571,
     'carr': 3118,
     'delivering': 5414,
     'strikes': 20167,
     'deck': 5254,
     'plus': 15667,
     'fans': 7641,
     'love': 12394,
     'amenities': 720,
     'room': 17867,
     'black': 2172,
     'hole': 9842,
     'garbage': 8603,
     'cans': 3027,
     'got': 8977,
     'bathrooms': 1799,
     'crockpot': 4855,
     'full': 8469,
     'chili': 3516,
     'house': 9993,
     'better': 2047,
     'spend': 19650,
     'several': 18701,
     'admitted': 306,
     'despite': 5619,
     'treacherous': 21510,
     'clothesline': 3808,
     'exposed': 7445,
     'tree': 21528,
     'roots': 17875,
     'far': 7645,
     'preferable': 15961,
     'games': 8588,
     'goddamn': 8910,
     'baseball': 1767,
     'humane': 10049,
     'deal': 5194,
     'suffering': 20333,
     'cleveland': 3748,
     'browns': 2691,
     'tuesday': 21668,
     'euthanized': 7212,
     'dawg': 5166,
     'pound': 15881,
     'rabies': 16642,
     'outbreak': 14714,
     'part': 15042,
     'heartbroken': 9583,
     'cutting': 5020,
     'lives': 12259,
     'short': 18913,
     'putting': 16513,
     'option': 14612,
     'owner': 14844,
     'jimmy': 11188,
     'haslam': 9493,
     'revealed': 17592,
     'concern': 4200,
     'piqued': 15525,
     'began': 1897,
     'chewing': 3493,
     'plastic': 15594,
     'seats': 18475,
     'salivating': 18119,
     'uncontrollably': 21863,
     'discovered': 5940,
     'late': 11841,
     'cure': 4974,
     'administered': 292,
     'put': 16499,
     'seemed': 18531,
     'fun': 8477,
     'approachable': 1016,
     'getting': 8766,
     'aggressive': 442,
     'bit': 2153,
     'seem': 18530,
     'quality': 16563,
     'life': 12126,
     'never': 14020,
     'battling': 1818,
     'constant': 4374,
     'seizures': 18550,
     'hydrophobia': 10122,
     'resulting': 17532,
     'making': 12599,
     'impossible': 10355,
     'drink': 6368,
     'beer': 1887,
     'emphasized': 6799,
     'sadness': 18066,
     'mercy': 13078,
     'killing': 11535,
     'assured': 1283,
     'comfort': 3997,
     'knowing': 11620,
     'suffer': 20331,
     'recognition': 16967,
     'brave': 2544,
     'altruistic': 679,
     'risk': 17747,
     'health': 9569,
     'greater': 9081,
     'good': 8948,
     'pentagon': 15248,
     'thursday': 21132,
     'honor': 9900,
     'sacrifices': 18052,
     'jerseys': 11156,
     'throughout': 21111,
     'december': 5235,
     'every': 7239,
     'week': 22853,
     'men': 13048,
     'gridiron': 9116,
     'bodies': 2320,
     'soldiers': 19435,
     'wear': 22825,
     'caps': 3059,
     'show': 18938,
     'support': 20442,
     'spokesperson': 19716,
     'amato': 692,
     'active': 223,
     'duty': 6473,
     'sporting': 19729,
     'gear': 8664,
     'teams': 20824,
     'raise': 16700,
     'awareness': 1489,
     'people': 15251,
     'aside': 1203,
     'preserve': 16010,
     'families': 7624,
     'travel': 21499,
     'cities': 3647,
     'across': 217,
     'uphold': 22172,
     'nation': 13869,
     'traditions': 21408,
     'battered': 1808,
     'bruised': 2696,
     'years': 23304,
     'often': 14459,
     'cut': 5014,
     'sit': 19125,
     'barracks': 1739,
     'enjoy': 6929,
     'freedom': 8360,
     'end': 6862,
     'service': 18670,
     'hopefully': 9926,
     'shows': 18948,
     'officers': 14443,
     'true': 21629,
     'heroes': 9701,
     'welling': 22887,
     'emotion': 6786,
     'upon': 22181,
     'finally': 7929,
     'setting': 18683,
     'foot': 8189,
     'hallowed': 9347,
     'tile': 21170,
     'college': 3941,
     'senior': 18593,
     'anthony': 901,
     'harper': 9471,
     'fulfilled': 8465,
     'lifelong': 12131,
     'dream': 6347,
     'saturday': 18231,
     'allowed': 626,
     'shower': 18944,
     'notre': 14232,
     'dame': 5079,
     'showers': 18945,
     'knew': 11604,
     'worked': 23167,
     'quit': 16622,
     'takes': 20698,
     'lather': 11845,
     'ovation': 14765,
     'brian': 2603,
     'kelly': 11452,
     'tossed': 21328,
     'conditioner': 4234,
     'bench': 1969,
     'locker': 12297,
     'watch': 22783,
     'wishing': 23094,
     'always': 684,
     'thought': 21082,
     'soap': 19384,
     'goes': 8919,
     'grit': 9144,
     'determination': 5668,
     'anyone': 930,
     'achieve': 186,
     'bathe': 1794,
     'entire': 6998,
     'witness': 23114,
     'announcement': 867,
     'perceived': 15260,
     'major': 12588,
     'reassurance': 16908,
     'parents': 15014,
     'children': 3514,
     'low': 12401,
     'cognitive': 3889,
     'abilities': 49,
     'subpar': 20260,
     'reasoning': 16903,
     'pediatric': 15199,
     'experts': 7404,
     'report': 17360,
     'claims': 3667,
     'contact': 4413,
     'poses': 15826,
     'little': 12247,
     'brains': 2521,
     'already': 656,
     'well': 22883,
     'tackle': 20663,
     'long': 12332,
     'known': 11623,
     'high': 9743,
     'sport': 19727,
     'particularly': 15052,
     'poor': 15772,
     'guys': 9281,
     'knuckle': 11625,
     'draggers': 6321,
     'away': 1490,
     'lose': 12368,
     'university': 22028,
     'chicago': 3497,
     'childhood': 3512,
     'development': 5702,
     'expert': 7403,
     'dr': 6314,
     'maureen': 12884,
     'clifford': 3759,
     'neuropathological': 14007,
     'research': 17420,
     'led': 11954,
     'conclusion': 4215,
     'chronic': 3592,
     'traumatic': 21497,
     'encephalopathy': 6840,
     'caused': 3226,
     'repeated': 17336,
     'severe': 18702,
     'impacts': 10297,
     'mitigated': 13363,
     'percent': 15261,
     'cases': 3154,
     'youth': 23364,
     'presented': 16006,
     'signs': 19044,
     'huge': 10037,
     'dumbass': 6437,
     'clearly': 3737,
     'couple': 4680,
     'screws': 18413,
     'loose': 12350,
     'course': 4692,
     'cte': 4928,
     'danger': 5099,
     'comes': 3996,
     'sports': 19730,
     'ages': 435,
     'crucial': 4886,
     'healthy': 9571,
     'neurological': 14005,
     'growth': 9175,
     'symptoms': 20623,
     'mood': 13515,
     'swings': 20586,
     'difficult': 5801,
     'thinking': 21055,
     'memory': 13047,
     'sounds': 19516,
     'kid': 11515,
     'precious': 15935,
     'dude': 6425,
     'bonehead': 2379,
     'blocking': 2256,
     'tackling': 20664,
     'hit': 9806,
     'crossing': 4871,
     'routes': 17915,
     'reasons': 16904,
     'idiot': 10188,
     'study': 20211,
     'concluded': 4213,
     'halfwits': 9342,
     'shot': 18926,
     'success': 20301,
     'staring': 19890,
     'wide': 22995,
     'eyed': 7517,
     'table': 20651,
     'unopened': 22059,
     'presents': 16009,
     'largely': 11823,
     'ignored': 10214,
     'guests': 9226,
     'local': 12288,
     'rick': 17683,
     'joseph': 11252,
     'reportedly': 17363,
     'watched': 22785,
     'helplessly': 9660,
     'white': 22966,
     'elephant': 6679,
     'exchange': 7299,
     'devolved': 5721,
     'friends': 8395,
     'chatting': 3437,
     'nice': 14059,
     'christ': 3579,
     'turn': 21709,
     'pick': 15448,
     'ago': 450,
     'derailed': 5568,
     'everyone': 7242,
     'blabbing': 2171,
     'fucking': 8451,
     'christmas': 3588,
     'forced': 8207,
     'listen': 12225,
     'engaged': 6901,
     'pleasant': 15620,
     'conversations': 4513,
     'favorite': 7722,
     'recipes': 16947,
     'beloved': 1961,
     'memories': 13045,
     'festive': 7843,
     'season': 18467,
     'ugh': 21790,
     'disaster': 5906,
     'chumps': 3597,
     'strategizing': 20122,
     'screw': 18411,
     'best': 2030,
     'instead': 10729,
     'wasting': 22781,
     ...}




```python
df['administration'].value_counts()
```




    0.000000    933
    0.056537      2
    0.130272      2
    0.045361      1
    0.085129      1
               ... 
    0.027153      1
    0.031600      1
    0.032887      1
    0.037210      1
    0.065937      1
    Name: administration, Length: 66, dtype: int64



# Pair: 

For a final exercise, work through in pairs the following exercise.

Create a document term matrix of the 1000 document corpus.  The vocabulary should have no stopwords, no numbers, no punctuation, and be lemmatized.  The Document-Term Matrix should be created using tfidf.


```python
#__SOLUTION__
corpus = pd.read_csv('data/satire_nosatire.csv')

```


```python
#__SOLUTION__
def doc_preparer(doc, stop_words=custom_sw):
    '''
    
    :param doc: a document from the satire corpus 
    :return: a document string with words which have been 
            lemmatized, 
            parsed for stopwords, 
            made lowercase,
            and stripped of punctuation and numbers.
    '''
    
    regex_token = RegexpTokenizer(r"([a-zA-Z]+(?:’[a-z]+)?)")
    doc = regex_token.tokenize(doc)
    doc = [word.lower() for word in doc]
    doc = [word for word in doc if word not in stop_words]
    doc = pos_tag(doc)
    doc = [(word[0], get_wordnet_pos(word[1])) for word in doc]
    lemmatizer = WordNetLemmatizer() 
    doc = [lemmatizer.lemmatize(word[0], word[1]) for word in doc]
    return ' '.join(doc)

```


```python
#__SOLUTION__
docs = [doc_preparer(doc) for doc in corpus.body]
```


```python
#__SOLUTION__
tf_idf = TfidfVectorizer(min_df = .05)
X = tf_idf.fit_transform(docs)

df = pd.DataFrame(X.toarray())
df.columns = tf_idf.vocabulary_
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>secretary</th>
      <th>mark</th>
      <th>third</th>
      <th>top</th>
      <th>administration</th>
      <th>official</th>
      <th>less</th>
      <th>three</th>
      <th>week</th>
      <th>tell</th>
      <th>...</th>
      <th>reserve</th>
      <th>economy</th>
      <th>october</th>
      <th>militant</th>
      <th>reporting</th>
      <th>seven</th>
      <th>syria</th>
      <th>dec</th>
      <th>minimum</th>
      <th>lawmaker</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.157285</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.077917</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.10626</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.131235</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.208009</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.077732</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.061092</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.128231</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.149826</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 660 columns</p>
</div>


