# #100DaysOfMLCode

## Task 1: Topic modeling the Blog Authorship Corpus

The [blog authorship corpus](http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm) consists of the collected posts of 19,320 bloggers gathered from blogger.com in August 2004. The corpus is available in an xml format that contains tags about age and gender of the blog author. 

I had previously parsed out those containing over 500 words, which gave me a total of 2,49,844 male blogs and 2,54,879 female blogs. This task leverages two algorithms to observe the relevant topics that prevailing in the blogs of both the genders:

* [lda.py](https://github.com/Saurav0074/100daysOfMLCode/blob/master/blog_topic_modeling/code/lda.py) uses the gensim and mallet's implementation of Latent Dirichlet Allocation (LDA) along with the code to for the optimal number of topics on gensim's classical LDA based upon their coherence scores.
* [nmf.py](https://github.com/Saurav0074/100daysOfMLCode/blob/master/blog_topic_modeling/code/nmf.py) uses scikit-learn's implementation of the Non-negative matrix factorization (NMF) alogirthm. 

![Optimal no. of topics](/blog_topic_modeling/outputs/female_blogs.png)


