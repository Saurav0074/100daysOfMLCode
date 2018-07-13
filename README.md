# #100DaysOfMLCode

## Task 1: Topic modeling the Blog Authorship Corpus

The [blog authorship corpus](http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm) consists of the collected posts of 19,320 bloggers gathered from blogger.com in August 2004. The corpus is available in an xml format that contains tags about age and gender of the blog author. 

I parsed out only those spanning over 500 words giving me a total of 2,49,844 male blogs and 2,54,879 female blogs, out of which I chose 50,000 for either of the genders. This task leverages two algorithms to observe the relevant topics prevailing in the blogs of both the genders:

* [lda.py](https://github.com/Saurav0074/100daysOfMLCode/blob/master/blog_topic_modeling/code/lda.py) uses the gensim and mallet's implementation of Latent Dirichlet Allocation (LDA) along with the code for the optimal number of topics on gensim's classical LDA based upon their coherence scores.
* [nmf.py](https://github.com/Saurav0074/100daysOfMLCode/blob/master/blog_topic_modeling/code/nmf.py) uses scikit-learn's implementation of the Non-negative matrix factorization (NMF) alogirthm. 

Here is a plot showing that the coherence score is maximum for 60 topics on the female blog corpus:
![Optimal no. of topics](/blog_topic_modeling/outputs/female_blogs.png)

Here are the top 5 topics for either of the genders:

### 1.  Using LDA:
**Female blogs:**

`(0,
 '0.016*"love" + 0.014*"know" + 0.013*"life" + 0.011*"would" + 0.011*"time" + '
 '0.010*"make" + 0.009*"feel" + 0.009*"thing" + 0.009*"people" + 0.008*"think"')
(1,
 '0.009*"ogtahay" + 0.009*"king_arthur" + 0.008*"fucken_hell" + '
 '0.007*"rosepedal" + 0.005*"ayn" + 0.004*"love_boink" + 0.004*"merlin" + '
 '0.004*"iyo" + 0.004*"yung" + 0.004*"aan"')
(2,
 '0.024*"get" + 0.023*"go" + 0.013*"day" + 0.011*"work" + 0.011*"time" + '
 '0.008*"home" + 0.008*"female" + 0.007*"good" + 0.007*"take" + 0.007*"back"')
(3,
 '0.041*"urllink" + 0.015*"read" + 0.013*"book" + 0.013*"female" + '
 '0.011*"blog" + 0.010*"write" + 0.010*"post" + 0.008*"com" + 0.006*"page" + '
 '0.006*"site"')
(4,
 '0.087*"jumper" + 0.022*"pm" + 0.011*"coolgal" + 0.009*"lethalithuanian" + '
 '0.007*"ritzbtz" + 0.006*"lol" + 0.005*"shev_ev" + 0.005*"yea" + '
 '0.005*"fabityfabfab" + 0.005*"dayna"')
`
