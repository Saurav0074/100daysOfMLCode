# #100DaysOfMLCode

## Task 1: Topic modeling the Blog Authorship Corpus

Dependencies: `gensim 3.4.0, scikit-learn 0.19.1, spacy 2.0.11`

The [blog authorship corpus](http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm) consists of the collected posts of 19,320 bloggers gathered from blogger.com in August 2004. The corpus is available in an xml format that contains tags about age and gender of the blog author. 

I parsed out only those spanning over 500 words giving me a total of 2,49,844 male blogs and 2,54,879 female blogs, out of which I chose 50,000 for either of the genders. The code for parsing can be found [here](https://github.com/Saurav0074/Gender-Classification-of-Blog-Author/blob/master/parse_xml.py). This task leverages two algorithms to observe the relevant topics prevailing in the blogs written by both the genders:

* [lda.py](https://github.com/Saurav0074/100daysOfMLCode/blob/master/blog_topic_modeling/code/lda.py) uses the gensim and mallet's implementation of Latent Dirichlet Allocation (LDA) [1] along with finding the optimal number of topics on gensim's classical LDA based upon their coherence scores.
* [nmf.py](https://github.com/Saurav0074/100daysOfMLCode/blob/master/blog_topic_modeling/code/nmf.py) uses the scikit-learn's implementation of Non-negative matrix factorization (NMF) alogirthm [2]. 

Here is a plot showing that the coherence score attains maximum value for 60 topics on the female blog corpus:
![Optimal no. of topics](/blog_topic_modeling/outputs/female_blogs.png)

Below are few interesting results showing top 10 words for each such topic on the blog authorship corpus, others can be found in [blog_topics.txt](https://github.com/Saurav0074/100daysOfMLCode/blob/master/blog_topic_modeling/outputs/blog_topics.txt). The text file also contains the topics extracted from the blog corpus introduced by [3]. However, their smaller corpora (~3000 blogs) do not provide much promising insights.

### 1.  Using LDA: 

The float values indicate the contribution of each word to the overall topic.

**Female blogs:**

`(0,
 '0.016*"love" + 0.014*"know" + 0.013*"life" + 0.011*"would" + 0.011*"time" + '
 '0.010*"make" + 0.009*"feel" + 0.009*"thing" + 0.009*"people" + 0.008*"think"')
 `
 
 `
(1,
 '0.009*"ogtahay" + 0.009*"king_arthur" + 0.008*"fucken_hell" + '
 '0.007*"rosepedal" + 0.005*"ayn" + 0.004*"love_boink" + 0.004*"merlin" + '
 '0.004*"iyo" + 0.004*"yung" + 0.004*"aan"')
`

`(2,
 '0.024*"get" + 0.023*"go" + 0.013*"day" + 0.011*"work" + 0.011*"time" + '
 '0.008*"home" + 0.008*"female" + 0.007*"good" + 0.007*"take" + 0.007*"back"')
`

`(3,
 '0.041*"urllink" + 0.015*"read" + 0.013*"book" + 0.013*"female" + '
 '0.011*"blog" + 0.010*"write" + 0.010*"post" + 0.008*"com" + 0.006*"page" + '
 '0.006*"site"')
`

`(4,
 '0.087*"jumper" + 0.022*"pm" + 0.011*"coolgal" + 0.009*"lethalithuanian" + '
 '0.007*"ritzbtz" + 0.006*"lol" + 0.005*"shev_ev" + 0.005*"yea" + '
 '0.005*"fabityfabfab" + 0.005*"dayna"')
`

**Male blogs**

`(0,
 '0.027*"go" + 0.023*"get" + 0.009*"girl" + 0.009*"eat" + 0.008*"say" + '
 '0.007*"home" + 0.007*"house" + 0.006*"night" + 0.006*"guy" + 0.006*"fuck"')
 `
 
 `(2,
 '0.012*"say" + 0.010*"war" + 0.009*"bush" + 0.009*"american" + '
 '0.007*"urllink" + 0.007*"president" + 0.007*"iraq" + 0.007*"country" + '
 '0.006*"would" + 0.006*"government"')
 `
 
 `(3,
 '0.019*"game" + 0.019*"go" + 0.013*"play" + 0.009*"team" + 0.008*"den" + '
 '0.007*"get" + 0.007*"today" + 0.006*"tat" + 0.006*"male" + 0.006*"match"')
`

`(7,
 '0.013*"nickmac_pm" + 0.010*"ermagetton_pm" + 0.009*"nickmac" + '
 '0.005*"johnnydr" + 0.005*"sbristowsd" + 0.005*"samuraipanda" + 0.005*"ltt" + '
 '0.004*"eriol" + 0.004*"loren" + 0.003*"tnod_etov"')
 `
 
 `(9,
 '0.108*"urllink" + 0.023*"male" + 0.021*"blog" + 0.018*"post" + 0.015*"com" + '
 '0.014*"site" + 0.010*"link" + 0.010*"read" + 0.008*"http_www" + 0.008*"page"')
`
### 2. Using NMF:

**Female blogs**

`Topic 0:
pron hand mother mom family eye sister mind parent name
`

`Topic 1:
urllink female http bring com www link bear site page
`

`Topic 2:
go home today fun back come house mom pron tomorrow
`

`Topic 12:
say ask tell word mean something anything hear nothing give
`

`Topic 13:
night last sleep dream morning wake bed hour week saturday
`

**Male blogs**

`Topic 0:
pron eye mind name hand mom dad parent face brother
`

`Topic 1:
urllink article link male read news check picture find photo
`

`Topic 3:
bush kerry vote president republican election campaign george party administration
`

`Topic 6:
game play team win player score lose ball football hit
`

`Topic 10:
comment full post male urllink leave make thank box add
`

A closer analysis reveals trend somewhat similar to that mentioned in [3], i.e. female topics show more frequent occurrences of emotionally intensive adverbs and ajectives, acronyms, etc. while male topics talk more of politics, games and religion. 

The words in each topic can be analysed individually as well to infer what they are actually taking about. For example, `(game play team win player score lose ball football hit)` clearly implies a football match while `(urllink article link male read news check picture find photo)` could be about a website that posts news.

Further, the topics generated by the NMF model seem to make more sense than that of those from LDA, which seems plausible given the smaller data set size.[4]

Future work: Use `pyLDAvis` to visualize the results on a jupyter notebook; notice the variation of coherence scores on the unseen blogs.

# References
[1] Blei, David M. et al. “Latent Dirichlet Allocation.” Journal of Machine Learning Research 3 (2003): 993-1022.

[2] Lee, Daniel D. and H. Sebastian Seung. “Algorithms for Non-negative Matrix Factorization.” NIPS (2000).

[3] Mukherjee, Arjun and Bing Liu. “Improving Gender Classification of Blog Authors.” EMNLP (2010).

[4] [Topic Modeling with LDA and NMF on the ABC News Headlines dataset](https://medium.com/ml2vec/topic-modeling-is-an-unsupervised-learning-approach-to-clustering-documents-to-discover-topics-fdfbf30e27df)
