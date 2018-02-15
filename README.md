# MusicMagal
## Group recommendation system for music tracks.


## ITC FELLOWS 2017-2018
### PERSONAL CHALLENGE
### DANIEL FRANCH & ALEX INGBERG


MusicMagal is a group recommendation system that recommends n music tracks to a group of m users considering all of the m users preferences into account.
To achieve this we've based our machine learning and deep learning models in Last.Fm data. After computing and when the resulting playlist is output, we create a real playlist using Spotify API's python wrapper: Spotipy.

A typical version of the program's flow is presented in [musicmagal_flow.ipynb](musicmagal_flow.ipynb).

To see our evaluation metrics you can check [musicmagal_evaluation.ipynb](musicmagal_evaluation.ipynb).

Our database exploration is presented in [db_exploration.ipynb](db_exploration.ipynb).

You can read the article going over our model bit by bit in [my Medium account](https://medium.com/p/c93e9dabd01a/edit).



### References:

[1] Mezei, Zsolt, and Carsten Eickhoff. ["Evaluating Music Recommender Systems for Groups."](papers/Evaluating\ Music\ Recommender\ Systems\ for\ Groups.pdf) arXiv preprint arXiv:1707.09790 (2017).

[2] Yoshii, Kazuyoshi, et al. ["Hybrid Collaborative and Content-based Music Recommendation Using Probabilistic Model with Latent User Preferences."](papers/Hybrid\ Collaborative\ and\ Content-based\ Music\ Recommendation.pdf) ISMIR. Vol. 6. 2006.

[3] Parra, Denis, et al. ["Implicit feedback recommendation via implicit-to-explicit ordinal logistic regression mapping."](papers/Implicit\ Feedback\ Recommendation.pdf) Proceedings of the CARS-2011 (2011).

[4] Hu, Yifan, Yehuda Koren, and Chris Volinsky. ["Collaborative filtering for implicit feedback datasets."](papers/cf-hkmethod.pdf) Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on. Ieee, 2008.

[5] Barkan, Oren, and Noam Koenigstein. ["Item2vec: neural item embedding for collaborative filtering."](papers/item2vec.pdf) Machine Learning for Signal Processing (MLSP), 2016 IEEE 26th International Workshop on. IEEE, 2016.

[6] Leskovec, Jure, Anand Rajaraman, and Jeffrey David Ullman. ['Mining of massive datasets.'](papers/Recommendation\ Systems.pdf) Cambridge university press, 2014.
