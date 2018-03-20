# Kaggle submissions

Some entries to Kaggle competitions

---

These aren't the most scintillating entries, but are simple applications of a powerful machine learning technique, namely, a Random Forest.
It is quite surprising the predictive power and flexibility this algorithm has, for a range of problems. There is also some degree of interpretablility, 
which is extremely useful (especially in this era of Deep Learning, which is significantly more opaque in terms of internal workings and feature selsction).

The first application here is to the famous MNIST handwritten digit set:

### [Digit Recognizer - Learn computer vision fundamentals with the famous MNIST data](https://www.kaggle.com/c/digit-recognizer)

Here, I use a Random Forest model to get a not too shabby accuracy score of 0.96585. However, ensemble methods would have given slightly more accuracy, although it would have taken a lot of tweaking. In my spare time...


### [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

This is a nice dataset. It's slightly more realistic, as there are a number of missing data points that need to be handled carefully. I made use of the Python library, Pandas, to perform inbuilt interpolation of missing values. This isn't very optimal, and likely the reason my submission score here was a not-all-that-impressive 0.73205. Ensemble methods would do well here, and/or some crafty feature engineering, after a bit of exploratory data analysis to see _how_ the data is missing (_e.g._ what biases exist).

---

Hopefully more to come in the future!
