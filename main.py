from PlotAdaBoost import plot_staged_adaboost
from Dataset import make_toy_dataset
# from sklearn.ensemble import AdaBoostClassifier
from AdaBoostClassifier import AdaBoost

# AdaBoost.fit = fit
# AdaBoost.predict = predict

X, y = make_toy_dataset(n=10, random_seed=10)

clf = AdaBoost().fit(X, y, iters=10)
plot_staged_adaboost(X, y, clf)

train_err = (clf.predict(X) != y).mean()
print(f'Train error: {train_err:.1%}')
