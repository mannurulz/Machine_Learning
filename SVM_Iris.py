from sklearn import svm
model = svm.SVC(kernel='linear',c=1,gamma=1)
model.fit(X, y)
model.score(X, y)
predicted = model.predict(x_test)
