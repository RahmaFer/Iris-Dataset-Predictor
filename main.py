# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   import streamlit as st
   from sklearn.datasets import  load_iris
   from sklearn.ensemble import  RandomForestClassifier
   from sklearn.model_selection import train_test_split
   import pandas as pd

   df = load_iris()

   X = df.data
   y = df.target

   x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # splitting data with test size of 30%
   clf = RandomForestClassifier(n_estimators=10)  # Creating a random forest with 10 decision trees
   clf.fit(x_train, y_train)  # Training our model

   st.title("Iris Data Predictor")
   st.header("Random Forest Classifier")

   iris = pd.DataFrame(df.data, columns=df.feature_names,
                    index=pd.Index([i for i in range(df.data.shape[0])])).join(
      pd.DataFrame(df.target, columns=pd.Index(["species"]),
                index=pd.Index([i for i in range(df.target.shape[0])])))


   level0 = st.slider("Sepal length",min(iris["sepal length (cm)"]),max(iris["sepal length (cm)"]),float(iris["sepal length (cm)"].mean()))
   st.text('Selected: {}'.format(level0))

   level1 = st.slider("Sepal width",min(iris["sepal width (cm)"]),max(iris["sepal width (cm)"]),float(iris["sepal width (cm)"].mean()))
   st.text('Selected: {}'.format(level1))

   level2 = st.slider("Petal length",min(iris["petal length (cm)"]),max(iris["petal length (cm)"]),float(iris["petal length (cm)"].mean()))
   st.text('Selected: {}'.format(level2))

   level3 = st.slider("Petal width",min(iris["petal width (cm)"]),max(iris["petal width (cm)"]),float(iris["petal width (cm)"].mean()))
   st.text('Selected: {}'.format(level3))

   parameter_default_values = [level0, level1, level2, level3]

   test=pd.DataFrame([parameter_default_values],columns=['Sepal length (cm)','Sepal Width (cm)','Petal length (cm)','Petal Width (cm)'],dtype=float)


   if (st.button("Prediction")):
     prediction= clf.predict(test)# testing our model
     print(prediction)
     if prediction == 0:
         st.text("Setosa")
     elif prediction == 1:
         st.text("versicolor")
     else:
         st.text("virginica")