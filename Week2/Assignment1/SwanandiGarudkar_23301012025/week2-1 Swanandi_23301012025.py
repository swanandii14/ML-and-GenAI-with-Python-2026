#Q1. Load the dataset and display the first five records
import pandas as pd
df = pd.read_csv("Dataset 2.csv")
print(df.head())

# Q2. Determine the number of rows and columns in the dataset
print("Number of rows and columns:")
print(df.shape)


# Q3. Display all column names
print("Column Names:")
print(df.columns)

# Q4. Identify numerical and categorical features
import numpy as np
print("Numerical Features:")
print(df.select_dtypes(include=np.number).columns)
print("\nCategorical Features:")
print(df.select_dtypes(exclude=np.number).columns)

# Q5. Check missing values
print("Missing Values:")
print(df.isnull().sum())

# Q6. Average age of users
print("Average Age:")
print(df['Age'].mean())

# Q7. Average watch hours per week
print("Average Watch Hours Per Week:")
print(df['WatchHoursPerWeek'].mean())

# Q8. Average monthly spending
print("Average Monthly Spending:")
print(df['MonthlySpend'].mean())

# Q9. Count users in each subscription category
print("Subscription Category Counts:")
print(df['SubscriptionType'].value_counts())

# Q10. Percentage of users who renewed subscriptions
renewal_percentage = df['SubscriptionRenewed'].value_counts(normalize=True) * 100
print("Subscription Renewal Percentage:")
print(renewal_percentage)

# Q11. Convert categorical features into numerical form
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['SubscriptionType'] = le.fit_transform(df['SubscriptionType'])
df['FavoriteGenre'] = le.fit_transform(df['FavoriteGenre'])
df['SubscriptionRenewed'] = le.fit_transform(df['SubscriptionRenewed'])
print(df.head())

# Q12. Define X and y
X = df.drop(['UserID', 'SubscriptionRenewed'], axis=1)
y = df['SubscriptionRenewed']
print("Feature Set (X):")
print(X.head())
print("\nTarget Variable (y):")
print(y.head())

# Q13. Split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("Training and Testing data created successfully")

# Q14. Train Decision Tree model
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
print("Decision Tree Model Trained Successfully")

# Q15. Evaluate model accuracy

from sklearn.metrics import accuracy_score
y_pred_dt = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:")
print(accuracy)

# Q16. Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_dt)
print("Confusion Matrix:")
print(cm)

# Q17. Train KNN classifier with K = 5
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
print("KNN Model Trained Successfully")

# Q18. Compare accuracy of KNN and Decision Tree
from sklearn.metrics import accuracy_score
y_pred_knn = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:")
print(knn_accuracy)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:")
print(dt_accuracy)

# Q19. Train Linear Regression model
from sklearn.linear_model import LinearRegression
X_reg = df.drop(['UserID', 'MonthlySpend'], axis=1)
y_reg = df['MonthlySpend']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)
lr_model = LinearRegression()
lr_model.fit(X_train_reg, y_train_reg)
print("Linear Regression Model Trained Successfully")

# Q20. Predict monthly spending for a new user
new_user = [[
    25,   # Age
    1,    # Gender
    2,    # SubscriptionType
    20,   # WatchHoursPerWeek
    3,    # DevicesUsed
    1,    # FavoriteGenre
    15,   # AdClicks
    1     # SubscriptionRenewed
]]
predicted_spend = lr_model.predict(new_user)
print("Predicted Monthly Spending:")
print(predicted_spend[0])

