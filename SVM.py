import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('c:/Users/LAPTOP/Downloads/1) iris.csv')

# Check the data structure
print("Dataset shape:", df.shape)
print("Column names:", df.columns.tolist())
print("First few rows:")
print(df.head())

# Initialize encoders and scalers
le = LabelEncoder()
imputer = SimpleImputer(strategy='mean')  # For missing values
scaler = StandardScaler()  # For scaling features

# Prepare features (X) - first 4 columns are features
X = df.iloc[:, :4].values  # Don't reshape! Keep as (150, 4)

# Prepare target (Y) - assuming 'species' is the target column
# If your target is in column 4 (index 4), use:
if df.shape[1] > 4:  # If there are more than 4 columns
    Y = df.iloc[:, 4]  # 5th column (index 4)
else:  # If species is the last column
    Y = df.iloc[:, -1]  # Last column

# Handle missing values if any
X = imputer.fit_transform(X)

# Scale features (optional but recommended for SVM)
X = scaler.fit_transform(X)

# Encode target labels
Y_encoded = le.fit_transform(Y)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_encoded, test_size=0.2, random_state=20
)

print(f"\nData splits:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"Y_test shape: {Y_test.shape}")

# Create target column in original dataframe for plotting
df['target'] = le.fit_transform(Y)

# Separate by species
df0 = df[df.target == 0]  # First species
df1 = df[df.target == 1]  # Second species
df2 = df[df.target == 2]  # Third species

# Get species names
species_names = le.classes_
print(f"\nSpecies mapping:")
for i, species in enumerate(species_names):
    print(f"{i}: {species}")

# Plot 1: Sepal Length vs Sepal Width
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Sepal Length vs Sepal Width')

plt.scatter(df0['sepal_length'], df0['sepal_width'], 
           color='green', marker='o', label=species_names[0], alpha=0.7)
plt.scatter(df1['sepal_length'], df1['sepal_width'], 
           color='blue', marker='s', label=species_names[1], alpha=0.7)
plt.scatter(df2['sepal_length'], df2['sepal_width'], 
           color='red', marker='^', label=species_names[2], alpha=0.7)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Petal Length vs Petal Width
plt.subplot(1, 2, 2)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Petal Length vs Petal Width')

plt.scatter(df0['petal_length'], df0['petal_width'], 
           color='green', marker='o', label=species_names[0], alpha=0.7)
plt.scatter(df1['petal_length'], df1['petal_width'], 
           color='blue', marker='s', label=species_names[1], alpha=0.7)
plt.scatter(df2['petal_length'], df2['petal_width'], 
           color='red', marker='^', label=species_names[2], alpha=0.7)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional: Plot all feature pairs
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

combinations = [
    ('sepal_length', 'sepal_width'),
    ('sepal_length', 'petal_length'), 
    ('sepal_width', 'petal_length'),
    ('petal_length', 'petal_width')
]

for idx, (feat1, feat2) in enumerate(combinations):
    row, col = idx // 2, idx % 2
    ax = axes[row, col]
    
    ax.scatter(df0[feat1], df0[feat2], color='green', marker='o', 
              label=species_names[0], alpha=0.7)
    ax.scatter(df1[feat1], df1[feat2], color='blue', marker='s', 
              label=species_names[1], alpha=0.7)
    ax.scatter(df2[feat1], df2[feat2], color='red', marker='^', 
              label=species_names[2], alpha=0.7)
    
    ax.set_xlabel(feat1.replace('_', ' ').title())
    ax.set_ylabel(feat2.replace('_', ' ').title())
    ax.set_title(f'{feat1.replace("_", " ").title()} vs {feat2.replace("_", " ").title()}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nData is ready for SVM training!")
print(f"Features shape: {X.shape}")
print(f"Target shape: {Y_encoded.shape}")
print(f"Number of classes: {len(np.unique(Y_encoded))}")

hector = SVC(kernel= 'linear', C = 1)
hector.fit(X_train, Y_train)
print(hector.score(X_train, Y_train))

Y_predict = hector.predict(X_test)
print(classification_report(Y_test, Y_predict))
