import pandas as pd

# Load the first dataset
file_path1 = "master_hard.abnv"
df_hard = pd.read_csv(file_path1, header=0)

# Add a new column to indicate the source dataset
df_hard['Dataset'] = 'Hard'

# Load the second dataset
file_path2 = "master_BIB.abnv"
df_bib = pd.read_csv(file_path2, header=0)

# Add a new column to indicate the source dataset
df_bib['Dataset'] = 'BIB'

# Concatenate the two dataframes
combined_df = pd.concat([df_hard, df_bib], ignore_index=True)

# Print the combined dataframe
print(combined_df)
# Filter rows with CollectionID ECBC
df_ecbc = combined_df[combined_df['CollectionID'] == 'ECBC']

# Print the new dataframe
print(df_ecbc)
#count number of Hard and BIB datasets
print(df_ecbc['Dataset'].value_counts())

#make a new dataframe by selecting 500,000 BIB rows and 500,000 Hard rows at random
df_sample = df_ecbc.groupby('Dataset').apply(lambda x: x.sample(n=500000, random_state=42)).reset_index(drop=True)

# Print the new dataframe
print(df_sample)
#exclude outlier values of Energy column for BIB and Hard datasets using Z-Score Method
from scipy import stats

# Calculate the z-scores for the Energy column in the BIB and Hard datasets
df_sample['ZScore'] = df_sample.groupby('Dataset')['Energy'].transform(lambda x: stats.zscore(x))

# Filter out rows with z-scores greater than 3 or less than -3
df_sample = df_sample[(df_sample['ZScore'] < 2) & (df_sample['ZScore'] > -2)]

# Print the new dataframe
print(df_sample)

#plot Energy distribution for BIB and Hard datasets using a scatter plot for df_sample
import matplotlib.pyplot as plt

# Create a figure with two subplots





#plot the side by side graph of PPDG vs energy for BIB and Hard datasets for each PPDG
#exclude PPDGs that are not present in both datasets
import matplotlib.pyplot as plt

# Filter out PPDGs that are not present in both datasets
common_ppdgs = set(df_sample[df_sample['Dataset'] == 'Hard']['PPDG']).intersection(set(df_sample[df_sample['Dataset'] == 'BIB']['PPDG']))

# Plot the side-by-side graphs, instead of dots, use some type of heatmap to show BIB and Hard values vs PPDG
import seaborn as sns

# for ppdg in common_ppdgs:
#     fig, ax = plt.subplots()

#     # Filter data for the specific PPDG and dataset
#     data_hard = df_sample[(df_sample['PPDG'] == ppdg) & (df_sample['Dataset'] == 'Hard')]
#     data_bib = df_sample[(df_sample['PPDG'] == ppdg) & (df_sample['Dataset'] == 'BIB')]

#     # Combine data for violin plot
#     data_combined = pd.concat([data_hard, data_bib])

#     # Plot violin plot
#     sns.violinplot(x='Dataset', y='Energy', data=data_combined, ax=ax)

#     # Set title
#     plt.title(f'PPDG {ppdg} Energy Distribution by Dataset')

#     # Show plot
#     plt.show()







import matplotlib.pyplot as plt

# Set the transparency level
alpha = 0.1

# Define the offset
offset = 1  # Adjust this value based on your preference

# Iterate over common_ppdgs
import matplotlib.pyplot as plt

# Set the transparency level
alpha = 0.1

# Define the offset
offset = 0.1  # Adjust this value based on your preference

# Iterate over common_ppdgs
# for ppdg in common_ppdgs:
#     fig, ax = plt.subplots()

#     # Filter data for the specific PPDG and dataset
#     data_hard = df_sample[(df_sample['PPDG'] == ppdg) & (df_sample['Dataset'] == 'Hard')]
#     data_bib = df_sample[(df_sample['PPDG'] == ppdg) & (df_sample['Dataset'] == 'BIB')]

#     # Plot data points with transparency and offset
#     ax.plot(data_hard['Energy'], data_hard['PPDG'], 'o', color='red', alpha=alpha, label='Hard')
#     ax.plot(data_bib['Energy'], data_bib['PPDG'] + offset, 'o', color='blue', alpha=alpha, label='BIB')

#     # Set labels and title
#     ax.set_xlabel('Energy')
#     ax.set_ylabel('PPDG')
#     plt.title(f'PPDG {ppdg} Energy vs PPDG')
#     perc_bib = len(data_bib) / len(data_hard) * 100
#     #label number of bib and hard points used in the plot, in brackets, show percentage, keeping hard points as 100%
#     plt.text(0.5, 0.5, f'BIB: {len(data_bib)} ({perc_bib:.2f}%)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
#     plt.text(0.5, 0.45, f'Hard: {len(data_hard)} (100%)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

#    # Set y-axis limits
#     min_ppdg = min(data_hard['PPDG'].min(), data_bib['PPDG'].min() + offset)
#     max_ppdg = max(data_hard['PPDG'].max(), data_bib['PPDG'].max() + offset)
#     ax.set_ylim(min_ppdg - 0.1, max_ppdg + 0.1)  # Adjust the padding as needed

#     # Set y-axis tick labels
#     ax.set_yticks(ax.get_yticks())
#     ax.set_yticklabels([ppdg] * len(ax.get_yticks()))  # Set all y-axis tick labels to the same PPDG

#     # Add legend
#     plt.legend()

#     # Show plot
#     #plt.show()


#take 1000 random samples each from BIB and Hard rows of df_sample and make a new dataframe

# #train a ML model to classify BIB and Hard datasets based on PPDG,Energy ,    Time,  Length ,  X    ,   Y   ,    Z columns
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # Define the features and target variable
# X = df_sample_subset[['PPDG', 'Energy', 'Time', 'Length', 'X', 'Y', 'Z']]
# y = df_sample_subset['Dataset']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Test all classifiers
# #import the classifiers
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# #more
# from sklearn.metrics import accuracy_score

# # Initialize the classifiers
# rf = RandomForestClassifier(random_state=42)
# svc = SVC(random_state=42)
# lr = LogisticRegression(random_state=42)
# knn = KNeighborsClassifier()
# nb = GaussianNB()
# dt = DecisionTreeClassifier(random_state=42)

# from sklearn.metrics import f1_score
# from sklearn.metrics import confusion_matrix
# # Create a list of classifiers
# classifiers = [rf, svc, lr, knn, nb, dt]
# best_accuracy=0
# # Train and evaluate each classifier
# for clf in classifiers:
#     # Train the classifier
#     clf.fit(X_train, y_train)

#     # Make predictions
#     y_pred = clf.predict(X_test)

#     # Calculate the accuracy
#     accuracy = accuracy_score(y_test, y_pred)

#     # Print the classifier and accuracy
#     print(f'{clf.__class__.__name__} Accuracy: {accuracy:.2f}')
#     #save the best classifier
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         best_classifier = clf

#     #print confusion matrix for the best classifier
#     y_pred = best_classifier.predict(X_test)
#     cm = confusion_matrix(y_test, y_pred)
#     print(f'Confusion Matrix:\n{cm}')


import numpy as np
#train a Deep Learning model to classify BIB and Hard datasets based on PPDG,Energy ,    Time,  Length ,  X    ,   Y   ,    Z columns
#use popular neural network architectures like CNN, RNN, LSTM, etc. Use tensorflow
#use dataframe df_sample_subset


#generate a BIB dataframe with 500,000 rows that has random values for PPDG, Energy, Time, Length, X, Y, Z columns, the values should be within the range of the original dataset

# Generate random values for the BIB datasetimport numpy as np

np.random.seed(42)

# Generate random samples for PPDG
bib_ppdg = np.random.choice(df_sample['PPDG'], 500000)

# Define parameters for energy distribution
mean_energy = 0.0000392  # Mean energy value
energy_std = 0.0001767    # Standard deviation for energy

# Define parameters for time distribution
mean_time = 651.30976     # Mean time value
time_std = 1648.124      # Standard deviation for time

# Generate random samples for energy and time
num_samples = 500000
bib_energy = []
bib_time = []

# Generate samples until desired number is reached
while len(bib_energy) < num_samples:
    energy_sample = np.random.normal(mean_energy, energy_std)
    time_sample = np.random.normal(mean_time, time_std)
    if energy_sample > 0 and time_sample > 0:
        bib_energy.append(energy_sample)
        bib_time.append(time_sample)

# Convert lists to arrays
bib_energy = np.array(bib_energy)
bib_time = np.array(bib_time)
import numpy as np

# Assuming bib_time is your existing array
indices = np.random.choice(len(bib_time), 50000)
bib_time[indices] += 4000



bib_length = np.random.uniform(df_sample['Length'].min(), df_sample['Length'].max(), 500000)
bib_x = np.random.uniform(df_sample['X'].min(), df_sample['X'].max(), 500000)
bib_y = np.random.uniform(df_sample['Y'].min(), df_sample['Y'].max(), 500000)
bib_z = np.random.uniform(df_sample['Z'].min(), df_sample['Z'].max(), 500000)

# Create a new BIB dataframe
df_bib_new = pd.DataFrame({
    'PPDG': bib_ppdg,
    'Energy': bib_energy,
    'Time': bib_time,
    'Length': bib_length,
    'X': bib_x,
    'Y': bib_y,
    'Z': bib_z,
    'Dataset': 'BIB'
})

#add Hard rows to the new BIB dataset from df_sample, take same values for hard rows as in the original dataset

# Select 500,000 random Hard rows from the df_sample dataset
df_hard_sample = df_sample[df_sample['Dataset'] == 'Hard'].sample(n=400000, random_state=42)

# Concatenate the BIB and Hard dataframes
df_combined_new = pd.concat([df_bib_new, df_hard_sample])

# Print the new combined dataframe
print(df_combined_new)
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten,Conv2D, MaxPooling2D, Flatten, Dense, Dropout,Embedding,LSTM
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

df_sample_subset = df_combined_new.groupby('Dataset').apply(lambda x: x.sample(n=400000, random_state=42)).reset_index(drop=True)

# Define the features and target variable
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from keras.models import load_model

# Load the saved best model from the file
best_model = load_model('best_model.h5')
encoder = LabelEncoder()
# y_encoded = encoder.fit_transform(y)
# Preprocess the entire dataset
X = df_sample_subset[['Energy', 'Time', 'Length', 'X', 'Y', 'Z']]
y = encoder.fit_transform(df_sample_subset['Dataset'])

# Make predictions on the entire dataset
predictions = best_model.predict(X)

# Convert the predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Evaluate the performance of the model on the entire dataset
test_loss, test_accuracy = best_model.evaluate(X, y, verbose=1)
print(f'Test Accuracy on the entire dataset: {test_accuracy}')

# Define batch size for evaluation
batch_size = 25000

# Calculate the number of batches
num_batches = len(X) // batch_size
if len(X) % batch_size != 0:
    num_batches += 1

total_loss = 0
total_accuracy = 0

# Evaluate the model batch by batch
for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(X))
    X_batch = X[start_idx:end_idx]
    y_batch = y[start_idx:end_idx]
    loss, accuracy = best_model.evaluate(X_batch, y_batch, verbose=0)
    total_loss += loss * len(X_batch)
    total_accuracy += accuracy * len(X_batch)

# Calculate average loss and accuracy
average_loss = total_loss / len(X)
average_accuracy = total_accuracy / len(X)

print(f'Average Loss on the entire dataset: {average_loss}')
print(f'Average Accuracy on the entire dataset: {average_accuracy}')

from keras.utils import plot_model

# Plot the architecture of the best model
plot_model(best_model, to_file='best_model_architecture.png', show_shapes=True, show_layer_names=True)
