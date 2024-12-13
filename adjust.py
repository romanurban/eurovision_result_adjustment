import pandas as pd

file_path = 'dataset/eurovision.csv'
df = pd.read_csv(file_path)

# Remove duplicates
df = df[df['Duplicate'] != 'x']
# Take into account finals only
df = df[df['(semi-) final'] == 'f']

# Now drop unnecessary columns
df = df.drop(columns=['Duplicate', '(semi-) final', 'Edition'])

# Display the first few rows of the DataFrame
print(df.head())

# Print the number of rows and columns
print(f'Total number of rows: {df.shape[0]}')
print(f'Total number of columns: {df.shape[1]}')

# Group by 'Year' and 'To country', then sum the points
grouped = df.groupby(['Year', 'To country'])['Points'].sum().reset_index()

# Sort by 'Year' and 'Points' in descending order
grouped = grouped.sort_values(by=['Year', 'Points'], ascending=[True, False])

# Get the top 3 countries for each year
top3_per_year = grouped.groupby('Year').head(3)

# Count the frequency of each country appearing in the top 3
top3_counts = top3_per_year['To country'].value_counts().reset_index()
top3_counts.columns = ['Country', 'Top3_Count']

# Get the top 10 countries with the highest frequency
top10_countries = top3_counts.head(10)

# Display the top 10 countries
print(top10_countries)

# Find the country with the maximum points for each year
winners = grouped.loc[grouped.groupby('Year')['Points'].idxmax()]

# Display the winners
print(winners)

# Find potential vote abusers
high_scores = df[df['Points'] >= 8]
abusers = high_scores.groupby(['From country', 'To country']).size().reset_index(name='Count')
abusers = abusers[abusers['Count'] > 5]  # Adjust the threshold as needed

# Sort the results by count in descending order
abusers = abusers.sort_values(by='Count', ascending=False)

# Set pandas option to display all rows without truncation
pd.set_option('display.max_rows', None)

# Display potential vote abusers
print(abusers)

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Izveido balsu matricu
vote_matrix = df.pivot_table(index='From country', columns='To country', values='Points', aggfunc='sum', fill_value=0)

# Normalizē datus
scaler = MinMaxScaler()
normalized_matrix = scaler.fit_transform(vote_matrix)

# Determine the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(normalized_matrix)
    wcss.append(kmeans.inertia_)

# Plot the Elbow graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Piemēro K-Means klasterēšanu
kmeans = KMeans(n_clusters=7, random_state=42) 
vote_matrix['Cluster'] = kmeans.fit_predict(normalized_matrix)

# Datu vizualizācija
plt.figure(figsize=(12, 8))
sns.heatmap(vote_matrix.iloc[:, :-1], cmap='coolwarm', cbar=True)
plt.title('Balsošanas punktu siltumkarte')
plt.show()

# Parāda klasteru rezultātus
cluster_results = vote_matrix['Cluster'].reset_index()
print(cluster_results.sort_values('Cluster'))

# Analizē katru klasteri
for cluster in range(7):
    members = cluster_results[cluster_results['Cluster'] == cluster]['From country']
    print(f"Klasteris {cluster}: {', '.join(members)}")


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Veic PCA, lai samazinātu dimensiju skaitu uz 2
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(normalized_matrix)

# Izveido PCA diagrammu ar klasteriem
plt.figure(figsize=(10, 6))
for cluster in range(kmeans.n_clusters):
    cluster_points = reduced_data[vote_matrix['Cluster'] == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Klasteris {cluster}')

plt.title('Klasteru vizualizācija ar PCA')
plt.xlabel('Pirma galvenā komponenta')
plt.ylabel('Otrā galvenā komponenta')
plt.legend()
plt.grid()
plt.show()


# Apply a scaling factor to the points given by abusers
scaling_factor = 0.5  # Adjust the scaling factor as needed
for index, row in abusers.iterrows():
    from_country = row['From country']
    to_country = row['To country']
    df.loc[(df['From country'] == from_country) & (df['To country'] == to_country), 'Points'] *= scaling_factor

# Recalculate the winners after softening the impact of vote abusers
grouped = df.groupby(['Year', 'To country'])['Points'].sum().reset_index()
winners = grouped.loc[grouped.groupby('Year')['Points'].idxmax()]

# Display the recalculated winners
print(winners)

# Sort by 'Year' and 'Points' in descending order
grouped = grouped.sort_values(by=['Year', 'Points'], ascending=[True, False])

# Get the top 3 countries for each year
top3_per_year = grouped.groupby('Year').head(3)

# Count the frequency of each country appearing in the top 3
top3_counts = top3_per_year['To country'].value_counts().reset_index()
top3_counts.columns = ['Country', 'Top3_Count']

# Get the top 10 countries with the highest frequency
top10_countries = top3_counts.head(10)

# Display the top 10 countries
print(top10_countries)

# Eliminate abusers
for index, row in abusers.iterrows():
    from_country = row['From country']
    to_country = row['To country']
    df = df[~((df['From country'] == from_country) & (df['To country'] == to_country))]

# Recalculate the winners after eliminating abusers
grouped = df.groupby(['Year', 'To country'])['Points'].sum().reset_index()
winners = grouped.loc[grouped.groupby('Year')['Points'].idxmax()]

# Display the recalculated winners after eliminating abusers
print(winners)

# Sort by 'Year' and 'Points' in descending order
grouped = grouped.sort_values(by=['Year', 'Points'], ascending=[True, False])

# Get the top 3 countries for each year
top3_per_year = grouped.groupby('Year').head(3)

# Count the frequency of each country appearing in the top 3
top3_counts = top3_per_year['To country'].value_counts().reset_index()
top3_counts.columns = ['Country', 'Top3_Count']

# Get the top 10 countries with the highest frequency
top10_countries = top3_counts.head(10)

# Display the top 10 countries
print(top10_countries)