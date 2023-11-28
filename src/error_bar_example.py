
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data from the book
df = pd.DataFrame({
    'participant': list(range(1,13)) + list(range(1,13)), 'condition': ['pic']*12 + ['real']*12,
    'anxiety': [30, 35, 45, 40, 50, 35, 55, 25, 30, 45, 40, 50, 40, 35, 50, 55, 65, 55, 50, 35, 30, 50, 60, 39]
})
df

fig, ax = plt.subplots(1,2, figsize=(7,3), dpi=100, sharey=True)

# Plot with unadjusted errorbars
sns.barplot(x='condition', y='anxiety', data=df, ax=ax[0])
ax[0].set(title='Unadjusted CI')

### Applying the adjustment
grand_mean = df['anxiety'].mean()

# Create adjustment factor: (grand_mean - each_subject_mean)
df_adj = df.set_index('participant').join(grand_mean - df.groupby('participant').mean(), rsuffix='_adjustment_factor').reset_index()

# Add adjustment factor to the original values
df_adj['anxiety_adj'] = df_adj['anxiety'] + df_adj['anxiety_adjustment_factor']

#After adjustment all participants have the same mean


# # Plot with adjusted errorbars
sns.barplot(x='condition', y='anxiety_adj', data=df_adj, ax=ax[1])
ax[1].set(title='Adjusted CI')

plt.show()
