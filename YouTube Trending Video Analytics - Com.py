# YouTube Trending Video Analytics - Complete Code Block

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import sqlite3
import numpy as np
import json

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load datasets
df_us = pd.read_csv("data/USvideos.csv")
df_in = pd.read_csv("data/INvideos.csv")

# Add country column
df_us["country"] = "US"
df_in["country"] = "IN"

# Merge data
df = pd.concat([df_us, df_in])

# Convert date fields
df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')
df['publish_time'] = pd.to_datetime(df['publish_time'])
# Ensure both are timezone-naive
df['trending_date'] = df['trending_date'].dt.tz_localize(None)
df['publish_time'] = df['publish_time'].dt.tz_localize(None)

# Extract time-based features
df['publish_hour'] = df['publish_time'].dt.hour
df['days_to_trend'] = (df['trending_date'] - df['publish_time'].dt.normalize()).dt.days
df['publish_day'] = df['publish_time'].dt.day_name()

# Drop duplicate trending records
df.drop_duplicates(subset=["video_id", "trending_date"], inplace=True)

# Sentiment analysis on title
df['title_sentiment'] = df['title'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df['sentiment_label'] = df['title_sentiment'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')

# Calculate engagement metrics
df['engagement_rate'] = (df['likes'] + df['dislikes'] + df['comment_count']) / df['views'] * 100

# Load category mapping for US (can be reused for IN as structure is the same)
with open('data/US_category_id.json', 'r') as f:
    cat_data = json.load(f)
cat_map = {str(item['id']): item['snippet']['title'] for item in cat_data['items']}
df['category'] = df['category_id'].astype(str).map(cat_map)

# Save to SQLite for SQL queries
conn = sqlite3.connect("youtube_trending.db")
df.to_sql("videos", conn, if_exists="replace", index=False)

# Create a figure with multiple subplots
plt.figure(figsize=(20, 15))

# 1. Sentiment vs Views
plt.subplot(2, 2, 1)
sns.boxplot(x='sentiment_label', y='views', data=df)
plt.yscale('log')
plt.title("Sentiment vs Views")
plt.xlabel("Sentiment")
plt.ylabel("Views (log scale)")

# 2. Category-wise Analysis
plt.subplot(2, 2, 2)
category_views = df.groupby('category_id')['views'].mean().sort_values(ascending=False)
sns.barplot(x=category_views.index, y=category_views.values)
plt.title("Average Views by Category")
plt.xticks(rotation=45)
plt.ylabel("Average Views")

# 3. Time-based Analysis
plt.subplot(2, 2, 3)
hourly_views = df.groupby('publish_hour')['views'].mean()
sns.lineplot(x=hourly_views.index, y=hourly_views.values)
plt.title("Average Views by Hour of Publication")
plt.xlabel("Hour of Day")
plt.ylabel("Average Views")

# 4. Engagement Analysis
plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='views', y='engagement_rate', alpha=0.5)
plt.xscale('log')
plt.title("Views vs Engagement Rate")
plt.xlabel("Views (log scale)")
plt.ylabel("Engagement Rate (%)")

plt.tight_layout()
plt.savefig('visuals/youtube_analysis.png')
plt.show()

# Print some interesting statistics
print("\n=== YouTube Trending Video Analysis Results ===")
print(f"\nTotal number of videos analyzed: {len(df)}")
print(f"\nTop 5 Categories by Average Views:")
print(df.groupby('category_id')['views'].mean().sort_values(ascending=False).head())
print(f"\nAverage Engagement Rate: {df['engagement_rate'].mean():.2f}%")
print(f"\nMost Common Publishing Days:")
print(df['publish_day'].value_counts().head())
print(f"\nAverage Days to Trend: {df['days_to_trend'].mean():.2f} days")

# Country comparison: Average views and engagement by country and category
country_cat_summary = df.groupby(['country', 'category']).agg(
    avg_views=('views', 'mean'),
    avg_engagement=('engagement_rate', 'mean'),
    count=('video_id', 'count')
).reset_index()

# Visualization: Country comparison by category
plt.figure(figsize=(14, 7))
sns.barplot(x='category', y='avg_views', hue='country', data=country_cat_summary)
plt.xticks(rotation=45, ha='right')
plt.title('Average Views by Category and Country')
plt.ylabel('Average Views')
plt.tight_layout()
plt.savefig('visuals/country_category_comparison.png')
plt.show()

# Export summary tables to CSV
country_cat_summary.to_csv('outputs/country_category_summary.csv', index=False)
df.groupby('category')['views'].mean().reset_index().to_csv('outputs/category_avg_views.csv', index=False)
df.groupby('channel_title').size().reset_index(name='trend_count').sort_values('trend_count', ascending=False).to_csv('outputs/top_trending_channels.csv', index=False)

# Correlation Analysis
numeric_cols = ['views', 'likes', 'dislikes', 'comment_count', 'engagement_rate', 'days_to_trend']
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Key Metrics')
plt.tight_layout()
plt.savefig('visuals/correlation_heatmap.png')
plt.show()

# Export correlation matrix to CSV
correlation_matrix.to_csv('outputs/correlation_matrix.csv')

# Time-based Analysis
plt.figure(figsize=(15, 10))

# Hourly distribution of views
plt.subplot(2, 2, 1)
hourly_views = df.groupby('publish_hour')['views'].mean()
sns.lineplot(x=hourly_views.index, y=hourly_views.values, marker='o')
plt.title('Average Views by Hour of Publication')
plt.xlabel('Hour of Day')
plt.ylabel('Average Views')

# Daily distribution of views
plt.subplot(2, 2, 2)
daily_views = df.groupby('publish_day')['views'].mean()
sns.barplot(x=daily_views.index, y=daily_views.values)
plt.title('Average Views by Day of Week')
plt.xticks(rotation=45)
plt.ylabel('Average Views')

# Sentiment Analysis by Category
plt.subplot(2, 2, 3)
sentiment_by_category = df.groupby('category')['title_sentiment'].mean().sort_values(ascending=False)
sns.barplot(x=sentiment_by_category.index, y=sentiment_by_category.values)
plt.title('Average Sentiment by Category')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Average Sentiment Score')

# Trend Duration Analysis
plt.subplot(2, 2, 4)
sns.histplot(data=df, x='days_to_trend', bins=30)
plt.title('Distribution of Days to Trend')
plt.xlabel('Days to Trend')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('visuals/detailed_analysis.png')
plt.show()

# Channel Performance Analysis
channel_metrics = df.groupby('channel_title').agg({
    'views': ['mean', 'sum', 'count'],
    'likes': 'mean',
    'comment_count': 'mean',
    'engagement_rate': 'mean'
}).round(2)

channel_metrics.columns = ['avg_views', 'total_views', 'video_count', 'avg_likes', 'avg_comments', 'avg_engagement']
channel_metrics = channel_metrics.sort_values('total_views', ascending=False)

# Export channel metrics
channel_metrics.to_csv('outputs/channel_performance.csv')

# Print summary statistics
print("\n=== Detailed Analysis Results ===")
print("\nTop 5 Categories by Sentiment:")
print(sentiment_by_category.head())
print("\nBest Publishing Hours (Top 3):")
print(hourly_views.nlargest(3))
print("\nBest Publishing Days (Top 3):")
print(daily_views.nlargest(3))
print("\nChannel Performance Summary:")
print(channel_metrics.head())

# Country-specific Time Analysis
plt.figure(figsize=(20, 15))

# 1. Country-specific Hourly Patterns
plt.subplot(3, 2, 1)
for country in ['US', 'IN']:
    country_hourly = df[df['country'] == country].groupby('publish_hour')['views'].mean()
    sns.lineplot(x=country_hourly.index, y=country_hourly.values, label=country, marker='o')
plt.title('Average Views by Hour - Country Comparison')
plt.xlabel('Hour of Day')
plt.ylabel('Average Views')
plt.legend()

# 2. Country-specific Daily Patterns
plt.subplot(3, 2, 2)
country_daily = df.pivot_table(
    values='views',
    index='publish_day',
    columns='country',
    aggfunc='mean'
)
country_daily.plot(kind='bar', ax=plt.gca())
plt.title('Average Views by Day - Country Comparison')
plt.xlabel('Day of Week')
plt.ylabel('Average Views')
plt.xticks(rotation=45)
plt.legend(title='Country')

# 3. Category Performance by Country
plt.subplot(3, 2, 3)
category_country = df.pivot_table(
    values='views',
    index='category',
    columns='country',
    aggfunc='mean'
).sort_values('US', ascending=False)
category_country.plot(kind='bar', ax=plt.gca())
plt.title('Category Performance by Country')
plt.xlabel('Category')
plt.ylabel('Average Views')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Country')

# 4. Engagement Analysis
plt.subplot(3, 2, 4)
engagement_by_category = df.groupby(['category', 'country'])['engagement_rate'].mean().unstack()
engagement_by_category.plot(kind='bar', ax=plt.gca())
plt.title('Engagement Rate by Category and Country')
plt.xlabel('Category')
plt.ylabel('Engagement Rate (%)')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Country')

# 5. Trend Duration by Country
plt.subplot(3, 2, 5)
sns.boxplot(x='country', y='days_to_trend', data=df)
plt.title('Trend Duration by Country')
plt.xlabel('Country')
plt.ylabel('Days to Trend')

# 6. Sentiment Distribution by Country
plt.subplot(3, 2, 6)
sns.boxplot(x='country', y='title_sentiment', data=df)
plt.title('Title Sentiment by Country')
plt.xlabel('Country')
plt.ylabel('Sentiment Score')

plt.tight_layout()
plt.savefig('visuals/country_comparison_analysis.png')
plt.show()

# Detailed Category Analysis
category_metrics = df.groupby(['category', 'country']).agg({
    'views': ['mean', 'median', 'std'],
    'likes': 'mean',
    'comment_count': 'mean',
    'engagement_rate': 'mean',
    'days_to_trend': 'mean'
}).round(2)

category_metrics.columns = ['avg_views', 'median_views', 'std_views', 'avg_likes', 
                          'avg_comments', 'avg_engagement', 'avg_days_to_trend']
category_metrics.to_csv('outputs/detailed_category_metrics.csv')

# Print detailed analysis results
print("\n=== Country-Specific Analysis Results ===")
print("\nBest Publishing Hours by Country:")
for country in ['US', 'IN']:
    country_hours = df[df['country'] == country].groupby('publish_hour')['views'].mean()
    print(f"\n{country}:")
    print(country_hours.nlargest(3))

print("\nCategory Performance by Country:")
print(category_country)

print("\nEngagement Analysis by Country:")
print(engagement_by_category)

print("\nDetailed Category Metrics:")
print(category_metrics)

# Prepare Power BI Datasets
print("\nPreparing Power BI datasets...")

# 1. Main Dataset with all metrics
powerbi_main = df.copy()
powerbi_main['publish_date'] = powerbi_main['publish_time'].dt.date
powerbi_main['trending_date'] = powerbi_main['trending_date'].dt.date
powerbi_main['publish_month'] = powerbi_main['publish_time'].dt.month
powerbi_main['publish_year'] = powerbi_main['publish_time'].dt.year

# Calculate additional metrics
powerbi_main['like_ratio'] = powerbi_main['likes'] / powerbi_main['views']
powerbi_main['comment_ratio'] = powerbi_main['comment_count'] / powerbi_main['views']
powerbi_main['engagement_score'] = (powerbi_main['likes'] + powerbi_main['dislikes'] + powerbi_main['comment_count']) / powerbi_main['views']

# 2. Time-based Analysis Dataset
time_analysis = df.groupby(['country', 'publish_hour', 'publish_day']).agg({
    'views': ['mean', 'sum', 'count'],
    'likes': 'mean',
    'comment_count': 'mean',
    'engagement_rate': 'mean'
}).reset_index()
time_analysis.columns = ['country', 'hour', 'day', 'avg_views', 'total_views', 'video_count', 
                        'avg_likes', 'avg_comments', 'avg_engagement']

# 3. Category Performance Dataset
category_performance = df.groupby(['category', 'country']).agg({
    'views': ['mean', 'sum', 'count'],
    'likes': 'mean',
    'comment_count': 'mean',
    'engagement_rate': 'mean',
    'days_to_trend': 'mean',
    'title_sentiment': 'mean'
}).reset_index()
category_performance.columns = ['category', 'country', 'avg_views', 'total_views', 'video_count',
                              'avg_likes', 'avg_comments', 'avg_engagement', 'avg_days_to_trend',
                              'avg_sentiment']

# 4. Channel Performance Dataset
channel_performance = df.groupby(['channel_title', 'country', 'category']).agg({
    'views': ['mean', 'sum', 'count'],
    'likes': 'mean',
    'comment_count': 'mean',
    'engagement_rate': 'mean',
    'days_to_trend': 'mean'
}).reset_index()
channel_performance.columns = ['channel', 'country', 'category', 'avg_views', 'total_views', 
                             'video_count', 'avg_likes', 'avg_comments', 'avg_engagement',
                             'avg_days_to_trend']

# Export datasets for Power BI
powerbi_main.to_csv('powerbi/main_dataset.csv', index=False)
time_analysis.to_csv('powerbi/time_analysis.csv', index=False)
category_performance.to_csv('powerbi/category_performance.csv', index=False)
channel_performance.to_csv('powerbi/channel_performance.csv', index=False)

print("\nPower BI datasets have been exported to the 'powerbi' directory.")
print("You can now import these files into Power BI to create your dashboard.")

# Close the database connection
conn.close()
