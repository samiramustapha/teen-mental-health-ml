# Samira Mustapha
# TAC 216, 31832
# Final Project
# Description: Predicting teen mental health outcomes based on social media usage, sleep, academic performance, and current physical and mental health levels.

# Data wrangling and visualization imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Machine learning imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report


def main():
    # ---------- DATA WRANGLING ----------
    # Loading and cleaning the data
    df = pd.read_csv('Teen_Mental_Health_Dataset.csv')
    df = df.drop_duplicates()
    df = df.dropna()
    print("DATA INFO:")
    df.info() # structure of the data; 3 categorical & 10 numerical columns
    

    # ---------- SCIENTIFIC COMPUTATION ----------
    # Average stress by addiction level
    stress_by_addict = df.groupby('addiction_level')['stress_level'].mean()
    print("\nSTRESS LEVEL BY ADDICTION LEVEL\n", stress_by_addict)
    # Average anxiety level by daily social media usage -- Is heavy usage linked to higher anxiety?
    anxiety_by_social_media = df.groupby('daily_social_media_hours')['anxiety_level'].mean()
    print("\nANXIETY LEVEL BY DAILY SOCIAL MEDIA USAGE\n", anxiety_by_social_media)
    # Percentage of people with depression by social media usage level
    # Note: the depression_label column only has binary values (0 = not depressed, 1 = depressed)
    ppl_per_sm_group = df.groupby('daily_social_media_hours')['depression_label'].count() # num ppl in each social media group
    depression_social_media_counts = df.groupby('daily_social_media_hours')['depression_label'].sum() # num people depressed in each social media group
    depression_social_media_percent = (depression_social_media_counts / ppl_per_sm_group) * 100
    print("\nPERCENTAGE OF DEPRESSED TEENS BY DAILY SOCIAL MEDIA USAGE\n", depression_social_media_percent)
    # Average social media usage by gender
    avg_sm_by_gender = df.groupby('gender')['daily_social_media_hours'].mean()
    print("\nAVERAGE SOCIAL MEDIA USAGE BY GENDER\n", avg_sm_by_gender)
    # Total usage per platform
    platform_counts = df['platform_usage'].value_counts(ascending=False)
    print("\nTOTAL USAGE PER PLATFORM\n", platform_counts)
    # Average Depression Rates per Platform
    depression_by_platform = df.groupby('platform_usage')['depression_label'].mean() 
    print("\nAVERAGE DEPRESSION RATES BY PLATFORM")
    print(f"Both: {float(depression_by_platform['Both'])*100:.5f}%.")
    print(f"Instagram: {float(depression_by_platform['Instagram'])*100:.5f}%.")
    print(f"TikTok: {float(depression_by_platform['TikTok'])*100:.5f}%.")
    # Impact of sleep on anxiety and depression -- do anxiety and depression levels increase or decrease with hours slept?
    sleep_anx_dep = df.groupby('sleep_hours')[['anxiety_level', 'depression_label']].mean()
    print("\nIMPACT OF SLEEP ON ANXIETY AND DEPRESSION\n", sleep_anx_dep)
    # Impact of physical activity on anxiety and depression -- do anxiety and depression levels increase or decrease with physical activity?
    activity_anx_dep = df.groupby('physical_activity')[['anxiety_level', 'depression_label']].mean()
    print("\nIMPACT OF PHYSICAL ACTIVITY ON ANXIETY AND DEPRESSION\n", activity_anx_dep)
    # Impact of social media hours on academic performance (GPA)
    print("\nIMPACT OF SOCIAL MEDIA USAGE ON ACADEMIC PERFORMANCE (GPA)")
    median_sm = df['daily_social_media_hours'].median()
    low_sm = df[df['daily_social_media_hours'] <= median_sm]
    high_sm = df[df['daily_social_media_hours'] > median_sm]
    print(f"Avg GPA of teens with LOW social media usage: {low_sm['academic_performance'].mean():.5f}")
    print(f"Avg GPA of teens with HIGH social media usage: {high_sm['academic_performance'].mean():.5f}")

    # ---------- MACHINE LEARNING ----------
    # Splitting the data into features (X: numerical data) and target (y: target)
    X = df[['daily_social_media_hours', 'sleep_hours', 'screen_time_before_sleep', 
           'academic_performance', 'physical_activity', 'stress_level', 'anxiety_level', 'addiction_level']]
    y = df['depression_label']
    # train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 100)
    # Scaling
    scaler = StandardScaler()
    scaler.fit(X_train) 
    X_train = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index = X_test.index)
    # Creating and training the model
    model = MLPClassifier(hidden_layer_sizes=(8, 8), max_iter=1000, random_state=100) # used 8,8 for my hidden layer size because the dataset has 8 features
    model.fit(X_train, y_train)
    # Getting predictions
    y_pred = model.predict(X_test)
    print("\nML Predictions:")
    print(y_pred)
    print("\nActual values:")
    print(y_test.values)
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"\nTest Accuracy: {model.score(X_test, y_test):.4f}")

    # ---------- DATA VISUALIZATION ----------
    # Create figure with 2 subplots
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Teen Mental Health: Social Media Impact Analysis')
    # Plot 1: Raw data bar plot of social media hours vs stress, colored by depression
    platform_list = ['Instagram', 'TikTok', 'Both']
    depression_rates = [float(depression_by_platform['Instagram'])*100, 
                        float(depression_by_platform['TikTok'])*100, 
                        float(depression_by_platform['Both'])*100]
    ax[0].bar(platform_list, depression_rates, color=['blue', 'orange', 'green'], alpha=0.7)
    ax[0].set_title('Raw Data: Depression Rates by Platform')
    ax[0].set_xlabel('Platform')
    ax[0].set_ylabel('Depression Rate (%)')

    ax[0].grid(True, axis='y')
    # Plot 2: ML results and confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Depression', 'Depression'])
    disp.plot(ax=ax[1], cmap='Blues')
    ax[1].set_title('ML Model: Confusion Matrix')
    fig.tight_layout()
    fig.savefig('teen_mental_health_data_analysis.png', dpi=150)
    plt.show()
    
    # ---------- USER INPUT: "DEEP DIVE" ----------
    print("\nLet's take a deep dive!")
    dive = input("What would you like to explore?\n1. Stress, anxiety, and depression in a specific age group\n2. Screen addiction and depression rate by platform\n3. Depression rates between high vs. low social media usage\nEnter 1, 2, or 3: ")
    while not dive.isdigit() or not 1 <= int(dive) <= 3:
      dive = input("What would you like to explore?\n1. Stress, anxiety, and depression in a specific age group\n2. Screen addiction and depression rate by platform\n3. Depression rates between high vs. low social media rates\nEnter 1, 2, or 3: ")

    if dive == '1':
      age = input("Which age (13-19) would you like to examine? ")
      while not age.isdigit() or not 13 <= int(age) <= 19:
         age = input("Which age (13-19) would you like to examine? ")
      age = int(age)
      age_df = df[df['age'] == age]
      if len(age_df) == 0:
         print(f"\nNo teens aged {age} in this dataset.")
      else:
         print(f"\nFor teens aged {age}:")
         print(f"Average stress: {age_df['stress_level'].mean():.2f}")
         print(f"Average anxiety: {age_df['anxiety_level'].mean():.2f}")
         print(f"Depression rate: {(age_df['depression_label'].sum() / len(age_df)) * 100:.2f}%")
    elif dive == '2':
      valid_platforms = {'instagram': 'Instagram', 'tiktok': 'TikTok', 'both': 'Both'}
      platform = input("Enter the platform you'd like to explore (Instagram/TikTok/Both): ").strip().lower()
      while platform not in valid_platforms:
         platform = input("Enter the platform you'd like to explore (Instagram/TikTok/Both): ").strip().lower()
      platform = valid_platforms[platform]
      platform_data = df[df['platform_usage'] == platform]
      if len(platform_data) == 0:
         print(f"\nNo {platform} users in this dataset.")
      else:
         print(f"\nScreen addiction and depression in {platform} users:")
         print(f"Average daily usage: {platform_data['daily_social_media_hours'].mean():.2f} hours")
         print(f"Average addiction level: {platform_data['addiction_level'].mean():.2f}")
         print(f"Depression rate: {(platform_data['depression_label'].sum() / len(platform_data)) * 100:.2f}%")
    elif dive == '3':
      median = df['daily_social_media_hours'].median()
      low_users = df[df['daily_social_media_hours'] <= median]
      high_users = df[df['daily_social_media_hours'] > median]
      low_dep_rate = (low_users['depression_label'].sum()/len(low_users))*100
      high_dep_rate = (high_users['depression_label'].sum()/len(high_users))*100
      print("\nDepression rates between high vs. low social media usage:")
      print(f"Depression rate in high-volume users: {high_dep_rate:.5f}")
      print(f"Depression rate in low-volume users: {low_dep_rate:.5f}")

if __name__ == '__main__':
   main()
