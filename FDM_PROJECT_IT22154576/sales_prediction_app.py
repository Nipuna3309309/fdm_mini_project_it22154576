import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the pre-trained Random Forest Regressor model
model_path = 'FDM_PROJECT_IT22154576/rf_regressor.pkl'  # Path to your saved model
rf_regressor = joblib.load(model_path)

# Load dataset for preprocessing
data3 = pd.read_csv('Video_Games.csv')

# Preprocessing for training
scaler = MinMaxScaler()
columns_to_scale = ['Critic_Score', 'Critic_Count', 'Year_of_Release']
data3[columns_to_scale] = scaler.fit_transform(data3[columns_to_scale])

# One-hot encode categorical columns
data_final = pd.get_dummies(data3, columns=['Genre', 'Rating', 'Publisher', 'Platform', 'Developer'])

# Define the title of the web app
st.title("Global Sales Prediction")



# Create a tab for introduction
tab1, tab2, tab3 = st.tabs(["Introduction", "Charts", "Prediction"])

# Introduction tab
with tab1:
    st.title("Introduction to Dataset Prediction")
    st.write("Welcome to our dataset prediction application!")
    st.write("This application uses a pre-trained Random Forest Regressor model to predict global sales of video games based on various factors such as platform, genre, publisher, and critic score.")
    st.write("The dataset used for training the model is a collection of video game sales data from various sources.")
    st.write("The application is designed to provide a user-friendly interface for users to input the details of a video game and predict its global sales.")

    # Add some animations to make it more beautiful
# Plot charts in the "Charts" tab
with tab2:
    # Plot a bar chart to show the distribution of global sales
    fig, ax = plt.subplots()
    ax.bar(['0-10', '10-20', '20-30', '30-40'], [10, 20, 30, 40])
    ax.set_title('Distribution of Global Sales')
    ax.set_xlabel('Global Sales (million units)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Plot a line chart to show the trend of global sales over time
    fig, ax = plt.subplots()
    ax.plot([1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020], [10, 15, 20, 25, 30, 35, 40, 45])
    ax.set_title('Trend of Global Sales Over Time')
    ax.set_xlabel('Year')
    ax.set_ylabel('Global Sales (million units)')
    st.pyplot(fig)

    # Plot a pie chart to show the distribution of global sales by platform
    fig, ax = plt.subplots()
    ax.pie([10, 20, 30, 40], labels=['PS', 'Xbox', 'Nintendo', 'PC'], autopct='%1.1f%%')
    ax.set_title('Distribution of Global Sales by Platform')
    st.pyplot(fig)

# Taking inputs from user in the "Prediction" tab
with tab3:
    st.write("Enter the details to predict Global Sales")
    platform = st.selectbox('Platform', ['DS', 'PS2', 'Wii', 'PSP', 'PS3', 'X360', 'PS', 'PC', 'Others'])
    year_of_release = st.slider('Year of Release', 2020, 2040, 2024)
    genre = st.selectbox('Genre', ['Action', 'Sports', 'Misc', 'Role-Playing', 'Adventure', 'Shooter', 'Racing', 'Simulation', 'Fighting', 'Platform', 'Strategy', 'Puzzle'])
    publisher = st.selectbox('Publisher', ['Electronic Arts', 'Namco Bandai Games', 'Activision', 'Ubisoft', 'Konami Digital Entertainment', 'THQ', 'Sega', 'Sony Computer Entertainment', 'Others'])
    critic_score = st.slider('Critic Score', 0, 100, 75)
    critic_count = st.number_input('Critic Count', value=50)
    developer = st.selectbox('Developer', ['Ubisoft', 'Konami', 'EA Canada', 'EA Sports', 'Capcom', 'EA Tiburon', 'unknown', 'Others'])
    rating = st.selectbox('Rating', ['E', 'T', 'E10+', 'M', 'EC', 'RP', 'K-A', 'unknown'])

    # Convert user input into a DataFrame
    user_input = pd.DataFrame({
        'Platform': [platform],
        'Year_of_Release': [year_of_release],
        'Genre': [genre],
        'Publisher': [publisher],
        'Critic_Score': [critic_score],
        'Critic_Count': [critic_count],
        'Developer': [developer],
        'Rating': [rating]
    })

    # Preprocess user input
    # Scale the numerical columns in user input
    user_input[columns_to_scale] = scaler.transform(user_input[columns_to_scale])

    # One-hot encode the user input
    user_input_preprocessed = pd.DataFrame({
        'Year_of_Release': [year_of_release],
        'Critic_Score_Filled': [critic_score],
        'Critic_Count_Filled': [critic_count],
        
 'Genre_Action': [genre == 'Action'],
        #'Genre_Adventure': [genre == 'Adventure'],
        'Genre_Fighting': [genre == 'Fighting'],
        'Genre_Misc': [genre == 'Misc'],
        'Genre_Platform': [genre == 'Platform'],
        'Genre_Puzzle': [genre == 'Puzzle'],
        'Genre_Racing': [genre == 'Racing'],
        'Genre_Role-Playing': [genre == 'Role-Playing'],
        'Genre_Shooter': [genre == 'Shooter'],
        'Genre_Simulation': [genre == 'Simulation'],
        'Genre_Sports': [genre == 'Sports'],
        'Genre_Strategy': [genre == 'Strategy'],
        'Rating_E': [rating == 'E'],
        'Rating_E10+': [rating == 'E10+'],
        'Rating_EC': [rating == 'EC'],
        'Rating_K-A': [rating == 'K-A'],
        #'Rating_M': [rating == 'M'],
        'Rating_RP': [rating == 'RP'],
        'Rating_T': [rating == 'T'],
        'Rating_unknown': [rating == 'unknown'],
        'Publisher_Categorized_Activision': [publisher == 'Activision'],
        'Publisher_Categorized_Electronic Arts': [publisher == 'Electronic Arts'],
        'Publisher_Categorized_Konami Digital Entertainment': [publisher == 'Konami Digital Entertainment'],
        'Publisher_Categorized_Namco Bandai Games': [publisher == 'Namco Bandai Games'],
        'Publisher_Categorized_Others': [publisher == 'Others'],
        #'Publisher_Categorized_Sega': [publisher == 'Sega '],
        'Publisher_Categorized_Sony Computer Entertainment': [publisher == 'Sony Computer Entertainment'],
        'Publisher_Categorized_THQ': [publisher == 'THQ'],
        'Publisher_Categorized_Ubisoft': [publisher == 'Ubisoft'],
        'Platform_Categorized_DS': [platform == 'DS'],
        'Platform_Categorized_Others': [platform == 'Others'],
        'Platform_Categorized_PC': [platform == 'PC'],
        'Platform_Categorized_PS': [platform == 'PS'],
        #'Platform_Categorized_PS2': [platform == 'PS2'],
        'Platform_Categorized_PS3': [platform == 'PS3'],
        'Platform_Categorized_PSP': [platform == 'PSP'],
        'Platform_Categorized_Wii': [platform == 'Wii'],
        'Platform_Categorized_X360': [platform == 'X360'],
        'Developer_Categorized_Capcom': [developer == 'Capcom'],
        'Developer_Categorized_EA Canada': [developer == 'EA Canada'],
        'Developer_Categorized_EA Sports': [developer == 'EA Sports'],
        'Developer_Categorized_EA Tiburon': [developer == 'EA Tiburon'],
        'Developer_Categorized_Electronic Arts': [developer == 'Electronic Arts'],
        #'Developer_Categorized_Konami': [developer == 'Konami'],
        'Developer_Categorized_Others': [developer == 'Others'],
        'Developer_Categorized_Ubisoft': [developer == 'Ubisoft'],
        'Developer_Categorized_unknown': [developer == 'unknown'],
        
    })

    # Create a button to predict global sales
    if st.button("Predict Global Sale"):
        # Make predictions using the pre-trained model
        prediction = rf_regressor.predict(user_input_preprocessed)

        # Define the minimum and maximum values of the 'Global_Sales' column
        global_sales_min = 0.01
        global_sales_max = 40.24

        # Inverse transform the prediction to get the actual value
        prediction_actual = (prediction[0] * (global_sales_max - global_sales_min)) + global_sales_min

        # Display the prediction
        st.write(f"Predicted Global Sales: {prediction_actual:.2f} million units")
