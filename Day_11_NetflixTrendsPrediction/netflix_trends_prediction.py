import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from thirty_days_pyai_helpers.print import print_intro, slow_print, slow_print_header

def process_data(df):
    # Parse date_added to get year
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["year_added"] = df["date_added"].dt.year.fillna(0).astype(int)
    
    # Drop rows with missing year_added
    df = df[df["year_added"] != 0]
    return df

def calculate_types(df):
    # Group by year_added and type to count movies and TV shows
    counts = df.groupby(["year_added", "type"]).size().unstack(fill_value=0)
    counts["total"] = counts["Movie"] + counts["TV Show"]
    # Calculate proportions for both TV shows and movies
    counts["proportion_tv"] = counts["TV Show"] / counts["total"]
    counts["proportion_movies"] = counts["Movie"] / counts["total"]
    return counts

def predict_trends(counts):
    # Prepare data for regression (TV shows as the primary target)
    X = counts.index.values.reshape(-1, 1)
    y_tv = counts["proportion_tv"].values  # Proportion of TV shows
    y_movies = counts["proportion_movies"].values  # Proportion of movies for plotting
    
    # Fit linear regression model for TV shows
    model = LinearRegression()
    model.fit(X, y_tv)
    y_tv_pred = model.predict(X)  # Historical predictions for TV shows
    r2 = r2_score(y_tv, y_tv_pred)
    
    # Predict future years
    last_year = int(X.max())
    years_to_predict = 5
    future_years = np.arange(last_year + 1, last_year + 1 + years_to_predict).reshape(-1, 1)
    future_tv_predictions = model.predict(future_years)
    future_tv_predictions = np.clip(future_tv_predictions, 0, 1)
    # Calculate movie predictions as the complement
    future_movie_predictions = 1 - future_tv_predictions
    
    # Visualize results with both TV shows and movies
    visualize_result(X, y_tv, y_movies, model, future_years, future_tv_predictions, future_movie_predictions)
    
    # Print regression results and predictions
    slow_print(f"Slope (TV Show Trend): {model.coef_[0]:.4f}")
    slow_print(f"R-squared (TV Show Model): {r2:.4f}")
   
    slow_print_header("Historical Proportions:")
    print(counts[['proportion_tv', 'proportion_movies']])
    
    slow_print_header("Predicted Proportions for Future Years:")
    for year, tv_pred, movie_pred in zip(future_years.flatten(), future_tv_predictions, future_movie_predictions):
        slow_print(f"Year {int(year)}: TV Shows = {tv_pred:.4f}, Movies = {movie_pred:.4f}")
    

def visualize_result(X, y_tv, y_movies, model, future_years, future_tv_predictions, future_movie_predictions):
    plt.figure(figsize=(12, 6))
    
    # Historical data
    plt.scatter(X, y_tv, color='blue', label='Historical TV Shows')
    plt.scatter(X, y_movies, color='orange', label='Historical Movies')
    plt.plot(X, model.predict(X), color='blue', linestyle='-', label='TV Show Trend')
    plt.plot(X, 1 - model.predict(X), color='orange', linestyle='-', label='Movie Trend')
    
    # Future predictions
    plt.plot(future_years, future_tv_predictions, color='blue', linestyle='--', label='Predicted TV Shows')
    plt.plot(future_years, future_movie_predictions, color='orange', linestyle='--', label='Predicted Movies')
    
    plt.xlabel('Year Added')
    plt.ylabel('Proportion of Content')
    plt.title('Trend of TV Shows vs. Movies Added to Netflix Over Time with Future Predictions')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print_intro(1, "Netflix Trends", "Predicting TV vs Movies")
    
    # Download the dataset
    path = kagglehub.dataset_download("shivamb/netflix-shows")
    file_path = f"{path}/netflix_titles.csv"
    df = pd.read_csv(file_path)
    slow_print("Netflix Trend Data has been downloaded.")
    
    df = process_data(df)
    counts = calculate_types(df)
    predict_trends(counts)