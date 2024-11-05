import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
from model import weighted_knn_recommendation

df_sample = pd.read_csv('spotify_data.csv', header=0)

numerical_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                      'speechiness', 'acousticness', 'instrumentalness', 
                      'liveness', 'valence', 'tempo']

categorical_features = ['genre']  

# Handle missing features
for feature in numerical_features + categorical_features:
    if feature not in df_sample.columns:
        print(f"Feature '{feature}' not found in the DataFrame columns.")

df_numerical = df_sample[numerical_features]
df_categorical = df_sample[categorical_features]

one_hot_encoder = OneHotEncoder()
encoded_genres = one_hot_encoder.fit_transform(df_categorical)

full_features = pd.concat([df_numerical, pd.DataFrame(encoded_genres.toarray())], axis=1)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_numerical)

final_features = pd.concat([pd.DataFrame(scaled_features), pd.DataFrame(encoded_genres.toarray())], axis=1)

def main():
    st.title("Music Recommendation System")
    st.subheader("Enter Song IDs:")
    song_ids = st.text_area("Input song IDs (comma-separated)", '')

    if st.button("Get Recommendations"):
        if song_ids:
            input_songs = [id.strip() for id in song_ids.split(',')]
            recommended_songs = weighted_knn_recommendation(input_songs, df_sample, scaler, numerical_features, categorical_features, final_features, one_hot_encoder, k=10)
            st.subheader("Recommended Songs:")
            st.write(recommended_songs)
        else:
            st.error("Please enter at least one song ID.")

if __name__ == "__main__":
    main()
