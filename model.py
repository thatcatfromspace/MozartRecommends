import pandas as pd
from sklearn.neighbors import NearestNeighbors

def weighted_knn_recommendation(input_songs, df_sample, scaler, numerical_features, categorical_features, final_features, one_hot_encoder, k=10):
    
    model = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='minkowski')

    input_song_features = df_sample[df_sample['track_id'].isin(input_songs)][numerical_features + categorical_features]
    if input_song_features.empty:
        return []

    scaled_input_features = scaler.transform(input_song_features[numerical_features])

    encoded_input_genres = one_hot_encoder.transform(input_song_features[categorical_features])

    # Combine scaled numerical and one-hot encoded genre features
    combined_input_features = pd.concat([pd.DataFrame(scaled_input_features), 
                                         pd.DataFrame(encoded_input_genres.toarray())], axis=1)

    model.fit(final_features)
    distances, indices = model.kneighbors(combined_input_features, n_neighbors=k)

    weights = 1 / (distances + 1e-5) 

    recommended_songs = {}
    for idx, song_neighbors in enumerate(indices):
        for j, neighbor_idx in enumerate(song_neighbors):
            track_id = df_sample.iloc[neighbor_idx]['track_id']
            if track_id in recommended_songs:
                recommended_songs[track_id] += weights[idx][j]
            else:
                recommended_songs[track_id] = weights[idx][j]

    # Sort by highest weighted score
    sorted_recommendations = sorted(recommended_songs.items(), key=lambda x: x[1], reverse=True)
    
    # Extract the top N recommended songs
    recommended_song_ids = [track_id for track_id, _weight in sorted_recommendations[:k]]

    # Retrieve song names and artist names using song IDs
    recommended_songs_with_details = []
    for track_id in recommended_song_ids:
        song_info = df_sample[df_sample['track_id'] == track_id][['track_name', 'artist_name']].iloc[0]
        recommended_songs_with_details.append((track_id, song_info['track_name'], song_info['artist_name']))

    return recommended_songs_with_details





