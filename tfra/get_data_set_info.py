import tensorflow_datasets as tfds

# Load the dataset and include the dataset info
dataset, info = tfds.load('movielens/1m-ratings', split='train', with_info=True, data_dir='~/dataset')

# Print out all available features
print(info.features)

"""
FeaturesDict({
    'bucketized_user_age': float32,
    'movie_genres': Sequence(ClassLabel(shape=(), dtype=int64, num_classes=21)),
    'movie_id': string,
    'movie_title': string,
    'timestamp': int64,
    'user_gender': bool,
    'user_id': string,
    'user_occupation_label': ClassLabel(shape=(), dtype=int64, num_classes=22),
    'user_occupation_text': string,
    'user_rating': float32,
    'user_zip_code': string,
})
"""