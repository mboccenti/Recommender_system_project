import os

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from surprise import KNNBasic, NMF, Dataset, Reader

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Embedding,
    Flatten,
    Input,
    Multiply,
    Dropout,
    Dot
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

models = ("Course Similarity",
          "User Profile",
          "Clustering",
          "Clustering with PCA",
          "KNN",
          "NMF",
          "Neural Network",)
        #   "Regression with Embedding Features",
        #   "Classification with Embedding Features")

path = '/'.join([os.path.dirname(os.path.realpath('backend.py')), 'data'])

def reset_ratings():
    import ssl
    import urllib.request
    ssl._create_default_https_context = ssl._create_unverified_context
    p = '/'.join([path, 'ratings.csv'])
    rating_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/ratings.csv"
    urllib.request.urlretrieve(rating_url, p)


########### COMMENT OUT WHEN DEPLOYING ##########
# reset_ratings()
###########################################

def load_ratings():
    p = '/'.join([path, 'ratings.csv'])
    return pd.read_csv(p)

def load_course_sims():
    p = '/'.join([path, 'sim.csv'])
    return pd.read_csv(p)

def load_courses():
    p = '/'.join([path, 'course_processed.csv'])
    df = pd.read_csv(p)
    df['TITLE'] = df['TITLE'].str.title()
    return df

def load_bow():
    p = '/'.join([path, 'courses_bows.csv'])
    return pd.read_csv(p)

def load_course_genres():
    p = '/'.join([path, 'course_genres.csv'])
    return pd.read_csv(p)

def load_user_profiles():
    p = '/'.join([path, 'user_profiles.csv'])
    return pd.read_csv(p)


def add_new_ratings(selected_courses):
    res_dict = {}
    if len(selected_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        active_user = ratings_df['user'].max() + 1
        user = [active_user] * len(selected_courses)
        ratings = [3.0] * len(selected_courses)
        res_dict['user'] = user
        res_dict['item'] = selected_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        p = '/'.join([path, 'ratings.csv'])
        updated_ratings = pd.concat([ratings_df, new_df], ignore_index=True)
        updated_ratings.to_csv(p, index=False)
        return active_user


# Create course id to index and index to id mappings
def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict


# Model training

# Returned models:
km = KMeans()       #   untrained KMeans model
knn = KNNBasic()    #   untrained KNNBasic model
nmf = NMF()         #   untrained NMF model
nn = None           #   untrained Neural Network
emb = None          #   untrained embeddings
emb_user = None     #   untrained embeddings (user)
emb_clf = None      #   untrained embeddings classifier

user_id2idx = dict()
item_id2idx = dict()

# Training
def train(model_name, params, enrolled_courses):
    global km, knn, nmf, nn, emb, emb_user, emb_clf, user_id2idx, item_id2idx
    if model_name in [models[0], models[1]]: pass
    if model_name in [models[2], models[3]]:
        km = train_cluster(user_profiles=load_user_profiles(),
                                genre_matrix=load_course_genres(),
                                enrolled_courses=enrolled_courses,
                                pca=(model_name==models[3]),
                                exp_var = params['exp_var'],
                                n_clu=params['n_clu']
                                )
    if model_name == models[4]: knn = knn_train(k=params['n_neigh'], verbose=False)
    if model_name == models[5]: nmf = nmf_train(n_factors=params['nmf_factors'], n_epochs=params['nmf_epochs'], verbose=False)
    if model_name == models[6]: 
        # Check whether to use default or not
        if  (params['ncf_val_split'] == 0.1) & (params['ncf_batch_size'] == 512) & (params['ncf_epochs'] == 20):
            p = '/'.join([path, 'ncf_model.h5'])
            nn = tf.keras.models.load_model(p)
        else:
            df = ncf_data_prep(load_ratings())
            ds_train, ds_val = ncf_build_train_val_dataset(df, val_split=params['ncf_val_split'], batch_size=params['ncf_batch_size'])
            nn, _ = ncf_train(ds_train=ds_train, ds_val=ds_val, n_epochs=params['ncf_epochs'])
    # if model_name == models[7]: pass
    # if model_name == models[8]: 
    #     df, user_id2idx, item_id2idx = emb_data_prep()
    #     ds, _, _ = emb_ds_create(df, ['rating'], scale=True, tt_split=1, val_split=0, batch_size=512, rs=None)
    #     emb, emb_user, _ = emb_train(ds=ds, df=load_ratings(), epochs=params['emb_clf_epochs'])

# Prediction
def predict(model_name, active_user, params):
    idx_id_dict, id_idx_dict = get_doc_dicts()
    n_rec = 0
    users = []
    courses = []
    titles = []
    scores = []
    res_dict = {}
    ratings_df = load_ratings()
    courses_df = load_courses()
    user_ratings = ratings_df[ratings_df['user'] == active_user]
    enrolled_course_ids = user_ratings['item'].to_list()

    # Course Similarity model (Similarity Matrix, content based)
    if model_name == models[0]:
        n_rec = params['n_rec_sim']
        res = course_similarity_model(idx_id_dict, id_idx_dict, enrolled_course_ids, load_course_sims().to_numpy(), params['t_rec_sim'])

    # User Profile similarity model (content based)
    if model_name == models[1]:
        n_rec = params['n_rec_profile']
        res = user_profile_similarity_model(enrolled_course_ids, load_course_genres(), params['t_rec_profile'])
        
    # Clustering model (KMeans, with and without PCA)
    if model_name in [models[2], models[3]]:
        n_rec = params['n_rec_clu']
        res = clustering(km, load_user_profiles(), load_ratings(), enrolled_course_ids)

    # k-Nearest Neighbor (KNNBasic, collaborative filtering)
    if model_name == models[4]:
        n_rec = params['n_rec_knn']
        res = knn_predict(knn, enrolled_course_ids)

    # NMF Model (collaborative filtering)
    if model_name == models[5]:
        n_rec = params['n_rec_nmf']
        res = nmf_predict(nmf, enrolled_course_ids)

    if model_name == models[6]:
        n_rec = params['n_rec_ncf']
        res = ncf_predict(nn, enrolled_course_ids)

    # if model_name == models[7]:
    #     n_rec = params['n_rec_ref']
        

    # if model_name == models[8]:
    #     n_rec = params['n_rec_emb_clf']
    #     reg_ds = reg_ds_create(emb, user_id2idx=user_id2idx, item_id2idx=item_id2idx, df=load_ratings())
    #     emb_clf = emb_classifier_train(ds=reg_ds)
    #     res = emb_classifier_predict(model=emb_clf, enrolled_courses=enrolled_course_ids, df_regression=reg_ds)

    if res:
        for key, score in res.items():
            users.append(active_user)
            courses.append(key)
            titles.append(*courses_df['TITLE'][courses_df['COURSE_ID'] == key].values)
            scores.append(score)
        res_dict['USER'] = users
        res_dict['COURSE_ID'] = courses
        res_dict['TITLE'] = titles
        res_dict['SCORE'] = scores
        res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'TITLE', 'SCORE'])
        return res_df[:n_rec]
    else: return pd.DataFrame()


# # Course Similarity Model
def course_similarity_model(idx_id_dict: dict, id_idx_dict: dict, enrolled_courses: list, sim_matrix, t_rec: float) -> dict:
    """
    Returns a sorted (max to min) dictionary of course recommendations.
        Parameters
            idx_id_dict         :   dictionary with course indices as keys, ids as values
            id_idx_dict         :   dictionary with course ids as keys, indices as values
            enrolled_courses    :   list of selected courses by the active user
            sim_matrix          :   similarity matrix (indices x columns: both course ids)
            t_rec               :   recommendation score threshold (0-100)
        Return
            Dictionary  :   Key: Value are COURSE_ID: SCORE
    
    """
    if len(enrolled_courses) < 1: return False
    all_courses = set(idx_id_dict.values())
    # Extract similarities
    # - convert enrolled course IDs to indices
    # - convert course IDs to indices
    # - look up similarity in sim_matrix
    # - store course ID and similarity in sims
    r = {}
    for e in enrolled_courses:
        e_idx = id_idx_dict[e]
        for c in all_courses:
            if not (c in enrolled_courses):
                c_idx = id_idx_dict[c]
                sim = sim_matrix[e_idx][c_idx] * 100
                if sim >= t_rec:
                    r[c] = sim
    
    # sort dictionary
    r = {k: v for k, v in sorted(r.items(), key=lambda item: item[1], reverse=True)}
    return r

# # User Profile Similarity Model
def user_profile_similarity_model(enrolled_courses: list, genre_matrix: pd.DataFrame, t_rec: float) -> dict:
    """
    Returns a dictionary in the form of course_ID: recommendation score based on the user profile similarity to a course genre matrix.
        Parameters
            courses         :   ID list of available courses
            rated_courses   :   ID list of rated courses
            genre_matrix    :   Sparse dataframe with Course IDs and Genres
    """
    r = {}
    # Generating active user profile matrix:
    # - get enrolled courses from genre matrix 
    # - multiply selected courses' genres with 3 to get the weighted genre matrix
    # - flatten (sum up by column) and store as numpy array
    # - store new user profile vector as 'active_user'
    active_user = ((genre_matrix.copy()
            .drop('TITLE', axis=1)
            .set_index('COURSE_ID')
            .loc[enrolled_courses]
        ) * 3).sum().to_numpy()
    
    # multiply active_user with genre matrix of all non-seen courses to get recommendation matrix
    # - modify genre matrix to only contain non-seen courses
    # - dot product active_user and modified genre matrixto get recommendation matrix
    # - return dictionary with normalized (min-max) recommendation scores
    unseen = genre_matrix.loc[~genre_matrix.COURSE_ID.isin(enrolled_courses)]
    rec_matrix = np.dot(active_user, unseen.T[2:])

    for i, c in enumerate(unseen.COURSE_ID):
        if rec_matrix[i]*100 > t_rec:
            r[c] = (rec_matrix[i]/max(rec_matrix))*100

    # sort dictionary
    r = {k: v for k, v in sorted(r.items(), key=lambda item: item[1], reverse=True)}
    return r

# # KMeans clustering
def train_cluster(user_profiles, genre_matrix, enrolled_courses, pca: bool = False, exp_var: int = 80, n_clu: int = 15, init: str = 'k-means++', n_init: int = 10, tol: float = 0.0001, max_iter: int = 200):
    """
    Returns a fit KMeans model and new user profile matrix with the active user added as last position
        Parameters
            user_profiles       :   weighted user genre preferences matrix
            genre_matrix        :   sparse matrix with course ids and their genres
            rated_courses       :   courses already seen (and rated) by active user
            n_clu, optional     :   number of clusters, default = 15
            init, optional      :   clustering algorithm, default = k-means++
            n_init, optional    :   number of centroid initializations, default = 10
            tol, optional       :   tolerance of interia change before declaring convergence, default = 0.0001
            max_iter, optional  :   iterations of algorithm, default = 200
        Returns
            Trained KMeans model
    """

    # Generating active user profile matrix:
    # - get enrolled courses from genre matrix 
    # - multiply selected courses' genres with 3 to get the weighted genre matrix
    # - flatten (sum up by column) and store as numpy array
    # - store new user profile vector as 'active_user'
    active_user = ((genre_matrix.copy()
            .drop('TITLE', axis=1)
            .set_index('COURSE_ID')
            .loc[enrolled_courses]
        ) * 3).sum()
    # Generating dataset to fit KMeans to
    # - concat active_user with user_profiles
    # - extract features (genre weights)
    # - standard scale data
    features = pd.concat([user_profiles.iloc[:, 1:], pd.DataFrame(active_user).T], axis=0, ignore_index=True)
    if pca: features = do_pca(user_profiles, exp_var=exp_var)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    # Clustering training
    # - initialize KMeans according to parameters
    # - fit data to KMeans
    # - return the trained model
    params = {
            'n_clusters': n_clu,
            'init': init,
            'n_init': n_init,
            'tol': tol,
            'max_iter': max_iter
    }
    km = KMeans(**params)
    km.fit(features_scaled)

    return km

def do_pca(data, exp_var: int = 80):
    """
    Returns a PCA reduced dataset according to requested explained variance.
        Parameters
            data    :   dataset to be reduced
            exp_var :   variance to be explained in percent. Default: 80
        Returns
            DataFrame : DataFrame with columns USER and PC1-n
    """
    # Get component number according to explained variance requirement
    # - modify data: it still contains the "user" column
    # - initialize PCA with increasing number of components
    # - fit and transform the data
    # - check explained variance
    # - if it is equal to or exceeds required explained variance, stop.
    exp_var = exp_var / 100
    n_com = 0
    for i in range(min(data.shape[0], data.shape[1])):
        n_com = i
        pca = PCA(n_components=n_com)
        data_reduced = pca.fit_transform(data)
        if (sum(pca.explained_variance_ratio_) >= exp_var): break

    # Return reduced data
    # - store reduced data in a dataframe (df)
    # - label column headers with PC0-n
    df = pd.DataFrame(data_reduced).rename({i: 'PC'+str(i+1) for i in range(n_com)}, axis=1)
    return df

def clustering(model, user_profiles: pd.DataFrame, rated_courses: pd.DataFrame, enrolled_courses: list) -> dict:
    """
    Returns recommendations for the active user as a dictionary (key:value = COURSE_ID:SCORE).
    Recommendation score is calculated as ratio of number of enrollments per course divided by maximum number of enrollments for all recommended courses.
    E.g. if, in the list of recommended courses, course A is found x times and the maximum number of in enrollments is Y with course B, then the recommendation score for
    course A is x/y and the recommendation score for B is 1.
        Parameters
            model               :   Pretrained KMeans model
            rated_courses       :   DataFrame of user ratings
            enrolled_courses    :   List of active user's enrolled (rated) courses
        Returns
            Dictionary          :   Key: COURSE_ID, Value: Recommendation Score
    """
    # Predict recommendations for active user
    # - generate dict for user_ids and corresponding cluster (userID : cluster label)
    # - get cluster label for active user (au_cluster)
    # - extract all users from that cluster except the active user (sim_user)
    # - check similar users' rated courses in that cluster
    # - rank courses
    # - remove courses already rated by the active_user
    # - transform enrollment numbers into 0-100 scoring
    user_cluster_dict = {u: model.labels_[i] for i, u in enumerate(user_profiles.loc[:, user_profiles.columns == 'user'].user)}
    active_user = max(user_cluster_dict.keys())
    au_cluster = user_cluster_dict[active_user]
    sim_users = {'user': [user for user in user_cluster_dict if (user_cluster_dict[user] == au_cluster) & (user != active_user)]}
    sim_users_df = pd.DataFrame.from_dict(sim_users)
    sim_courses_df = pd.DataFrame.merge(sim_users_df, rated_courses, on='user')
    sim_courses_df['count'] = [1] * sim_courses_df.shape[0]
    sim_courses_df = (sim_courses_df.groupby(['item'])
                    .agg(enrollments = ('count', 'sum'))
                    .sort_values(by='enrollments', ascending=False)
                    .reset_index()
                    )
    sim_courses_df = sim_courses_df[~(sim_courses_df.item.isin(enrolled_courses))]
    sim_courses_df.enrollments = (sim_courses_df.enrollments / sim_courses_df.enrollments.max()) * 100
    r = {sim_courses_df.item.iloc[i]: sim_courses_df.enrollments.iloc[i] for i in range(sim_courses_df.shape[0])}

    # sort dictionary
    r = {k: v for k, v in sorted(r.items(), key=lambda item: item[1], reverse=True)}
    return r

# # KNNBasic (surprise)
def knn_train(k: int = 40, min_k: int = 2, metric: str = 'msd', user_based: bool = False, verbose: bool = True) -> KNNBasic:
    """
    Returns a KNNBasic model, trained on user ratings.
        Parameters
            k           :   Number of neighbors to consider (default: 40)
            min_k       :   Minimum numbers of neighbors necessary (default: 2)
            metric      :   Distance metric (default: msd)
            user_based  :   Whether to do user or item based recommendation (default: False)
            verbose     :   Prints process to console (default: False)
        Returns
            Trained KNNBasic Model

        Note: Switching to user_based = True can take a long time! There are tens of thousands of users, so building the similarity matrix is computationally heavy!
    """
    # Setup KNNBasic parameters
    sim_options = {
        'name': metric,
        'user_based': user_based
    }
    params = {
        'k': k,
        'min_k': min_k,
        'sim_options': sim_options,
        'verbose': verbose
    }
    model = KNNBasic(**params)

    # Get data
    reader = Reader(line_format='user item rating', rating_scale = (2,3))
    data = Dataset.load_from_df(load_ratings(), reader)
    # Build a Trainset
    trainset = data.build_full_trainset()
    # Fit the model
    model.fit(trainset)
    return model

def knn_predict(model: KNNBasic, enrolled_courses: list) -> dict:
    """
    Returns predicted course ratings for active user's unseen courses.
        Parameters
            model               :   Trained KNNBasic model
            enrolled_courses    :   List of previously rated courses by the active user
        Returns
            Dictionary  :   Key: Value are Course_ID: Predicted Rating
    """
    # Loading needed data
    # - User Ratings
    # - Course IDs
    # - Active user id (user_id, was appended earlier by add_new_ratings())
    r_df = load_ratings()
    c_df = load_courses()
    user_id = r_df.user.max()

    # Predict recommendation scores
    # - Check if course is not in already known courses
    # - Get ground truth of course rating if possible (THIS IS NOT NECESSARY ANYMORE)
    # - Set ground truths to np.nan (they can't exist yet as the courses are unseen)
    # - Predict active user's ratings using the trained KNNBasic model for each course
    # - Return dictionary with COURSE_ID: Predicted Rating
    r = {}
    for c in c_df.COURSE_ID:
        if not (c in enrolled_courses):
            r_ui = r_df.rating[(r_df.user == user_id) & (r_df.item == c)]
            if r_ui.size < 1: r_ui = np.nan
            else: r_ui = r_ui.values[0]
            p = model.predict(uid=user_id, r_ui=r_ui, iid=c)
            r[c] = (p[3]*100)/3

    # sort dictionary
    r = {k: v for k, v in sorted(r.items(), key=lambda item: item[1], reverse=True)}
    return r

# NMF Model
def nmf_train(n_factors: int = 100, n_epochs: int = 20, verbose: bool = False):
    """
    Returns dictionary of predicted rating scores based on the NMF method
        Parameters
            enrolled_courses    :   List of courses previously rated by the active user

        Returns
            Dictionary  :   Key: Value are Course_ID: Predicted Score
    """
    # load data
    r_df = load_ratings()
    # Build user interaction matrix
    uim = r_df.pivot(index='user', columns='item', values='rating').fillna(0).to_numpy()
    reader = Reader(line_format='user item rating', rating_scale=(2,3))
    data = Dataset.load_from_df(r_df, reader=reader)
    trainset = data.build_full_trainset()
    nmf = NMF(n_factors=n_factors, n_epochs=n_epochs, verbose=verbose)
    nmf.fit(trainset)
    return nmf

def nmf_predict(model, enrolled_courses: list, verbose: bool = False) -> dict:
    r_df = load_ratings()
    c_df = load_courses()
    active_user = r_df.user.max()
    r = dict()
    for c in c_df.COURSE_ID:
        if not (c in enrolled_courses):
            r_ui = r_df.rating[(r_df.user == active_user) & (r_df.item == c)].values
            if len(r_ui)>0: r_ui = r_ui[0]
            else: r_ui = np.nan
            try:
                r[c] = (model.predict(uid=active_user, r_ui=r_ui, iid=c, verbose=verbose)[3]*100)/3
            except:
                pass

    # sort dictionary
    r = {k: v for k, v in sorted(r.items(), key=lambda item: item[1], reverse=True)}
    return r

# NN Model
def ncf_create(n_users: int, n_items: int,                              # user and item number (= input dimensions)
               latent_dim_mf: int = 32, latent_dim_mlp: int = 32,       # latent dimentions for matrix factorization (mf) and multi-layer-perceptron (mlp)
               reg_mf: int = 0, reg_mlp: int = 0.001,                   # regularization strength for mf and mlp
               dense_layers: list = [16, 8, 4],                         # list length is number of dense layers, list value is number of units
               reg_layers: list = [0.01, 0.01, 0.01],                    # list index is dense layer, list value is regularization strength for that dense layer
               activation_dense: str = 'relu'                           # non-linear output layer activation
) -> keras.Model:
    
    # input layer
    user = Input(shape=(), dtype='int32', name='user_id')
    item = Input(shape=(), dtype='int32', name='item_id')

    # embedding layers
    mf_user_embedding = Embedding(input_dim = n_users +1,
                                  output_dim = latent_dim_mf,
                                  name = 'mf_user_embedding',
                                  embeddings_initializer = 'RandomNormal',
                                  embeddings_regularizer = l2(reg_mf),
                                  input_length = 1
                                 )
    mf_item_embedding = Embedding(input_dim = n_items +1,
                                  output_dim = latent_dim_mf,
                                  name = 'mf_item_embedding',
                                  embeddings_initializer = 'RandomNormal',
                                  embeddings_regularizer = l2(reg_mf),
                                  input_length = 1
                                 )

    mlp_user_embedding = Embedding(input_dim = n_users +1,
                                   output_dim = latent_dim_mlp,
                                   name = 'mlp_user_embedding',
                                   embeddings_initializer = 'RandomNormal',
                                   embeddings_regularizer = l2(reg_mlp),
                                   input_length = 1
                                  )
    mlp_item_embedding = Embedding(input_dim = n_items +1,
                                  output_dim = latent_dim_mlp,
                                  name = 'mlp_item_embedding',
                                  embeddings_initializer = 'RandomNormal',
                                  embeddings_regularizer = l2(reg_mlp),
                                  input_length = 1
                                 )

    # MF vectors (flattened embedding outputs)
    mf_user_latent = Flatten()(mf_user_embedding(user))
    mf_item_latent = Flatten()(mf_item_embedding(item))

    # MLP vectors (flattened embedding outputs)
    mlp_user_latent = Flatten()(mlp_user_embedding(user))
    mlp_item_latent = Flatten()(mlp_item_embedding(item))

    # Multiply MF vectors, concat MLP vectors
    mf_cat_latent = Multiply()([mf_user_latent, mf_item_latent])
    mlp_cat_latent = Concatenate()([mlp_user_latent, mlp_item_latent])

    # build dense layers for 
    mlp_vector = mlp_cat_latent
    for i in range(len(dense_layers)):
        layer = Dense(
                      units = dense_layers[i],
                      activation = activation_dense,
                      activity_regularizer = l2(reg_layers[i]),
                      name = 'layer%d' % i,
                     )
        mlp_vector = layer(mlp_vector)
    
    # Results layer (combine last dense layer with mf and put through resulting vector through dense layer 'interaction')
    predict_layer = Concatenate()([mf_cat_latent, mlp_vector])
    result = Dense(
                   units = 1, # only 1 neuron because result!
                   activation = 'sigmoid',
                   kernel_initializer = 'lecun_uniform',
                   name = 'interaction' # rating prediction of user item interaction
                  )
    # get the output
    output = result(predict_layer)

    model = Model(inputs = [user, item],
                  outputs = [output]
                 )
    
    return model

def ncf_data_prep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes ratings dataframe and returns it with interactions binarized ([0, 2, 3] --> [0, 1]), columns renamed
    (user --> user_id, item --> item_id, rating --> interaction) and items number encoded (string --> int).

        Parameters
            df  :   Pandas Dataframe (long) with user ids, items and ratings (shape: (n, 3))
        
        Returns
            df  :   Pandas Dataframe (long) with manipulations performed as mentioned above
    """
    # Build User-interaction matrix
    # Here, I am not merging the "missing" columns (items unseen by all users)
    # as it didn't help during training.
    df_uim = (df.pivot(index='user', columns='item', values='rating')
            .reset_index()
            .rename_axis(columns=None, index=None)
            .fillna(0)
        )
    # Swap item ids for numbers
    old_cols = df_uim.columns[1:]
    new_cols = [i for i in range(len(old_cols))]
    items_id2idx = {old_cols[i]: new_cols[i] for i in range(len(old_cols))}
    df_uim = df_uim.rename(mapper=items_id2idx, axis=1)
    # - Stack df_uim (convert back to long)
    # - convert ratings to binary ([0, 2, 3] --> [0, 1]) by setting everything > 0 = 1
    # - replace user_ids with ascending number list (keras datasets have to start at 0!)
    df_train = (pd.DataFrame(df_uim.iloc[:, 1:].stack())
                .reset_index()
                .sort_values(by='level_0')
                .rename({'level_0': 'user_id', 'level_1': 'item_id', 0: 'interaction'}, axis=1)
               )
    df_train['interaction'] = df_train['interaction'].apply(lambda x: 1 if x > 0 else 0)
    #df_train['user_id'] = [i for i in range(df_train.shape[0])]
    
    # Make sure everything is of a number dtype
    df_train['user_id'] = df_train['user_id'].astype('int')
    df_train['item_id'] = df_train['item_id'].astype('int')
    df_train['interaction'] = df_train['interaction'].astype('int')

    return df_train.sort_values(by=['user_id', 'item_id'])

def ncf_build_train_val_dataset(df: pd.DataFrame, val_split: float = 0.1, batch_size: int = 512, rs: int =42) -> tf.data.Dataset :
    # Generate training and validation datasets
    n_val = round(df.shape[0] * val_split)      # number of validation rows
    if rs: x = df.sample(frac=1, random_state=rs).to_dict('series')
    else:  x = df.to_dict('series')

    # Remove targets from x and store them in y in the same order
    y = dict()
    y['interaction'] = x.pop('interaction')     
    
    ds = tf.data.Dataset.from_tensor_slices((x, y))

    ds_val = ds.take(n_val).batch(batch_size=batch_size)
    ds_train = ds.skip(n_val).batch(batch_size=batch_size)

    return ds_train, ds_val

def ncf_train(ds_train, ds_val, n_epochs: int = 10):
    # # Generate training dataset
    # THIS IS NOW DONE BY IN THE def train()
    # TO ALLOW FOR USER INPUT
    # df = ncf_data_prep(load_ratings())
    # ds_train, ds_val = ncf_build_train_val_dataset(df)
    # Create and compile the model
    n_users, n_items = (load_ratings()
                        .pivot(index='user', columns='item', values='rating')
                        .reset_index()
                        .rename_axis(index=None, columns=None)
                        .shape)
    ncf_model = ncf_create(n_users=n_users, n_items=n_items)
    ncf_model.compile(optimizer = Adam(),
                    loss = 'binary_crossentropy',
                    metrics = [
                                tf.keras.metrics.TruePositives(name="tp"),
                                tf.keras.metrics.FalsePositives(name="fp"),
                                tf.keras.metrics.TrueNegatives(name="tn"),
                                tf.keras.metrics.FalseNegatives(name="fn"),
                                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                                tf.keras.metrics.Precision(name="precision"),
                                tf.keras.metrics.Recall(name="recall"),
                                tf.keras.metrics.AUC(name="auc"),
                                ]
                    )

    ncf_model._name = 'neural_collaborative_filtering'
    # Fit the model
    ncf_hist = ncf_model.fit(x=ds_train, 
                             validation_data=ds_val,
                             epochs=n_epochs,
                             verbose=1
                            )
    return ncf_model, ncf_hist

def ncf_predict(model, enrolled_courses: list = []) -> dict:
    # Get dataset for which to predict
    # - get active user (last user in the ratings dataframe)
    # - extract data only for them
    # - set user_id to 0 for that user
    df = ncf_data_prep(load_ratings())
    active_user = df['user_id'].max()
    df = df[df['user_id'] == active_user]
    df['user_id'] = 0
    ds_pred, _ = ncf_build_train_val_dataset(df=df, val_split=0, rs=None)
    
    # let model predict
    ncf_predictions = model.predict(ds_pred)
    
    # Store predictions in new column
    df['ncf_prediction'] = ncf_predictions
    df['user_id'] = active_user
    
    # return recommendations for the user as a 'sorted dictionary' of structure: ITEM_ID : RECOMMENDATION_SCORE
    # - retransform item_ids from numbers to course names
    # - reset the index and drop the old one (important for dictionary generation)
    # - scale scores to be between 0 and 100 (now: 0, 1)
    # - remove all already seen courses
    # - sort the dictionary and return
    courses = load_ratings().sort_values(by='item', ascending=True)['item'].unique()
    c_idx2id = {k: v for k, v in enumerate(courses)}
    df['item_id'] = df['item_id'].map(c_idx2id)
    df = df.reset_index().drop('index', axis=1)
    r = {}
    for i in range(df.shape[0]):
        if not (df['item_id'][i] in enrolled_courses):
            r[df.loc[i, 'item_id']] = df.loc[i, 'ncf_prediction'] * 100
    r = {k: v for k, v in sorted(r.items(), key=lambda item: item[1], reverse=True)}
    return r

# Embeddings with Regression / Classification
def emb_create(n_user, n_item, n_user_latent_dim: int = 16, n_item_latent_dim: int = 16, reg_users: int = 1e-6, reg_items: int = 1e-6) -> keras.Model:
    """
    Build Neural Network for the embedding. Returns the complete model, and the models for user and item embeddings. Is called by emb_train()
    """
    user_input = Input(shape=(), dtype='int32', name='user')
    item_input = Input(shape=(), dtype='int32', name='item')

    # USER
    user_embedding = Embedding(input_dim=n_user+1,
                    output_dim=n_user_latent_dim,
                    name='user_embedding',
                    embeddings_initializer="he_normal",
                    embeddings_regularizer=keras.regularizers.l2(reg_users)
                    )(user_input)
    user_vec = Flatten(name='user_flat')(user_embedding)
    user_bias = Embedding(input_dim=n_user+1,
                    output_dim=1,
                    name='user_bias',
                    embeddings_initializer="he_normal",
                    embeddings_regularizer=keras.regularizers.l2(reg_users)
                    )(user_input)
    user_model = Model(inputs=user_input, outputs=user_vec)

    # ITEM
    item_embedding = Embedding(input_dim=n_item+1,
                    output_dim=n_item_latent_dim,
                    name='item_embedding',
                    embeddings_initializer="he_normal",
                    embeddings_regularizer=keras.regularizers.l2(reg_items)
                    )(item_input)
    item_vec = Flatten(name='item_flat')(item_embedding)
    item_bias = Embedding(input_dim=n_user+1,
                    output_dim=1,
                    name='item_bias',
                    embeddings_initializer="he_normal",
                    embeddings_regularizer=keras.regularizers.l2(reg_users)
                    )(item_input)
    item_model = Model(inputs=item_input, outputs=item_vec)

    merged = Dot(name='dot', normalize=True, axes=1)([user_embedding, item_embedding])
    merged_dropout = Dropout(0.2)(merged)

    #hidden layers
    dense_1 = Dense(units=64, name='Dense_1')(merged_dropout)
    do_1 = Dropout(0.2, name='Dropout_1')(dense_1)

    dense_2 = Dense(units=32, name='Dense_2')(do_1)
    do_2 = Dropout(0.2, name='Dropout_2')(dense_2)

    dense_3 = Dense(units=16, name='Dense_3')(do_2)
    do_3 = Dropout(0.2, name='Dropout_3')(dense_3)

    dense_4 = Dense(units=8, name='Dense_4')(do_3)

    result = Dense(1, name='rating', activation='relu')(dense_4)

    model = Model(inputs=[user_input, item_input], outputs=[result])
    model._name = 'embedding_extraction_model'
    return model, user_model, item_model

def emb_data_prep(df: pd.DataFrame = load_ratings()):
    data = df.copy()
    user_id2idx = {k: v for v, k in enumerate(data['user'].unique())}
    item_id2idx = {k: v for v, k in enumerate(data['item'].unique())}
    data['user'] = data['user'].map(user_id2idx)
    data['item'] = data['item'].map(item_id2idx)
    return data, user_id2idx, item_id2idx

def emb_ds_create(df: pd.DataFrame, targets: list, scale: bool = True, tt_split: int = 0.8, val_split: int = 0.1, batch_size: int = 512, rs: int = 42):
    """
    Creates TensorFlow datasets from Pandas DataFrame.
        Parameters
            df          :   input DataFrame - only contains features and target(s)
            targets     :   list of columns names corresponding to targets
            val_split   :   fraction of the data that should be used for validation
            batch_size  :   batch size for training
            rs          :   random seed for shuffling data - `None` won't shuffle the data
        Returns
            ds_train    :   training dataset
            ds_val      :   validation dataset
            ds_test     :   test dataset
    """

    n_val = round(df.shape[0] * val_split)      # number of validation rows
    n_split = round(df.shape[0] * tt_split)

    x_train = df.iloc[:n_split, :]
    x_test = df.iloc[n_split:, :]

    if rs: x_train = df.sample(frac=1, random_state=rs).to_dict('series')
    else:  x_train = df.to_dict('series')

    y_train = dict()
    y_test = dict()
    for t in targets:
        y_train[t] = x_train.pop(t)     # remove targets from x and store them in y in the same order
        y_test[t] = x_test.pop(t)
    
    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    ds_val = ds_train.take(n_val).batch(batch_size=batch_size)
    ds_train = ds_train.skip(n_val).batch(batch_size=batch_size)
    ds_test = ds_test.batch(batch_size=batch_size)

    return ds_train, ds_val, ds_test

def emb_train(ds: tf.data.Dataset, df: pd.DataFrame = load_ratings(), epochs: int = 10):
    # compile model
    num_users = len(df['user'].unique())
    num_items = len(df['item'].unique())
    emb_model_all, user_emb_model_all , item_emb_model_all = emb_create(n_user=num_users, n_item=num_items)
    emb_model_all.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(), metrics=tf.keras.metrics.RootMeanSquaredError())
    # train model
    emb_model_all_hist = emb_model_all.fit(ds, epochs=epochs)

    return emb_model_all, user_emb_model_all, item_emb_model_all

def rev_dict(d: dict) -> dict:
    """
    Returns a dictionary with keys and values switched.
    CAUTION: ONLY USE IF KEY: VALUE PAIRS ARE UNIQUE!
    """
    return {v: k for k, v in d.items()}

def reg_ds_create(model, user_id2idx: dict, item_id2idx: dict, df: pd.DataFrame = load_ratings()):
    user_embedding_matrix = model.get_layer('user_embedding').get_weights()[0]
    item_embedding_matrix = model.get_layer('item_embedding').get_weights()[0]
    df_uem = pd.DataFrame(user_embedding_matrix, columns=[f'UFeature{i}' for i in range(16)])
    df_iem = pd.DataFrame(item_embedding_matrix, columns=[f'IFeature{i}' for i in range(16)])

    # insert user_ids
    if not 'user' in df_uem.columns:
        user = pd.DataFrame.from_dict(rev_dict(user_id2idx), orient='index').rename({0: 'user'}, axis=1)
        df_uem = user.merge(df_uem, left_index=True, right_index=True)
    # insert item_ids
    if not 'item' in df_iem.columns:
        items = pd.DataFrame.from_dict(rev_dict(item_id2idx), orient='index').rename({0: 'item'}, axis=1)
        df_iem = items.merge(df_iem, left_index=True, right_index=True)

    # Merge user embedding and course embedding features
    merged_df = pd.merge(df, df_uem, how='left', left_on='user', right_on='user').fillna(0)
    merged_df = pd.merge(merged_df, df_iem, how='left', left_on='item', right_on='item').fillna(0)

    u_features = [f"UFeature{i}" for i in range(16)]
    c_features = [f"IFeature{i}" for i in range(16)]

    user_embeddings = merged_df[u_features]
    item_embeddings = merged_df[c_features]
    ratings = merged_df['rating']
    user = merged_df['user']

    # Aggregate the two feature columns using element-wise add
    regression_dataset = user_embeddings + item_embeddings.values
    regression_dataset.columns = [f"Feature{i}" for i in range(16)]
    regression_dataset['rating'] = ratings
    regression_dataset['user'] = user
    return regression_dataset

def regressor_train(ds):
    X = ds.iloc[:-1, :-2]
    y = ds.iloc[:-1, -2]
    lrr = Ridge(alpha=1e-3)
    lrr.fit(X, y)
    return lrr

def regressor_predict(model, ds, enrolled_courses: list) -> dict:
    model.predict(ds.iloc[-1, :-2])

def emb_classifier_train(ds):
    # take embedding vectors from before and treat it as classification problem with KNN!
    X = ds.iloc[:, :-2]
    y = ds.iloc[:, -2]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y.values.ravel())
    knn_emb = KNeighborsClassifier(n_neighbors=20, n_jobs=-1)
    knn_emb.fit(X, y)

    return knn_emb

def emb_classifier_predict(model, enrolled_courses: list, df_regression: pd.DataFrame, df_ratings: pd.DataFrame = load_ratings()) -> dict:
    """
    Predicts neighbors for the latest user based on the embeddings generated earlier.
    Returns a dictionary in the form of COURSE_ID: SCORE
    """
    knn_emb_pred = model.kneighbors(df_regression.iloc[-1, :-2].values.reshape(1, -1),  n_neighbors=10)
    knn_pred_df = pd.DataFrame(knn_emb_pred[1].reshape(10, 1)).rename({0:'UserIdx'}, axis=1)

    # decode user indices
    if not 'UserID' in knn_pred_df.columns:
        user_ids = pd.DataFrame([df_regression.loc[u, 'user'] for u in knn_pred_df['UserIdx'].values]).rename({0: 'UserID'}, axis=1)
        knn_pred_df = user_ids.merge(knn_pred_df, left_index=True, right_index=True)
    
    knn_courses = list()
    for u in knn_pred_df['UserID'].values:
        for c in df_ratings['item'][df_ratings['user'] == u]:
            if not c in enrolled_courses: knn_courses.append(c)
    max = pd.Series(knn_courses, dtype=object).value_counts().max()
    return pd.Series(knn_courses, dtype=object).value_counts().apply(lambda x: x/max*100).to_dict()