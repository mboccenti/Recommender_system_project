import streamlit as st
import pandas as pd
import backend as backend

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode

# Basic webpage setup
st.set_page_config(
   page_title="Course Recommender System",
   layout="wide",
   initial_sidebar_state="expanded",
)


# ------- Functions ------
# Load datasets
@st.cache_data
def load_ratings():
    return backend.load_ratings()


@st.cache_data
def load_course_sims():
    return backend.load_course_sims()


@st.cache_data
def load_courses():
    return backend.load_courses()


@st.cache_data
def load_bow():
    return backend.load_bow()

@st.cache_data
def load_genre():
    return backend.load_course_genres()

@st.cache_data
def load_user_profiles():
    return backend.load_user_profiles()

# Initialize the app by first loading datasets
def init__recommender_app():

    with st.spinner('Loading datasets...'):
        ratings_df = load_ratings()
        sim_df = load_course_sims()
        course_df = load_courses()
        course_bow_df = load_bow()

    # Select courses
    st.success('Datasets loaded successfully...')

    st.markdown("""---""")
    st.subheader("Select courses that you have completed: ")

    # Build an interactive table for `course_df`
    gb = GridOptionsBuilder.from_dataframe(course_df)
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()
    grid_options = gb.build()

    # Create a grid response
    response = AgGrid(
        course_df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
    )

    results = pd.DataFrame(response["selected_rows"], columns=['COURSE_ID', 'TITLE', 'DESCRIPTION'])
    results = results[['COURSE_ID', 'TITLE']]
    st.subheader("Your courses: ")
    st.table(results)
    return results


def train(model_name, params):
    # Start training course similarity model
    with st.spinner('Training...'):
        backend.train(model_name, params, selected_courses_df.COURSE_ID)
        st.success('Done!')



def predict(model_name, active_user, params):
    res = None
    # Start making predictions based on model name, test user ids, and parameters
    with st.spinner('Generating course recommendations: '):
        #time.sleep(0.5)
        res = backend.predict(model_name, active_user, params)
    st.success('Recommendations generated!')
    return res

# ------ UI ------
# Sidebar
st.sidebar.title('Personalized Learning Recommender')
# Initialize the app
selected_courses_df = init__recommender_app()

# Model selection selectbox
st.sidebar.subheader('1. Select recommendation models')
model_selection = st.sidebar.selectbox(
    "Select model:",
    backend.models
)

# Hyper-parameters for each model
params = {}
st.sidebar.subheader('2. Tune Hyper-parameters: ')
# Course similarity model
if model_selection == backend.models[0]:
    n_rec_sim = st.sidebar.slider('Top courses', min_value=1, max_value=100, value=10, step=1)
    t_rec_sim = st.sidebar.slider('Course Similarity Threshold %', min_value=0, max_value=100, value=50, step=10)
    params['n_rec_sim'] = n_rec_sim
    params['t_rec_sim'] = t_rec_sim
# User profile model
elif model_selection == backend.models[1]:
    n_rec_profile = st.sidebar.slider('Top courses', min_value=1, max_value=100, value=10, step=1)
    t_rec_profile = st.sidebar.slider('User Profile Similarity Threshold %', min_value=0, max_value=100, value=50, step=10)
    params['n_rec_profile'] = n_rec_profile
    params['t_rec_profile'] = t_rec_profile
# Clustering model
elif model_selection == backend.models[2]:
    n_rec_clu = st.sidebar.slider('Top courses', min_value=1, max_value=100, value=10, step=1)
    n_clu = st.sidebar.slider('Number of Clusters', min_value=1, max_value=50, value=20, step=1)
    params['n_rec_clu'] = n_rec_clu
    params['n_clu'] = n_clu
    params['exp_var'] = 0
# Clustering + PCA model
elif model_selection == backend.models[3]:
    n_rec_clu_pca = st.sidebar.slider('Top courses', min_value=1, max_value=100, value=10, step=1)
    n_clu_pca = st.sidebar.slider('Number of Clusters', min_value=1, max_value=50, value=20, step=1)
    exp_var = st.sidebar.slider('Explained Variance', min_value=1, max_value=100, value=80, step=1)
    params['n_rec_clu'] = n_rec_clu_pca
    params['n_clu'] = n_clu_pca
    params['exp_var'] = exp_var
# KNN
elif model_selection == backend.models[4]:
    n_rec_knn = st.sidebar.slider('Top courses', min_value=1, max_value=100, value=10, step=1)
    n_neigh = st.sidebar.slider('Number of Neighbors', min_value=1, max_value=50, value=20, step=1)
    params['n_rec_knn'] = n_rec_knn
    params['n_neigh'] = n_neigh
# NMF model
elif model_selection == backend.models[5]:
    n_rec_nmf = st.sidebar.slider('Top courses', min_value=1, max_value=100, value=10, step=1)
    nmf_factors = st.sidebar.slider('NMF Factors', min_value=1, max_value=140, value=70, step=1)
    nmf_epochs = st.sidebar.slider('SGD Epochs', min_value=1, max_value=100, value=50, step=1)
    params['n_rec_nmf'] = n_rec_nmf
    params['nmf_factors'] = nmf_factors
    params['nmf_epochs'] = nmf_epochs
# Neural Net
elif model_selection == backend.models[6]:
    n_rec_ncf = st.sidebar.slider('Top courses', min_value=1, max_value=100, value=10, step=1)
    params['ncf_val_split'] = 0.1
    params['ncf_batch_size'] = 512
    params['ncf_epochs'] = 20
    with st.sidebar.expander('Advanced options.', False):
        st.caption('Changing these options will cause the app to train a new model, instead of using the pretrained one.')
        st.info('Depending on your settings, training can take a significant amount of time (minutes to  many hours).')
        st.caption('Training dataset generation')
        ncf_val_split = st.slider('Validation Split', min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        ncf_batch_size = st.slider('Batch size', min_value=0, max_value=1024, value=512, step=64)
        st.caption('Model training')
        ncf_epochs = st.slider('Epochs', min_value=1, max_value=100, value=20, step=1)
    params['ncf_val_split'] = ncf_val_split
    params['ncf_batch_size'] = ncf_batch_size
    params['ncf_epochs'] = ncf_epochs
    params['n_rec_ncf'] = n_rec_ncf
# # Regression with embedding
# elif model_selection == backend.models[7]: pass
#     # Regression with embedding features hyperparameters
# # Classification model
# elif model_selection == backend.models[8]:
#     n_rec_emb_clf = st.sidebar.slider('Top courses', min_value=1, max_value=100, value=10, step=1)
#     emb_clf_epochs = st.sidebar.slider('Epochs', min_value=1, max_value=100, value=10, step=1)
#     params['emb_clf_epochs'] = emb_clf_epochs
#     params['n_rec_emb_clf'] = n_rec_emb_clf
else:
    pass


# Training
st.sidebar.subheader('3. Training: ')
training_button = st.sidebar.button("Train Model")
training_text = ''
# Start training process
if training_button:
    train(model_name=model_selection, params=params)

# Prediction
st.sidebar.subheader('4. Prediction')
# Start prediction process
pred_button = st.sidebar.button("Recommend New Courses")
if pred_button and selected_courses_df.shape[0] > 0:
    # Create a new id for current user session
    active_user = backend.add_new_ratings(selected_courses_df['COURSE_ID'].values)
    user_ids = [active_user]
    # Get and display predictions
    # - call predict with necessary parameters
    # - format received dataframe to give only 2 decimals instead of 4.
    res_df = predict(model_selection, active_user, params)
    st.table(res_df.style.format({"SCORE": "{:.2f}"}))