import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import gower

@st.cache
def min_max_col(df_transform, cols):
    '''
    Input
    Takes in a dataframe and a column being a continuous feature to normalize (min =0, and max =1)

    Output
    New dataframe with the column passed normalized
    '''
    mmsc = MinMaxScaler()
    for col in cols:
        var_cont = df_transform.loc[:, col].values.reshape(-1, 1)
        var_cont_standarized = mmsc.fit_transform(var_cont)
        df_transform.loc[:, col] = var_cont_standarized
    return df_transform;

@st.cache
def neighbours_search(Id, df, df_target, n_top):
    '''
    Input
    Id - the id of the client you want to generate leads (string),
    df - the database where you want to look for leads (pandas dataframe),
    df_target - the database of the portfolio of the company the id of the client is in (pandas dataframe),
    n_top - The number of leads you want to generate (integer).

    Output
    Dictionary with the index of the ids generated and the respective coeficient of dissimilarity as keys.
    '''
    # list of non boolean columns that require preprocessing
    non_boolean_cols = ['idade_empresa_anos', 'idade_maxima_socios', 'idade_media_socios', 'idade_minima_socios',
                        'qt_filiais', 'qt_socios', 'qt_socios_st_regular']

    # temporary dataframe to look for the ids after the nearest neighbours are calculated
    temp_df = df

    # getting the index where the id is located in the dataframe
    index = df[df.loc[:, 'id'] == Id].index[0]

    # generating a pandas dataframe with the row where the id is located
    ID_neighbour = df[index:index + 1].drop(columns=['id'])

    # removing the rows with the ids from the portfolio in the database
    df = df[~df.id.isin(df_target.id)]

    # removing the id column as we don't want that column to be inputed as a distance between the clients
    df = df.drop(columns=['id'])

    # preprocessing with normalization the columns that aren't booleans in the dataframes
    df = min_max_col(df, non_boolean_cols)
    ID_neighbour = min_max_col(ID_neighbour, non_boolean_cols)

    # calculating the nearest neighbours with the gower distance
    Nearest_neighbours = gower.gower_topn(ID_neighbour, df, n=n_top)

    # searching and storing the ids found
    leads_ids = []
    for i in range(0, len(Nearest_neighbours.get('index'))):
        leads_ids.append(temp_df.id[Nearest_neighbours.get('index')[i]])

    return Nearest_neighbours, leads_ids;

@st.cache
def load_portfolios():
    df = {
        " ": None,
        "Portfolio 1": pd.read_csv('data/portfolio1_ETL.csv'),
        "Portfolio 2": pd.read_csv('data/portfolio2_ETL.csv'),
        "Portfolio 3": pd.read_csv('data/portfolio3_ETL.csv'),
    }
    return df;

@st.cache
def load_market():
    df = pd.read_csv('data/market_ETL.csv')
    return df;

def main():
    st.title('Similarity Recommender')
    st.markdown("---")
    st.text("This is a lead generator according to a company's portfolio.")


    Choices = st.sidebar.selectbox("Do you have a client you wish to generate leads from?",
    [" ","Yes","No"])


    if Choices == "Yes":
        st.sidebar.title("Lead Generator")
        st.sidebar.markdown("---")
        loading_portfolios = st.sidebar.text('Loading the portfolios...')
        portfolios = load_portfolios()
        loading_portfolios.text('Loading complete!\nNow you can start using the app!')
        portfolio = st.sidebar.selectbox("Select the portfolio of the company you want to look for leads.",
                                         list(portfolios.keys()))

        if portfolios[portfolio] is not None:
            load_database = st.text('Loading the database...')
            market_ID = load_market()
            load_database.text('Loading complete!')
            st.subheader("Market Database")
            st.dataframe(market_ID.head(5))
            df_target = portfolios[portfolio]
            values = df_target.index.tolist()
            options =df_target['id'].tolist()
            dic = dict(zip(options, values))
            Id = st.selectbox('Choose a client', options, format_func=lambda x: dic[x])
            st.write(Id)
            n_top = st.slider('Select the number of leads you want to look for',0, 5)
            st.text('For showcase purposes the maximum amount of leads was set to 5.')
            if n_top > 0:
                data_load_state = st.text('Searching for the neareast neighbours, this may take a while...')
                NN_ID, leads = neighbours_search(Id, market_ID, df_target, n_top)
                data_load_state.text('Found them!')
                for i in range (0,n_top):
                    st.subheader("Lead "+str(i+1))
                    st.markdown('**Index**: '+str(NN_ID.get('index')[i]))

                    st.markdown('**Id**: '+str(leads[i]))

                    st.markdown('**Dissimalirity**: '+str(round(NN_ID.get('values')[i],5)))

    if Choices == "No":
        st.sidebar.title("Cluster Generator")
        st.sidebar.markdown("---")
        loading_portfolios = st.sidebar.text('Loading the portfolios...')
        portfolios = load_portfolios()
        loading_portfolios.text('Loading complete!\nNow you can start using the app!')
        portfolio = st.sidebar.selectbox("Select the portfolio of the company to generate clusters.",
                                         list(portfolios.keys()))

        if portfolios[portfolio] is not None:
            load_database = st.text('Loading the database...')
            market_ID = load_market()
            load_database.text('Loading complete!')
            st.subheader("Market Database")
            st.dataframe(market_ID.head(5))

            '''portfolio1 = min_max_col(portfolio1, non_bolean_cols)
            dissimilarity_matrix = gower.gower_matrix(portfolio1)

            medoids_per_k = []
            k_scores = []
            wss = []
            random.seed(42)
            for i, k in enumerate([2, 3, 4, 5, 6, 7]):
                fig, (ax1) = plt.subplots(1)
                fig.set_size_inches(18, 7)

                initial_medoids_km = random.sample(range(1, 266), k)
                # Run the Kmeans algorithm
                km = kmedoids(dissimilarity_matrix, initial_medoids_km, data_type='distance_matrix')
                km.process()
                # centroids = kmedoids_instance.get_medoids()
                clusters_km = km.get_clusters()
                medoids_km = km.get_medoids()
                medoids_per_k.append(medoids_km)
                labels_km = pd.Series(0, index=range(0, portfolio1.shape[0]))
                for i in range(0, len(clusters_km)):
                    for n in range(0, len(clusters_km[i])):
                        index = clusters_km[i][n]
                        labels_km.iloc[index] = i

                clusters_distances = []
                for n in range(0, len(clusters_km)):
                    clusters_distances.append(X[medoids_km[n]][labels_km[labels_km == n].index].sum())

                wss.append(sum(clusters_distances))

                # Get silhouette samples
                silhouette_vals = silhouette_samples(X, labels_km, metric='precomputed')

                # Silhouette plot
                y_ticks = []
                y_lower, y_upper = 0, 0
                for i, cluster in enumerate(np.unique(labels_km)):
                    cluster_silhouette_vals = silhouette_vals[labels_km == cluster]
                    cluster_silhouette_vals.sort()
                    y_upper += len(cluster_silhouette_vals)
                    ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
                    ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
                    y_lower += len(cluster_silhouette_vals)

                # Get the average silhouette score and plot it

                avg_score = np.mean(silhouette_vals)
                k_scores.append(avg_score)
                ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
                ax1.set_yticks([])
                ax1.set_xlim([-0.1, 1])
                ax1.set_xlabel('Silhouette coefficient values')
                ax1.set_ylabel('Cluster labels')
                ax1.set_title('Silhouette plot for the various clusters', y=1.02);'''

    st.sidebar.title("Useful Links")
    st.sidebar.markdown("---")
    st.sidebar.markdown("[![Github]"
                        "(https://www.startpage.com/av/proxy-image?piurl=https%3A%2F%2Fcdn.iconscout.com%2Ficon%2Ffree%2Fpng-256%2Fgithub-153-675523.png&sp=1594759674Tdf76077b6f2588b1077c86da4bf33f55adb5d35e49be7104e1150f33fceb117a)]"
                        "(https://github.com/Rpinto02)")
    st.sidebar.markdown("[![Linkedin]"
                         "(https://www.startpage.com/av/proxy-image?piurl=https%3A%2F%2Fcdn.iconscout.com%2Ficon%2Ffree%2Fpng-256%2Flinkedin-42-151143.png&sp=1594758987Ta3a7ba5e5bc165c95644e199516c6fc7a4a136a143d412c97997fa27bd624989)]"
                         "(https://www.linkedin.com/in/rpinto02/)")
    st.sidebar.markdown("[![Codenation]"
                        "(<img src='file://codenation.png' alt='alt text' width='200'/>)]"
                        "(https://codenation.dev)")







if __name__ == '__main__':
    main()