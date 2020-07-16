import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
import random
import gower
import plotly.graph_objects as go
import umap as umap



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
def calculate_distance(df):
    # list of non boolean columns that require preprocessing
    non_boolean_cols = ['idade_empresa_anos', 'idade_maxima_socios', 'idade_media_socios',
                        'idade_minima_socios',
                        'qt_filiais', 'qt_socios', 'qt_socios_st_regular']

    #normalizing the non boolean columns
    df = min_max_col(df, non_boolean_cols)

    #calculating the gower distance matrix
    dissimilarity_matrix = gower.gower_matrix(df)
    return dissimilarity_matrix;

@st.cache(allow_output_mutation=True)
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
                data_load_state = st.text('Searching for the nearest neighbours, this may take a while...')
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


            calculating = st.text('Calculating the dissimilarity matrix! This may take a while...')
            dissimilarity_matrix = calculate_distance(portfolios[portfolio])
            calculating.text('Phew, we finally finished the calculus!')
            X = dissimilarity_matrix


            #creating the lists we'll want to save values to
            medoids_per_k = []#medoids for each number of clusters
            clusters_per_k = []#clusters for each number of clusters
            k_scores = []#average silhouette score of k clusters
            wss = []#the sum of dissimilarity of each cluster

            random.seed(42)
            for i, k in enumerate([2, 3, 4, 5, 6, 7]):

                #the medoids algorithm requires an initial point to start so we're setting it here
                initial_medoids_km = random.sample(range(1,portfolios[portfolio].shape[0]), k)

                # Run the Kmeans algorithm
                km = kmedoids(X, initial_medoids_km, data_type='distance_matrix')
                km.process()

                #saving the created clusters into a list
                clusters_km = km.get_clusters()
                clusters_per_k.append(clusters_km)

                #saving the medoids that were found
                medoids_km = km.get_medoids()

                #saving the medoids that were found per each number of clusters into a list
                medoids_per_k.append(medoids_km)

                #creating a dataframe with the labels of each cluster
                labels_km = pd.Series(0, index=range(0, portfolios[portfolio].shape[0]))
                for i in range(0, len(clusters_km)):
                    for n in range(0, len(clusters_km[i])):
                        index = clusters_km[i][n]
                        labels_km.iloc[index] = i

                #getting the sum of the dissimilarity per cluster
                clusters_distances = []
                for n in range(0, len(clusters_km)):
                    clusters_distances.append(X[medoids_km[n]][labels_km[labels_km == n].index].sum())

                #total sum of the dissimilarity
                wss.append(sum(clusters_distances))

                # Get silhouette samples
                silhouette_vals = silhouette_samples(X, labels_km, metric='precomputed')

                # Silhouette plot
                fig = go.Figure()
                fig.update_layout(
                    title={
                        'text': 'Silhouette plot for ' + str(k) + ' clusters',
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                    xaxis_title='Silhouette coefficient values',
                    yaxis_title='Cluster labels',
                    font=dict(
                        family="Courier New, monospace",
                        size=16,
                        color="RebeccaPurple"
                    ),
                    autosize=False,
                    width=1000,
                    height=600,
                    margin=dict(
                        l=50,
                        r=50,
                        b=100,
                        t=100,
                        pad=4
                    ),
                    paper_bgcolor="LightGrey"

                )
                y_lower, y_upper = 0, 0
                annotations = []
                for i, cluster in enumerate(np.unique(labels_km)):
                    cluster_silhouette_vals = silhouette_vals[labels_km == cluster]
                    cluster_silhouette_vals.sort()
                    y_upper += len(cluster_silhouette_vals)

                    fig.add_trace(go.Bar(x=cluster_silhouette_vals,
                                         y=np.array((range(y_lower, y_upper))),
                                         name=str(i + 1),
                                         orientation='h',
                                         showlegend=False))

                    annotations.append(dict(x=-0.03,
                                            y=(y_lower + y_upper) / 2,
                                            text=str(i + 1),
                                            showarrow=False))
                    y_lower += len(cluster_silhouette_vals)
                fig.update_layout(annotations=annotations)



                # Get the average silhouette score
                avg_score = np.mean(silhouette_vals)

                #saving the average silhouette score of k clusters in a list
                k_scores.append(avg_score)

                #plottting the average silhouette score
                fig.update_layout(shapes=[
                    dict(
                        type='line',
                        yref='paper', y0=0, y1=1,
                        xref='x', x0=avg_score, x1=avg_score,
                        line=dict(color='green', width=2,dash='dash')
                    )
                ])
                fig.update_yaxes(showticklabels=False)

                #plotting the graphs created in streamlit
                st.plotly_chart(fig)

            fig_wss = go.Figure()
            fig_wss.update_layout(
                title={
                    'text': 'Dissimilarity plot - The Elbow Method',
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                xaxis_title='Number of Clusters',
                yaxis_title='Dissimilarity',
                font=dict(
                    family="Courier New, monospace",
                    size=16,
                    color="RebeccaPurple"
                ),
                autosize=False,
                width=1000,
                height=600,
                margin=dict(
                    l=50,
                    r=50,
                    b=100,
                    t=100,
                    pad=4
                ),
                paper_bgcolor="LightGrey"

            )
            fig_wss.add_trace(go.Scatter(x=list(range(2, 8)), y=wss,
                                     mode='lines+markers'))
            st.plotly_chart(fig_wss)

            st.markdown("Now comes the fun part, I am going to challenge you to chose the best number of clusters!")
            list_clusters = [0,2,3,4,5,6,7]
            number_clusters = st.selectbox("How many clusters do you want to use?", list_clusters)
            if number_clusters is not 0:
                graphics = st.text("Creating shiny plots...")
                medoids = medoids_per_k[number_clusters-2]#The medoids and clusters lists starts at index 0 which is with 2 clusters, and finishes at 5, 7 clusters thus the -2
                clusters = clusters_per_k[number_clusters-2]
                fit_umap = umap.UMAP(n_neighbors=14, min_dist=0.1, n_components=3, metric='dice', random_state=42)
                p_umap = fit_umap.fit_transform(portfolios[portfolio].drop(columns=['id']))

                # Visualising the clusters

                fig_umap = go.Figure()
                for i in range(0,number_clusters):
                    fig_umap.add_trace(go.Scatter3d(x=p_umap[clusters[i], 0],
                                               y=p_umap[clusters[i], 1],
                                               z=p_umap[clusters[i], 2],
                                               name='Cluster '+str(i),
                                               mode='markers'))

                fig_umap.add_trace(go.Scatter3d(x=p_umap[medoids, 0],
                                           y=p_umap[medoids, 1],
                                           z=p_umap[medoids, 2],
                                           name='Medoids',
                                           mode='markers',
                                           marker_color="rgb(255,255,0)",
                                           marker=dict(size=16)))

                fig_umap.update_layout(
                    title={
                        'text': 'Clusters with the Dice Distance',
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                    font=dict(
                        family="Courier New, monospace",
                        size=16,
                        color="RebeccaPurple"
                    ),
                    autosize=False,
                    width=1000,
                    height=600,
                    margin=dict(
                        l=50,
                        r=50,
                        b=100,
                        t=100,
                        pad=4
                    )
                )

                st.plotly_chart(fig_umap)

                fit_umap_man = umap.UMAP(n_neighbors=14, min_dist=0.1, n_components=3, metric='manhattan', random_state=42)
                p_umap_man = fit_umap_man.fit_transform(portfolios[portfolio].drop(columns=['id']))

                fig_umap_man = go.Figure()
                for i in range(0, number_clusters):
                    fig_umap_man.add_trace(go.Scatter3d(x=p_umap_man[clusters[i], 0],
                                                    y=p_umap_man[clusters[i], 1],
                                                    z=p_umap_man[clusters[i], 2],
                                                    name='Cluster ' + str(i),
                                                    mode='markers'))

                fig_umap_man.add_trace(go.Scatter3d(x=p_umap_man[medoids, 0],
                                                y=p_umap_man[medoids, 1],
                                                z=p_umap_man[medoids, 2],
                                                name='Medoids',
                                                mode='markers',
                                                marker_color="rgb(255,255,0)",
                                                marker=dict(size=16)))

                fig_umap_man.update_layout(
                    title={
                        'text': 'Clusters with the Manhattan Distance',
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                    font=dict(
                        family="Courier New, monospace",
                        size=16,
                        color="RebeccaPurple"
                    ),
                    autosize=False,
                    width=1000,
                    height=600,
                    margin=dict(
                        l=50,
                        r=50,
                        b=100,
                        t=100,
                        pad=4
                    )
                )

                st.plotly_chart(fig_umap_man)
                graphics.text('3D clusters visualization complete!')


                selection = st.selectbox('Choose a representative client(Medoid)', medoids)
                Id = portfolios[portfolio].loc[selection,'id']
                st.write(Id)
                n_top = st.slider('Select the number of leads you want to look for', 0, 5)
                st.text('For showcase purposes the maximum amount of leads was set to 5.')
                df_target = portfolios[portfolio]
                if n_top > 0:
                    data_load_state = st.text('Searching for the nearest neighbours, this may take a while...')
                    NN_ID, leads = neighbours_search(Id, market_ID, df_target, n_top)
                    data_load_state.text('Found them!')
                    for i in range(0, n_top):
                        st.subheader("Lead " + str(i + 1))
                        st.markdown('**Index**: ' + str(NN_ID.get('index')[i]))

                        st.markdown('**Id**: ' + str(leads[i]))

                        st.markdown('**Dissimalirity**: ' + str(round(NN_ID.get('values')[i], 5)))





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