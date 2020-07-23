# Client Similarity Recommender engine

## Introduction

This a machine learning approach to generate leads for companies based on their portfolio.

## Getting Started

### Dependencies

* Python 3.8
* Pandas 1.05 
* Numpy 1.19.0
* Streamlit 0.62.1
* Scikit-learn 0.23.1
* Plotly 4.8.1
* Umap-learn 0.4.5
* Pyclustering 0.9.3.1
* [Gower](https://github.com/wwwjk366/gower) 0.0.5
**Warning** other dependencies are needed if you want to explore the old models, however they are not used in the program.

### Instructions
Open the notebook on the preparation folder for a step by step description of the decision making process of the data preparation and cleaning.
Afterwards you can find notebooks for the exploration of outliers in the database inside model folder as well as the old models that led to final working model.
In this same folder there is also the clustering analysis to solve the cold start problem of the recommendation engine as well as the notebook with the working function for the recommendations.
Finally follow the instructions below to run the app.

### Executing Program

* Run the following commands in the project's app root directory.
  * To run ETL Pipeline that cleans data and stores in database:
  ```
  python app/ETL.py data/estaticos_market.csv data/market_ETL.csv
  ```
* Run the following command in the app's directory to run your web app:
```
streamlit run main.py
```
* Alternatively you can visit the [website](https://market-lead-generator.herokuapp.com/)
**Sidenote** due to Heroku free restrictions only a sample of the database was deployed to showcase the project. However the app works properly with the whole database.

### Project presentation
A video with the presentation of the project with details behind the several decisions taken throught the development can be found here.
(Work in progress)

## License
[MIT](https://opensource.org/licenses/MIT)

## Issue/Bug

Please open issues on github to report bugs or make feature requests.


# Recomendador de clientes por similaridade

# Introdução

Este repositório é uma tentativa de usar aprendizado de máquina para gerar leads para empresas baseado no seu portfolio.

## Como Começar
### Dependências

* Python 3.8
* Pandas 1.05 
* Numpy 1.19.0
* Streamlit 0.62.1
* Scikit-learn 0.23.1
* Plotly 4.8.1
* Umap-learn 0.4.5
* Pyclustering 0.9.3.1
* [Gower](https://github.com/wwwjk366/gower) 0.0.5
**Aviso** outras dependências são necessárias para explorar os modelos antigos, no entanto elas não são usadas no programa. 

### Instruções
Abra o notebook na pasta preparation para uma descrição detalhada do processo decisório durante a preparação e limpeza dos dados.
Em seguida pode encontrar notebooks com a exploração de outliers na base dados assim como os antigos modelos que levaram ao modelo final.
Na mesma pasta pode encontrar também a análise dos clusters para resolver o cold start problem do recommendados e ainda o notebook com a função usada para as recomendações.
Finalmente siga as instruções abaixo para correr a aplicação web.

### Executando o programa

* Corra os seguintes comandos no diretório app do projeto:
  * Para correr a ETL Pipeline que limpa e salva a base de dados:
  ```
  python app/ETL.py data/estaticos_market.csv data/market_ETL.csv
  ```
* Corra o seguinte comando no diretório app para abrir a aplicação web:
```
streamlit run main.py
```
* Alternativamente pode visitar o [website](https://market-lead-generator.herokuapp.com/)
**Nota** devido às restrições da versão grátis do Heroku apenas uma amostra da base dados foi lançada no app. No entanto ela funciona devidamente com toda a base de dados.


### Apresentação do projeto
Um video com a apresentação do projeto contendo detalhes por detrás das decisões tomadas durante o desenvolvimento pode ser encontrado aqui.
(Em desenvolvimento)

## Licença
[MIT](https://opensource.org/licenses/MIT)

## Problema/Bug

Por favor abre um problema no github para reportar bugs ou pedidos para desenvolvimento de features.
