import pandas as pd
from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

app = FastAPI()

games_df = pd.read_csv('new_games.csv')
reviews_df = pd.read_csv('new_review.csv')
archivos_partes = ['items_parte_1.csv', 'items_parte_5.csv']

# Crear una lista de DataFrames, uno por cada parte
partes = [pd.read_csv(archivo) for archivo in archivos_partes]

# Concatenar los DataFrames en uno solo
items_df = pd.concat(partes, ignore_index=True)

@app.get('/')
def read_root():
    return {'message' : 'API para consultar datos de Juegos'}


@app.get('/userdata/{User_id}')
def userdata(User_id: str):
        # Encontrar todos los 'item_id' del usuario en 'items_df'
    user_items = items_df[items_df['user_id'] == User_id]['item_id']

    # Encontrar los precios de esos 'item_id' en 'games_df'
    user_prices = games_df[games_df['id'].isin(user_items)]['price']

    # Calcular la cantidad de dinero gastado
    total_money_spent = user_prices.sum()

     # Redondear el valor a dos decimales
    money_spent_rounded = round(total_money_spent.item(), 2)

    # Convertir a cadena con el formato deseado (dos decimales)
    money_spent_formatted = "{:.2f}".format(money_spent_rounded)

    # Encontrar el porcentaje de recomendación en 'reviews_df'
    user_reviews = reviews_df[reviews_df['user_id'] == User_id]
    recommend_percentage = (user_reviews['recommend'].sum() / len(user_reviews)) * 100

    # Encontrar la cantidad de items en 'items_df'
    user_item_count = items_df[items_df['user_id'] == User_id]['items_count'].iloc[0]

    return {
        "user_id": User_id,
        "money_spent": money_spent_formatted,
        "recommend_percentage": recommend_percentage,
        "item_count": int(user_item_count)
    }


@app.get('/counterviews/{start_date},{end_date}')
def countreviews(start_date: str, end_date: str):
    # Convertir las fechas de inicio y fin a objetos datetime para comparar
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Convertir la columna 'posted' a objetos Timestamp, con errores 'coerce'
    reviews_df['posted'] = pd.to_datetime(reviews_df['posted'], errors='coerce')

    # Filtrar las reseñas dentro del rango de fechas, excluyendo las fechas inválidas (NaT)
    filtered_reviews = reviews_df[(reviews_df['posted'] >= start_date) & (reviews_df['posted'] <= end_date) & (~reviews_df['posted'].isna())]

    # Contar la cantidad de usuarios únicos que realizaron reseñas dentro del rango de fechas
    unique_users_count = filtered_reviews['user_id'].nunique()

    # Calcular el porcentaje de recomendación y redondearlo
    recommend_percentage = round((filtered_reviews['recommend'].mean() * 100), 2)

    return {
        "start_date": start_date.strftime('%Y-%m-%d'),
        "end_date": end_date.strftime('%Y-%m-%d'),
        "unique_users_count": unique_users_count,
        "recommend_percentage": recommend_percentage
    }


# Paso 1: Calcular el ranking de géneros más jugados
genre_ranking = items_df.groupby('item_id')['playtime_forever'].sum().reset_index()
genre_ranking = genre_ranking.merge(games_df[['id', 'genres']], left_on='item_id', right_on='id', how='inner')

# Paso 2: Filtrar juegos con géneros no nulos
genre_ranking = genre_ranking.dropna(subset=['genres'])

# Paso 3: Identificar géneros y contar horas jugadas por género
genre_ranking['genres'] = genre_ranking['genres'].apply(lambda x: x.strip('[]').replace("'", "").split(', '))
genre_ranking = genre_ranking.explode('genres')
genre_ranking = genre_ranking.groupby('genres')['playtime_forever'].sum().reset_index()

# Paso 4: Clasificar los géneros en función del tiempo de juego
genre_ranking = genre_ranking.sort_values(by='playtime_forever', ascending=False).reset_index(drop=True)

@app.get('/genre/{genero}')
# Función para obtener el puesto de un género específico
def get_genre_ranking(genero: str):
    try:
        rank = genre_ranking.index[genre_ranking['genres'] == genero].tolist()[0] + 1
        return { 'Genero' : genero , 'Ranking': rank}
    except IndexError:
        return None
    

@app.get('/userforgenre/{genero}')
def userforgenre(genero: str):
    # Paso 1: Filtrar juegos por género
    genre_games = games_df[games_df['genres'].str.contains(genero, case=False, na=False)]
    genre_game_ids = genre_games['id'].tolist()

    # Paso 2: Filtrar usuarios por juegos del género
    genre_users = items_df[items_df['item_id'].isin(genre_game_ids)]

    # Paso 3: Crear un ranking de usuarios basado en las horas jugadas en ese género
    genre_users_ranking = genre_users.groupby('user_id')['playtime_forever'].sum().reset_index()
    genre_users_ranking = genre_users_ranking.sort_values(by='playtime_forever', ascending=False).head(5)

    # Paso 4: Obtener user_id, user_url y horas jugadas de los 5 mejores usuarios
    top_users = []
    for _, row in genre_users_ranking.iterrows():
        user_id = row['user_id']
        user_playtime = row['playtime_forever']
        user_url = items_df[items_df['user_id'] == user_id]['user_url'].iloc[0]
        top_users.append({"user_id": user_id, "user_url": user_url, "playtime_forever": user_playtime})

    return top_users


@app.get("/developer/{developer}")
def developer(developer: str):
    # Paso 1: Filtrar juegos por desarrollador
    developer_games = games_df[games_df['developer'] == developer]

    # Paso 2: Convertir la columna 'release_date' al formato de fecha
    developer_games['release_date'] = pd.to_datetime(developer_games['release_date'], errors='coerce')

    # Paso 3: Separar los juegos por año de lanzamiento
    developer_games['year'] = developer_games['release_date'].dt.year

    # Paso 4: Contar la cantidad total de juegos y la cantidad de juegos gratuitos por año
    yearly_stats = developer_games.groupby('year').agg(
        total_items=pd.NamedAgg(column='id', aggfunc='count'),
        free_items=pd.NamedAgg(column='price', aggfunc=lambda x: (x == 0).sum())
    ).reset_index()

    # Paso 5: Calcular el porcentaje de contenido gratuito por año
    yearly_stats['free_percentage'] = (yearly_stats['free_items'] / yearly_stats['total_items']) * 100

    # Paso 6: Crear un diccionario con los resultados en formato JSON
    result = {
        "developer": developer,
        "yearly_stats": yearly_stats[['year', 'free_percentage']].to_dict(orient='records')
    }

    return result


@app.get('/sentiment_analysis/{year}')
def sentiment_analysis(year: int):
    # Paso 1: Convertir la columna 'posted' al formato de fecha
    reviews_df['posted'] = pd.to_datetime(reviews_df['posted'], errors='coerce')

    # Paso 2: Filtrar reseñas por año
    reviews_year = reviews_df[reviews_df['posted'].dt.year == year]

    # Paso 3: Contar la cantidad de registros con análisis de sentimiento
    sentiment_counts = reviews_year['sentiment_analysis'].value_counts().to_dict()

    # Paso 4: Devolver los resultados en el formato deseado
    result = {
        "Negative": sentiment_counts.get(0.0, 0),
        "Neutral": sentiment_counts.get(1.0, 0),
        "Positive": sentiment_counts.get(2.0, 0)
    }

    return result

games_df['developer'].fillna('', inplace=True)
games_df['tags'].fillna('', inplace=True)
games_df['price'].fillna('', inplace=True)

# Crear una columna 'features' que contiene la concatenación de las columnas relevantes
games_df['features'] = games_df['developer'] + ' ' + games_df['tags'] + ' ' + games_df['price'].astype(str)

# Crear la matriz TF-IDF a partir de la columna 'features'
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(games_df['features'])

# Entrenar el modelo Nearest Neighbors con la matriz TF-IDF
nn_model = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
nn_model.fit(tfidf_matrix)

# Función para obtener recomendaciones de juegos similares
@app.get('/recomendacion_juego/{product_id}')
def recomendacion_juego(product_id: int):
    # Encontrar el índice del juego en función del 'id' del producto
    game_index = games_df[games_df['id'] == product_id].index[0]

    # Encontrar los juegos más similares utilizando Nearest Neighbors
    distances, indices = nn_model.kneighbors(tfidf_matrix[game_index], n_neighbors=6)

    # Crear una lista de juegos recomendados (excluyendo el juego de consulta)
    recommended_games = []
    for i in range(1, len(indices[0])):
        recommended_game = {
            "id": int(games_df.iloc[indices[0][i]]['id']),  # Convierte a int
            "app_name": str(games_df.iloc[indices[0][i]]['app_name']),  # Convierte a str
        }
        recommended_games.append(recommended_game)

    return {'juegos recomendados': recommended_games}