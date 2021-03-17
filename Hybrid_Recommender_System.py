#############################################
# PROJE: Hybrid Recommender System
#############################################

#############################################
# Adım 1: Verinin Hazırlanması
#############################################

import pandas as pd
from helpers.helpers import create_user_movie_df

user_movie_df = create_user_movie_df()


user_movie_df.head(10)
user_id = 108170

#############################################
# Adım 2: Öneri yapılacak kullanıcının izlediği filmlerin belirlenmesi
#############################################

# user_id ye göre veriyi filtreliyoruz
user_id_df = user_movie_df[user_movie_df.index == user_id]
user_id_df

# izlediği filmlerin bulunması, NA olmayanların belirlenmesi
movies_watched = user_id_df.columns[user_id_df.notna().any()].tolist()
movies_watched

# izlediği filmlerin sayısı
len(movies_watched)


#############################################
# Adım 3: Aynı filmleri izleyen diğer kullanıcıların verisine ve id'lerine erişmek
#############################################
pd.set_option('display.max_columns', 5)

# İlgilendiğimiz kullanıcının izlediği filmlere göre veriyi filtreledik
movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()
movies_watched_df.shape


# Her bir kullanıcının izlediği toplam film sayısı
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count.head()

# user'in izlediği film sayısının yüzde 60'ı hesaplandı.
perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
users_same_movies.head()
len(users_same_movies) # 2342

#############################################
# Adım 4: Öneri yapılacak kullanıcı ile en benzer kullanıcıların belirlenmesi
#############################################

# İlgilendiğimiz kullanıcı ile bu kullanıcıya benzer diğer kullanıcıların tüm verilerini birleştirdik

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies.index)],
                      user_id_df[movies_watched]])

final_df.head()

final_df.T.corr()

# Korelasyon df oluşturuldu.
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()
corr_df.head()

# user id = 108170 olan kullanıcıyla en benzer olan kullanıcıların bulunması (corr değeri 0.65'den büyük olanlar)
top_users = corr_df[(corr_df["user_id_1"] == user_id) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

top_users

# Bu kullanıcıların rating tablosundan hangi filme ne kadar puan verdiğini getirelim
rating = pd.read_csv(r'C:\Users\LENOVO\PycharmProjects\DSMLBC4\HAFTA_10\rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

top_users_ratings
# rating tablosuyla birleştirdiğimiz için kayıtlar çoklama geldi

#############################################
# Adım 5: Weighted rating'lerin  hesaplanması
#############################################

# En benzer kullanıcıların puanlarını göz önünde bulundurmak istiyoruz
# Bu nedenle corr ve rating i kullanarak tek bir metriğe dönüştürdük
# Ratinglere korelasyona bağlı bir düzeltme yapacağız.
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()


#############################################
# Adım 6: Weighted average recommendation score'un hesaplanması ve ilk beş filmin tutulması
#############################################
temp = top_users_ratings.groupby('movieId').sum()[['corr', 'weighted_rating']]
temp.columns = ['sum_corr', 'sum_weighted_rating']

temp.head()


recommendation_df = pd.DataFrame()
recommendation_df['weighted_average_recommendation_score'] = temp['sum_weighted_rating'] / temp['sum_corr']
recommendation_df['movieId'] = temp.index
recommendation_df = recommendation_df.sort_values(by='weighted_average_recommendation_score', ascending=False)
recommendation_df.head(20)

recommendation_df

# Bu filmler hangi filmler ?
movie = pd.read_csv(r'C:\Users\LENOVO\PycharmProjects\DSMLBC4\HAFTA_10\movie.csv')
movies_from_user_based = movie.loc[movie['movieId'].isin(recommendation_df['movieId'].head(10))]['title']
movies_from_user_based.head()


#############################################
# Adım 7: İzlediği filmlerden en son en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
#############################################

movie = pd.read_csv(r'C:\Users\LENOVO\PycharmProjects\DSMLBC4\HAFTA_10\movie.csv')
rating = pd.read_csv(r'C:\Users\LENOVO\PycharmProjects\DSMLBC4\HAFTA_10\rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()
#  Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınması:
movie_id = rating[(rating["userId"] == user_id) & (rating["rating"] == 5.0)]. \
               sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

# Movie title'inin alınması:
movie_title = movie[movie["movieId"] == movie_id]["title"].str.replace('(\(\d\d\d\d\))', '').str.strip().values[0]

#########
# title
#########

df['year_movie'] = df.title.str.extract('(\(\d\d\d\d\))', expand=False)

df['year_movie'] = df.year_movie.str.extract('(\d\d\d\d)', expand=False)

df['title'] = df.title.str.replace('(\(\d\d\d\d\))', '')

df['title'] = df['title'].apply(lambda x: x.strip())


#########
# genres
#########

df["genre"] = df["genres"].apply(lambda x: x.split("|")[0])
df.drop("genres", inplace=True, axis=1)
df.head()

############
# timestamp
############
df["timestamp"] = pd.to_datetime(df["timestamp"], format='%Y-%m-%d')
df.info()
df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day

######################################
# User Movie Df'inin Oluşturulması
######################################
a = pd.DataFrame(df["title"].value_counts())
a.head()

# Filtreleme
rare_movies = a[a["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape
common_movies["title"].nunique()

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.shape
user_movie_df.head(10)
user_movie_df.columns

len(user_movie_df.columns)
common_movies["title"].nunique()


######################################
# Korelasyona Dayalı Item-Based Film Önerilerinin Yapılması
######################################

movie = user_movie_df[movie_title]
movies_from_item_based = user_movie_df.corrwith(movie).sort_values(ascending=False)
movies_from_item_based[1:6].index

# user-based 5 öneri ve item-based 5 öneriyi bir araya getirilmesi

recommendation_10 = pd.DataFrame()
recommendation_10['user_based_recommendations'] = movies_from_user_based[:5].values.tolist()
recommendation_10['item_based_recommendations'] = movies_from_item_based[1:6].index
recommendation_10