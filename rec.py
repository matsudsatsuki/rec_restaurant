import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
from notion_client import Client

# 環境変数のロード
load_dotenv()
# Streamlit Secrets または環境変数から値を取得
NOTION_API_KEY = st.secrets["NOTION_API_KEY"] if "NOTION_API_KEY" in st.secrets else os.environ.get("NOTION_API_KEY")
DATABASE_ID = st.secrets["DATABASE_ID"] if "DATABASE_ID" in st.secrets else os.environ.get("DATABASE_ID")

# 環境変数が設定されていることを確認
if not NOTION_API_KEY or not DATABASE_ID:
    st.error("Notion API Key または Database ID が設定されていません。")
    st.stop()
# Notion APIクライアントを初期化
notion = Client(auth=NOTION_API_KEY)

# Notionからデータを取得する関数
def get_notion_data():
    query = notion.databases.query(database_id=DATABASE_ID)
    results = query["results"]
    
    data = []
    for page in results:
        properties = page["properties"]
        item = {
            "店名": properties["店名"]["title"][0]["plain_text"] if properties["店名"]["title"] else "",
            "タグ": get_tags(properties["タグ"]),
            "URL": properties["URL"]["url"] if properties["URL"]["url"] else "",
            "紹介者": properties["紹介者"]["rich_text"][0]["plain_text"] if properties["紹介者"]["rich_text"] else "",
            "店舗最寄り駅": get_station(properties["店舗最寄り駅"])
        }
        data.append(item)
    
    return pd.DataFrame(data)

def get_tags(tag_property):
    if "multi_select" in tag_property:
        return ", ".join([tag["name"] for tag in tag_property["multi_select"]])
    elif "select" in tag_property:
        return tag_property["select"]["name"] if tag_property["select"] else ""
    else:
        return ""

def get_station(station_property):
    if "select" in station_property:
        return station_property["select"]["name"] if station_property["select"] else ""
    elif "rich_text" in station_property:
        return station_property["rich_text"][0]["plain_text"] if station_property["rich_text"] else ""
    else:
        return ""

# データの取得
df = get_notion_data()

# ユーザー-アイテムマトリックスの作成
user_restaurant_pairs = df['紹介者'].str.split(',').explode().reset_index()
user_restaurant_pairs.columns = ['restaurant_index', 'user']
user_restaurant_pairs['rating'] = 1  # すべての推薦を1としてカウント

user_item_matrix = user_restaurant_pairs.pivot_table(
    values='rating',
    index='user',
    columns=df['店名'],
    aggfunc='sum',
    fill_value=0
)

# 各ユーザーの評価を正規化
user_item_matrix = user_item_matrix.div(user_item_matrix.sum(axis=1), axis=0)

# コサイン類似度の計算
user_similarity = cosine_similarity(user_item_matrix)
item_similarity = cosine_similarity(user_item_matrix.T)

# ユーザーベースの推薦
def user_based_recommendation(user, n=3):
    if user not in user_item_matrix.index:
        return []
    
    user_index = user_item_matrix.index.get_loc(user)
    similar_users = user_similarity[user_index].argsort()[::-1][1:6]  # 上位5人の類似ユーザー
    
    recommendations = user_item_matrix.iloc[similar_users].mean().sort_values(ascending=False)
    already_liked = user_item_matrix.loc[user]
    recommendations = recommendations[recommendations.index.difference(already_liked[already_liked > 0].index)]
    
    return recommendations.head(n).index.tolist()

# アイテムベースの推薦
def item_based_recommendation(restaurant, n=3):
    if restaurant not in user_item_matrix.columns:
        return []
    
    item_index = user_item_matrix.columns.get_loc(restaurant)
    similar_items = item_similarity[item_index].argsort()[::-1][1:6]  # 上位5個の類似アイテム
    
    return user_item_matrix.columns[similar_items][:n].tolist()

# Streamlitアプリ
st.title('PFD部おすすめごはん')

# データテーブルの表示
st.subheader("レストランデータ")
st.dataframe(df)

# ユーザー選択
users = user_item_matrix.index.tolist()
selected_user = st.selectbox('ユーザーを選択してください:', users)

# レストラン選択
restaurants = df['店名'].tolist()
selected_restaurant = st.selectbox('お気に入りのレストランを選択してください:', restaurants)

# ユーザーベースの推薦
if st.button('ユーザーベースの推薦を表示'):
    recommendations = user_based_recommendation(selected_user)
    st.write(f"{selected_user}さんへのおすすめレストラン:")
    for restaurant in recommendations:
        restaurant_info = df[df['店名'] == restaurant].iloc[0]
        st.write(f"- {restaurant_info['店名']} ({restaurant_info['タグ']}) - {restaurant_info['店舗最寄り駅']}駅")
        st.write(f"  URL: {restaurant_info['URL']}")
    
    st.markdown("""
    <details>
    <summary>レコメンドの仕組み</summary>
    <p style="font-size: 0.9em; color: #666;">
    このレコメンドは、あなたと似たお店の好みを持つユーザーが推薦しているレストランを表示しています。
    具体的には、コサイン類似度を使用してユーザー間の類似性を計算し、最も似ている上位5人のユーザーが推薦したレストランを抽出しています。
    </p>
    </details>
    """, unsafe_allow_html=True)

# アイテムベースの推薦
if st.button('アイテムベースの推薦を表示'):
    recommendations = item_based_recommendation(selected_restaurant)
    st.write(f"{selected_restaurant}を好きな人におすすめのレストラン:")
    for restaurant in recommendations:
        restaurant_info = df[df['店名'] == restaurant].iloc[0]
        st.write(f"- {restaurant_info['店名']} ({restaurant_info['タグ']}) - {restaurant_info['店舗最寄り駅']}駅")
        st.write(f"  URL: {restaurant_info['URL']}")
    
    st.markdown("""
    <details>
    <summary>レコメンドの仕組み</summary>
    <p style="font-size: 0.9em; color: #666;">
    このレコメンドは、選択されたレストランと似た特徴を持つ他のレストランを表示しています。
    具体的には、コサイン類似度を使用してレストラン間の類似性を計算し、最も似ている上位5つのレストランを抽出しています。
    類似性は、それぞれのレストランを推薦しているユーザーの重なりに基づいて判断されます。
    </p>
    </details>
    """, unsafe_allow_html=True)