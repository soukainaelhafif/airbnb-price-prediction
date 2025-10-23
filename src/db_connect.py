import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    user="souka",
    password="password",
    dbname="airbnb"
)

print("Verbindung zu PostgreSQL erfolgreich!")
conn.close()