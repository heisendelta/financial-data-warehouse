from sqlalchemy import create_engine
import pandas as pd

DATABASE_URL = (
    "postgresql+psycopg2://quant:quant@localhost:5432/warehouse"
)

engine = create_engine(DATABASE_URL)

df = pd.read_sql("SELECT * FROM assets", engine)

print(df)
