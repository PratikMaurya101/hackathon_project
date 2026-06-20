import sqlite3
import pandas as pd


class ClimateDatabase:

    def __init__(self, db_path="dbms/operating_data.db"):
        self.db_path = db_path

    def _connect(self):
        return sqlite3.connect(self.db_path)

    # --------------------------------------------------
    # Schema Creation
    # --------------------------------------------------

    def create_tables(self):
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS operating_data (

            id TEXT,

            last_seen_at TEXT NOT NULL,

            status_temperature_in_celsius REAL,
            status_humidity_in_percent REAL,

            status_target_temperature_in_celsius REAL,
            status_temperature_outside_in_celsius REAL,

            status_carbon_dioxide_in_ppm INTEGER,

            status_air_flow_supply_in_percent REAL,
            status_air_flow_return_in_percent REAL,

            status_is_heating_required INTEGER,
            status_is_cooling_required INTEGER,
                       
            baseline_power_con REAL,
            model_power_con REAL,

            op_mode INTEGER,                     
                      
            PRIMARY KEY (id, last_seen_at)
        )
        """)

        conn.commit()
        conn.close()

    # --------------------------------------------------
    # Insert DataFrame
    # --------------------------------------------------

    def insert_dataframe(self, df):
        conn = self._connect()

        df.to_sql(
            "operating_data",
            conn,
            if_exists="append",
            index=False
        )

        conn.close()

    # --------------------------------------------------
    # Read Latest Record
    # --------------------------------------------------

    def get_latest_reading(self):

        conn = self._connect()

        query = """
        SELECT *
        FROM operating_data
        ORDER BY last_seen_at DESC
        LIMIT 1
        """

        df = pd.read_sql(query, conn)

        conn.close()

        if df.empty:
            return None

        return df.iloc[0]

    # --------------------------------------------------
    # Last N Records
    # --------------------------------------------------

    def get_last_n_readings(self, n=100):

        conn = self._connect()

        query = f"""
        SELECT *
        FROM operating_data
        ORDER BY last_seen_at DESC
        LIMIT {n}
        """

        df = pd.read_sql(query, conn)

        conn.close()

        return df

    # --------------------------------------------------
    # Get All Data
    # --------------------------------------------------

    def get_all_readings(self):

        conn = self._connect()

        df = pd.read_sql(
            "SELECT * FROM operating_data",
            conn
        )

        conn.close()

        return df
    
    # --------------------------------------------------
    # Get Data in Range
    # --------------------------------------------------

    def get_data_between(self, start_datetime, end_datetime) : 
        
        conn = self._connect()

        query = """
        SELECT *
        FROM operating_data
        WHERE last_seen_at BETWEEN ? AND ?
        ORDER BY last_seen_at ASC
        """

        df = pd.read_sql_query(
            query,
            conn,
            params=(
                start_datetime,
                end_datetime
            )
        )

        conn.close()

        return df

    # --------------------------------------------------
    # Get Heating Data (not necesary I think)
    # --------------------------------------------------

    def get_heating_data(self):

        conn = self._connect()

        query = """
        SELECT *
        FROM operating_data
        WHERE strftime('%m', last_seen_at) = '03'
        """

        df = pd.read_sql(query, conn)

        conn.close()

        return df

    # --------------------------------------------------
    # Get Cooling Data (not necesary I think)
    # --------------------------------------------------

    def get_cooling_data(self):

        conn = self._connect()

        query = """
        SELECT *
        FROM operating_data
        WHERE strftime('%m', last_seen_at) = '05'
        """

        df = pd.read_sql(query, conn)

        conn.close()

        return df

    # --------------------------------------------------
    # Clear Table
    # --------------------------------------------------

    def clear_table(self):

        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM operating_data"
        )

        conn.commit()
        conn.close()

    # --------------------------------------------------
    # Row Count
    # --------------------------------------------------

    def get_row_count(self):

        conn = self._connect()

        query = """
        SELECT COUNT(*) as count
        FROM operating_data
        """

        df = pd.read_sql(query, conn)

        conn.close()

        return int(df["count"][0]) 
    
    # --------------------------------------------------
    # Add new Attribute
    # --------------------------------------------------

    def add_column(
        self,
        column_name,
        column_type="REAL"
    ):

        schema = self.show_schema()

        if column_name in schema["name"].values:

            print(
            f"{column_name} already exists."
            )

            return

        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute(
            f"""
            ALTER TABLE operating_data
            ADD COLUMN {column_name}
            {column_type}
            """
        )

        conn.commit()
        conn.close()

    # --------------------------------------------------
    # Show Schema
    # --------------------------------------------------

    def show_schema(self):

        conn = self._connect()

        query = """
        PRAGMA table_info(
            operating_data
        )
        """

        df = pd.read_sql(
        query,
        conn
        )

        conn.close()

        return df