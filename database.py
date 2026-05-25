import sqlite3

# =========================
# CONNECT DATABASE
# =========================

conn = sqlite3.connect("monuments.db")

cursor = conn.cursor()

# =========================
# CREATE TABLE
# =========================

cursor.execute("""

CREATE TABLE IF NOT EXISTS monuments (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    name TEXT UNIQUE,

    history TEXT,

    dynasty TEXT,

    construction_period TEXT,

    architecture TEXT,

    unesco_status TEXT,

    tourism_facts TEXT

)

""")

# =========================
# INSERT DATA
# =========================

monuments_data = [

    (
        "Taj Mahal",

        "The Taj Mahal was commissioned by Mughal emperor Shah Jahan in memory of his wife Mumtaz Mahal.",

        "Mughal Empire",

        "1632–1653",

        "Mughal Architecture",

        "UNESCO World Heritage Site since 1983",

        "Over 7 million tourists visit annually."
    ),

    (
        "Qutub Minar",

        "Qutub Minar is one of the tallest brick minarets in the world.",

        "Delhi Sultanate",

        "1192–1220",

        "Indo-Islamic Architecture",

        "UNESCO World Heritage Site since 1993",

        "Height: 72.5 meters."
    ),

    (
        "Gateway of India",

        "Built to commemorate the visit of King George V and Queen Mary.",

        "British Raj",

        "1911–1924",

        "Indo-Saracenic Architecture",

        "Not a UNESCO site",

        "Located on the Mumbai waterfront."
    ),

    (
        "Charminar",

        "Charminar was built by Muhammad Quli Qutb Shah.",

        "Qutb Shahi Dynasty",

        "1591",

        "Indo-Islamic Architecture",

        "Tentative UNESCO List",

        "Major symbol of Hyderabad."
    ),

    (
        "Sun Temple Konark",

        "The temple is designed as a gigantic stone chariot dedicated to Surya.",

        "Eastern Ganga Dynasty",

        "13th Century",

        "Kalinga Architecture",

        "UNESCO World Heritage Site since 1984",

        "Known for intricate stone carvings."
    )

]

# =========================
# INSERT IGNORE DUPLICATES
# =========================

cursor.executemany("""

INSERT OR IGNORE INTO monuments (

    name,
    history,
    dynasty,
    construction_period,
    architecture,
    unesco_status,
    tourism_facts

)

VALUES (?, ?, ?, ?, ?, ?, ?)

""", monuments_data)

# =========================
# SAVE
# =========================

conn.commit()

conn.close()

print("Database Created Successfully!")