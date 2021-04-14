""" db와의 연결을 계속 유지할까? 말까? 말자."""

import sqlite3


def db_connect(dbname="./database/nopain.db"):
    conn = sqlite3.connect(dbname)
    return conn


def execute_query(query):
    conn = db_connect()

    cur = conn.cursor()
    cur.execute(query)

    conn.commit()
    conn.close()


def sample_strategy():
    """ PENDING되지 않은 전략 중, """
    conn = db_connect()
    query = "SELECT strategy, status, cnt, v_avg_score FROM STRATEGY WHERE status is not 'PENDING'"  # 오름차순

    cur = conn.cursor()
    cur.execute(query)

    row_tuple = cur.fetchall()  # fetchone, fetchmany

    conn.close()

    row_tuple = sorted(row_tuple, key=lambda x: (x[2], -x[3]))
    strategy, status, cnt, v_avg_score = row_tuple[0]

    return strategy, status, cnt, v_avg_score
