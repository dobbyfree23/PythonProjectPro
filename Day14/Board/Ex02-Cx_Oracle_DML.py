import cx_Oracle

# 데이터베이스 접속 정보
username = 'hr'
password = 'hr'
host = 'localhost'
port = '1521'
sid = 'xe'
dsn = cx_Oracle.makedsn(host, port, sid)

# 데이터 삽입
conn = cx_Oracle.connect(username, password, dsn)
cur = conn.cursor()
sql = "INSERT INTO employees (employee_id, first_name, last_name, email, hire_date, job_id, salary, department_id) VALUES (:1, :2, :3, :4, :5, :6, :7, :8)"
cur.execute(sql, (1001, 'John', 'Doe', 'johndoe@example.com', '2022-03-01', 'IT_PROG', 5000, 90))
conn.commit()
cur.close()
conn.close()

# 데이터 수정
conn = cx_Oracle.connect(username, password, dsn)
cur = conn.cursor()
sql = "UPDATE employees SET salary=:1 WHERE employee_id=:2"
cur.execute(sql, (5500, 1001))
conn.commit()
cur.close()
conn.close()

# 데이터 삭제
conn = cx_Oracle.connect(username, password, dsn)
cur = conn.cursor()
sql = "DELETE FROM employees WHERE employee_id=:1"
cur.execute(sql, (1001,))
conn.commit()
cur.close()
conn.close()