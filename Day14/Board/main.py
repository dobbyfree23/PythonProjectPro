import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QListWidget, QTextEdit, QAction, QMenuBar, QPushButton, \
    QMessageBox, QListWidgetItem
import cx_Oracle

class Board(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.load()

    def initUI(self):
        self.setGeometry(300, 300, 500, 500)
        self.setWindowTitle('Board')

        self.listWidget = QListWidget(self)
        self.listWidget.setGeometry(50, 50, 200, 350)
        self.listWidget.itemClicked.connect(self.show_detail)

        self.textEdit_title = QTextEdit(self)
        self.textEdit_title.setGeometry(270, 50, 200, 30)
        self.textEdit_content = QTextEdit(self)
        self.textEdit_content.setGeometry(270, 90, 200, 350)

        self.btn_add = QPushButton('글쓰기', self)
        self.btn_add.setGeometry(50, 420, 80, 30)
        self.btn_add.clicked.connect(self.add)

        self.btn_edit = QPushButton('수정', self)
        self.btn_edit.setGeometry(140, 420, 80, 30)
        self.btn_edit.clicked.connect(self.edit)

        self.btn_delete = QPushButton('삭제', self)
        self.btn_delete.setGeometry(230, 420, 80, 30)
        self.btn_delete.clicked.connect(self.delete)

        self.btn_refresh = QPushButton('조회', self)
        self.btn_refresh.setGeometry(320, 420, 80, 30)
        self.btn_refresh.clicked.connect(self.load)

        self.show()

    def load(self):
        # Oracle 데이터베이스에 접속
        conn = cx_Oracle.connect('hr/hr@localhost/xe')
        print("Connected to Oracle Database")

        # board 테이블에서 게시글 조회
        cur = conn.cursor()
        cur.execute('SELECT id, title, content FROM board ORDER BY id DESC')
        data = cur.fetchall()

        # QListWidget 초기화
        self.listWidget.clear()

        # 데이터베이스에서 가져온 게시글을 QListWidget에 추가
        for d in data:
            item = QListWidgetItem(d[1])
            item.setData(0, d[0])
            self.listWidget.addItem(item)

        # 테이블 조회 후 커서와 연결 종료
        cur.close()
        conn.close()

    def show_detail(self, item):
        # QListWidgetItem에서 id를 가져와 해당 게시글 상세 정보 조회
        id = item.data(0)

        # Oracle 데이터베이스에 접속
        conn = cx_Oracle.connect('hr/hr@localhost/xe')
        print("Connected to Oracle Database")

        # board 테이블에서 게시글 조회
        cur = conn.cursor()
        cur.execute('SELECT title, content FROM board WHERE id=:id', {'id': id})
        data = cur.fetchone()

        # QTextEdit 위젯에 게시글 상세 정보 표시
        self.textEdit_title.setText(data[0])
        self.textEdit_content.setText(data[1])

        # 커서와 연결 종료
        cur.close()
        conn.close()

    def add(self):
        # QTextEdit 위젯에서 입력된 제목과 내용을 가져오기
        title = self.textEdit_title.toPlainText()
        content = self.textEdit_content.toPlainText()

        # Oracle 데이터베이스에 접속
        conn = cx_Oracle.connect('hr/hr@localhost/xe')
        print("Connected to Oracle Database")

        # board 테이블에 새로운 게시글 추가
        cur = conn.cursor()
        cur.execute('INSERT INTO board (title, content) VALUES (:title, :content)',
                    {'title': title, 'content': content})
        conn.commit()

        # 커서와 연결 종료
        cur.close()
        conn.close()

        # 게시글 목록 다시 로드
        self.load()

    def edit(self):
        # 현재 선택된 QListWidgetItem에서 id를 가져오기
        item = self.listWidget.currentItem()
        id = item.data(0)

        # QTextEdit 위젯에서 입력된 제목과 내용을 가져오기
        title = self.textEdit_title.toPlainText()
        content = self.textEdit_content.toPlainText()

        # Oracle 데이터베이스에 접속
        conn = cx_Oracle.connect('hr/hr@localhost/xe')
        print("Connected to Oracle Database")

        # board 테이블에서 해당 게시글 수정
        cur = conn.cursor()
        cur.execute('UPDATE board SET title=:title, content=:content WHERE id=:id',
                    {'title': title, 'content': content, 'id': id})
        conn.commit()

        # 커서와 연결 종료
        cur.close()
        conn.close()

        # 게시글 목록 다시 로드
        self.load()

    def delete(self):
        # 현재 선택된 QListWidgetItem에서 id를 가져오기
        item = self.listWidget.currentItem()
        id = item.data(0)

        # Oracle 데이터베이스에 접속
        conn = cx_Oracle.connect('hr/hr@localhost/xe')
        print("Connected to Oracle Database")

        # board 테이블에서 해당 게시글 삭제
        cur = conn.cursor()
        cur.execute('DELETE FROM board WHERE id=:id', {'id': id})
        conn.commit()

        # 커서와 연결 종료
        cur.close()
        conn.close()

        # 게시글 목록 다시 로드
        self.load()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    board = Board()
    sys.exit(app.exec_())
