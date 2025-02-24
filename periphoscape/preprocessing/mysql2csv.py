import os
import MySQLdb
import csv
import sys

class MySql2csv():
    def __init__(self, host, user, passwd, dbname):
        self.host = host
        self.user = user
        self.passwd = passwd
        self.dbname = dbname

    def open(self):
        self.conn = MySQLdb.connect(
            user=self.user,
            passwd=self.passwd,
            host=self.host,
            db=self.dbname
        )
        self.cursor = self.conn.cursor()

    def close(self):
        self.cursor.close()
        self.conn.close()

    def save_pages_as_csv(self, filename, ndata=None):
        with open(filename, "w") as f:
            writer = csv.writer(f)
            sql = 'select page_id, page_namespace, page_title, page_is_redirect from page'
            self.cursor.execute(sql)
            c = 0
            for r in self.cursor.fetchall():
                writer.writerow([r[0], r[1], r[2].decode('utf-8'), r[3]])
                c += 1
                if ndata and c == ndata:
                    break

    def save_pagelinks_as_csv(self, filename, ndata=None):
        with open(filename, "w") as f:
            writer = csv.writer(f)
            sql = 'select pl_from, pl_namespace, pl_title, pl_from_namespace from pagelinks'
            self.cursor.execute(sql)
            c = 0
            for r in self.cursor.fetchall():
                try:
                    writer.writerow([r[0], r[1], r[2].decode('utf-8'), r[3]])
                except:
                    print(f"Exeption: ({c}) {r[0]} {r[1]}")
                    continue
                c += 1
                if ndata and c == ndata:
                    break

    def save_categories_as_csv(self, filename, ndata=None):
        with open(filename, "w") as f:
            writer = csv.writer(f)
            sql = 'select cat_id, cat_title from category'
            self.cursor.execute(sql)
            c = 0
            for r in self.cursor.fetchall():
                writer.writerow([r[0], r[1].decode('utf-8')])
                c += 1
                if ndata and c == ndata:
                    break

    def save_categorylinks_as_csv(self, filename, ndata=None):
        with open(filename, "w") as f:
            writer = csv.writer(f)
            sql = 'select cl_from, cl_to, cl_type from categorylinks'
            self.cursor.execute(sql)
            c = 0
            for r in self.cursor.fetchall():
                writer.writerow([r[0], r[1].decode('utf-8'), r[2].decode('utf-8')])
                c += 1
                if ndata and c == ndata:
                    break

    def save_as_csv(self, table, filename):
        if table == 'category':
            self.save_categories_as_csv(filename)
        elif table == 'categorylinks':
            self.save_categorylinks_as_csv(filename)
        elif table == 'page':
            self.save_pages_as_csv(filename)
        elif table == 'pagelinks':
            self.save_pagelinks_as_csv(filename)


if __name__ == '__main__':
    SQL_HOST = 'localhost'
    SQL_USER = ''
    SQL_PASS = ''
    DB_TIMESTAMP = '20240220'
    DB_NAME = f'jawiki_{DB_TIMESTAMP}'
    CSV_FILES = {
        'category' : f'jawiki-{DB_TIMESTAMP}-category.csv',
        'categorylinks' : f'jawiki-{DB_TIMESTAMP}-categorylinks.csv',
        'page' : f'jawiki-{DB_TIMESTAMP}-page.csv',
        'pagelinks' : f'jawiki-{DB_TIMESTAMP}-pagelinks.csv'
        }
    mysql = MySql2csv(SQL_HOST, SQL_USER, SQL_PASS, DB_NAME)
    mysql.open()
    for table in ['category', 'categorylinks', 'page', 'pagelinks']:
        print(f'Saving {table} as csv...')
        mysql.save_as_csv(table, CSV_FILES[table])
    mysql.close()
