import json
from pathlib import Path
import sqlite3
import time
import uuid


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS cases(id TEXT PRIMARY KEY,meta TEXT);
              CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,case_id TEXT,meta TEXT,created REAL);
              CREATE TABLE IF NOT EXISTS reviews(id INTEGER PRIMARY KEY,run_id TEXT,decision TEXT,note TEXT,created REAL);
              CREATE TABLE IF NOT EXISTS error_pool(id INTEGER PRIMARY KEY,run_id TEXT,taxonomy TEXT,created REAL);''')

    def connection(self):
        return sqlite3.connect(self.root/'workflow.sqlite', timeout=30)

    def new_case(self, meta):
        case_id = 'CASE-'+uuid.uuid4().hex[:12]
        directory = self.root/case_id
        directory.mkdir()
        meta = {**meta, 'id': case_id}
        with self.connection() as db:
            db.execute('INSERT INTO cases VALUES(?,?)',(case_id,json.dumps(meta)))
        return meta, directory

    def cases(self):
        with self.connection() as db:
            return [json.loads(x[0]) for x in db.execute('SELECT meta FROM cases ORDER BY rowid DESC')]

    def case(self, case_id):
        with self.connection() as db:
            row = db.execute('SELECT meta FROM cases WHERE id=?',(case_id,)).fetchone()
        if not row:
            raise ValueError('Unknown case')
        return json.loads(row[0]), self.root/case_id

    def new_run(self, case_id, meta):
        run_id = 'RUN-'+uuid.uuid4().hex[:12]
        directory = self.root/run_id
        directory.mkdir()
        meta = {**meta,'id':run_id,'case_id':case_id,'review':'pending'}
        with self.connection() as db:
            db.execute('INSERT INTO runs VALUES(?,?,?,?)',(run_id,case_id,json.dumps(meta),time.time()))
        return meta, directory

    def runs(self):
        with self.connection() as db:
            return [json.loads(x[0]) for x in db.execute('SELECT meta FROM runs ORDER BY created DESC LIMIT 100')]

    def run(self, run_id):
        with self.connection() as db:
            row=db.execute('SELECT meta FROM runs WHERE id=?',(run_id,)).fetchone()
        if not row:
            raise ValueError('Unknown run')
        return json.loads(row[0]), self.root/run_id

    def review(self, run_id, decision, note='', taxonomy=None):
        if decision not in ('accept','edit','reject'):
            raise ValueError('Decision must be accept, edit or reject')
        allowed = {'small_lesion_miss','boundary_deviation','remote_fp','low_contrast','other'}
        if taxonomy and taxonomy not in allowed:
            raise ValueError('Unknown failure taxonomy')
        meta, _ = self.run(run_id)
        meta['review']=decision
        with self.connection() as db:
            db.execute('UPDATE runs SET meta=? WHERE id=?',(json.dumps(meta),run_id))
            db.execute('INSERT INTO reviews(run_id,decision,note,created) VALUES(?,?,?,?)',
                       (run_id,decision,note[:2000],time.time()))
            if decision in ('edit','reject'):
                db.execute('INSERT INTO error_pool(run_id,taxonomy,created) VALUES(?,?,?)',
                           (run_id,taxonomy or 'other',time.time()))
        return meta

    def hard_cases(self):
        with self.connection() as db:
            rows = db.execute('SELECT run_id,taxonomy,created FROM error_pool ORDER BY created DESC LIMIT 100').fetchall()
        return [{'run_id':x[0],'taxonomy':x[1],'created':x[2]} for x in rows]
