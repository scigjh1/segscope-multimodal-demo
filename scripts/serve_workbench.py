"""Serve the upstream VolView UI and proxy an existing local GPU runtime."""
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, unquote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import argparse
import math
import array

ROOT = Path(__file__).resolve().parents[1]


def public_volume():
    """A geometric phantom: no patient, image archive or trained model involved."""
    values = array.array('h')
    for z in range(48):
        for y in range(96):
            for x in range(96):
                body = ((x-48)/35)**2 + ((y-48)/29)**2 < 1
                organ = ((x-39)/14)**2 + ((y-43)/15)**2 + ((z-24)/19)**2 < 1
                spine = ((x-48)/6)**2 + ((y-67)/6)**2 < 1
                value = -900 if not body else 30 + int(12*math.sin(x/8)*math.cos(y/9))
                if organ: value = 135
                if spine: value = 850
                values.append(value)
    header = ('NRRD0005\ntype: short\ndimension: 3\nsizes: 96 96 48\n'
              'space: left-posterior-superior\nspace directions: (1,0,0) (0,1,0) (0,0,2)\n'
              'space origin: (0,0,0)\nencoding: raw\nendian: little\n\n').encode()
    return header + values.tobytes()


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        clean = unquote(urlsplit(path).path)
        if clean.startswith('/volview/'):
            base, relative = ROOT/'web/volview/dist', clean[len('/volview/'):]
        else:
            base, relative = ROOT/'static', clean.lstrip('/')
        target = (base/relative).resolve()
        if not target.is_relative_to(base.resolve()):
            return str(base/'__invalid__')
        return str(target)

    def proxy(self):
        origin = self.headers.get('Origin')
        if origin and urlsplit(origin).netloc != self.headers.get('Host'):
            self.send_error(403, 'Cross-origin access disabled')
            return
        length = int(self.headers.get('Content-Length', '0'))
        if length > 128 * 1024 * 1024:
            self.send_error(413)
            return
        body = self.rfile.read(length) if length else None
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in {'host', 'origin', 'connection', 'content-length', 'accept-encoding'}}
        request = Request(self.server.runtime+self.path, data=body, headers=headers, method=self.command)
        try:
            response = urlopen(request, timeout=180)
        except HTTPError as error:
            response = error
        except URLError:
            self.send_error(502, 'GPU runtime is unavailable')
            return
        with response:
            data = response.read()
            self.send_response(response.status)
            self.send_header('Content-Type', response.headers.get('Content-Type', 'application/octet-stream'))
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def do_GET(self):
        if urlsplit(self.path).path == '/':
            self.send_response(302)
            self.send_header('Location', '/volview/?urls=/volview/public-demo.nrrd')
            self.end_headers()
        elif self.path.startswith('/api/'):
            self.proxy()
        elif urlsplit(self.path).path == '/volview/public-demo.nrrd':
            data = public_volume()
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            super().do_GET()

    def do_POST(self):
        if self.path.startswith('/api/'):
            self.proxy()
        else:
            self.send_error(404)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=4200)
    parser.add_argument('--runtime', default='http://127.0.0.1:4186')
    args = parser.parse_args()
    if not (ROOT/'web/volview/dist/index.html').is_file():
        raise SystemExit('Build web/volview first: npm ci && npm run build')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.runtime = args.runtime.rstrip('/')
    print(f'SegScope / VolView: http://127.0.0.1:{args.port}')
    server.serve_forever()
