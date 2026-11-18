"""EmbodiedLab client of EmbodiedLab's public device API; no simulator imports.

Use a stable request_id when retrying an uncertain call. The benchmark retains
the original receipt, so a transport retry cannot execute an action twice.
"""
import json
import urllib.request
import uuid


class EmbodiedLabTransport:
    def __init__(self, url, *, timeout=180):
        self.url = url.rstrip('/')
        self.timeout = timeout

    def task(self):
        with urllib.request.urlopen(self.url+'/v1/task', timeout=self.timeout) as response:
            return json.load(response)

    def call(self, name, arguments, *, request_id=None):
        body = {'request_id': request_id or uuid.uuid4().hex,
                'name': name, 'arguments': arguments}
        request = urllib.request.Request(self.url+'/v1/call',
            data=json.dumps(body, allow_nan=False).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        if result.get('ok') is not True:
            raise RuntimeError('Benchmark rejected command: '+str(result.get('message') or result.get('error')))
        return result['result']
