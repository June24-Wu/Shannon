# python 3.8
import time
import hmac
import hashlib
import base64
import urllib.parse
from urllib.request import urlopen, Request
import json
from DevTools.config.config import config
import ssl


class Ding(object):
    def __init__(self, secret_key, access_token):
        self.secret_key = secret_key
        self.access_token = access_token

    def get_sign(self, timestamp):
        secret = self.secret_key
        secret_enc = secret.encode('utf-8')
        string_to_sign = '{}\n{}'.format(timestamp, secret)
        string_to_sign_enc = string_to_sign.encode('utf-8')
        hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
        sign = base64.b64encode(hmac_code)
        return sign

    @staticmethod
    def get_mark_down(title, msg, to):
        r = f"# [{config.ENV_NAME}]{title} \n"
        for line in msg.split("\n"):
            r += "###### " + line + " \n"
        r += "\n"

        for t in to:
            r += "@" + t + "\n"

        return r

    def send_ding(self, title, msg, to=None):
        if to is None:
            to = []

        elif not isinstance(to, list):
            return

        url = "https://oapi.dingtalk.com/robot/send"
        timestamp = str(round(time.time() * 1000))
        sign = self.get_sign(timestamp)

        _params = {
            "access_token": self.access_token,
            "timestamp": str(timestamp),
            "sign": sign
        }

        url += "?" + urllib.parse.urlencode(_params)

        _to = to

        params = {
            "msgtype": "markdown",
            "markdown": {
                "title": f"[{config.ENV_NAME}]{title}",
                "text": self.get_mark_down(title, msg, _to)
            },
            "at": {
                "atMobiles": _to,
                "isAtAll": False
            }
        }

        if len(params["at"]["atMobiles"]) == 0:
            params["at"]["isAtAll"] = True

        for i in range(0, 3):
            try:
                req = Request(url, json.dumps(params).encode(), headers={"Content-Type": "application/json"})
                ssl_content = ssl.SSLContext()
                urlopen(req, context=ssl_content).read()
                break
            except:
                time.sleep(2)
