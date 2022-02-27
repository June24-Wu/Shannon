# platform_ops

投研平台因子自动维护，自动更新，算子拉取等功能。


## API包括三部分
* 钉钉
* logger
* mail

## 调用方式
### 钉钉

```python
from tools.ding import Ding

ding = Ding('secret_key', 'token')
ding.send_ding('title', 'message', to=['phone num1', 'phone num2', '...'])
```

### logger

```python

from tools import logger

log = logger.get_logger('who call the logger')
log.info('the info content of log')
log.error('the error content of log')
```

### mail

```python

from tools import mail

mail.send_mail('mail title or subject', 'the content of mail', ['receiver1', 'receiver2', '...'])
```
