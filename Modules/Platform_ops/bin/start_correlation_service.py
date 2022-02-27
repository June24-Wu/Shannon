from feature.ops import Distribute
from Platform_ops.config.config import configs

dist = Distribute(configs, logger=None)
# dist.clear_queue(dist.QUEUE_KEY_PRODUCER_P)
print('Correlation service is ready...')
dist.distribute_correlation_calculator()
