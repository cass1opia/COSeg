import multiprocessing as mp
from multiprocessing.managers import SyncManager
from pathlib import Path
from queue import Queue
from typing import Any, Optional

import torch


# gui bridges, need to be accessed by any script other than the gui scripts themselves
epoch_bridge: Any = None
progress_bridge: Any = None
train_settings_bridge: Any = None
predict_results_bridge: Any = None

# logging
logging_manager: Optional[SyncManager] = None
logging_process: Optional[mp.process.BaseProcess] = None
logging_queue: Optional[Queue] = None

# global shared memory object for predict progress reporting
finished_files_lock: Any = None
finished_files: Any = None

# other globals
train_metric_export_file: Optional[Path] = None
torch_device = torch.device('cpu')
cuda_available = False
deterministic_mode = "none"
