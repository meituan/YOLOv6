#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
from pathlib import Path

def increment_name(path, master_process):
    "increase save directory's id"
    path = Path(path)
    sep = ''
    if path.exists() and master_process:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(1, 9999):
            p = f'{path}{sep}{n}{suffix}'
            if not os.path.exists(p):
                break
        path = Path(p)
    return path