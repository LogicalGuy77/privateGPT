# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

datas = [('src/private_gpt_app/ui/styles.qss', 'ui'), ('src/private_gpt_app/ui/styles_modern.qss', 'ui'), ('models/Qwen2.5-3B-Instruct-AWQ', 'models/Qwen2.5-3B-Instruct-AWQ')]
binaries = [('/lib/x86_64-linux-gnu/libcuda.so.1', 'nvidia/cuda/lib'), ('/lib/x86_64-linux-gnu/libcuda.so', 'nvidia/cuda/lib')]
hiddenimports = ['vllm', 'torch', 'PyQt6', 'qdrant_client', 'sentence_transformers', 'transformers', 'rank_bm25', 'unittest.mock', 'private_gpt_app.backend.vllm_service', 'private_gpt_app.backend.session_manager', 'private_gpt_app.backend.router', 'private_gpt_app.rag.vector_store', 'private_gpt_app.rag.embeddings']
datas += copy_metadata('torch')
datas += copy_metadata('nvidia-cublas-cu12')
datas += copy_metadata('nvidia-cuda-cupti-cu12')
datas += copy_metadata('nvidia-cuda-nvrtc-cu12')
datas += copy_metadata('nvidia-cuda-runtime-cu12')
datas += copy_metadata('nvidia-cudnn-cu12')
datas += copy_metadata('nvidia-cufft-cu12')
datas += copy_metadata('nvidia-curand-cu12')
datas += copy_metadata('nvidia-cusolver-cu12')
datas += copy_metadata('nvidia-cusparse-cu12')
datas += copy_metadata('nvidia-nccl-cu12')
datas += copy_metadata('nvidia-nvjitlink-cu12')
datas += copy_metadata('nvidia-nvtx-cu12')
datas += copy_metadata('triton')
tmp_ret = collect_all('vllm')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torch')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('transformers')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('sentence_transformers')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['src/private_gpt_app/main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PrivateGPT',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PrivateGPT',
)
