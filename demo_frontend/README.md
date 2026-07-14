# 多模态语音系统离线展示前端

该前端只使用 Python 标准库和浏览器原生 HTML/CSS/JavaScript。模型推理继续运行在 C1-C5 原有 Conda 环境中，不需要服务器联网，也不需要安装 npm、Gradio、FastAPI 等新依赖。

## 1. 目录要求

将 `demo_frontend/` 与以下目录放在同一个项目根目录：

```text
group-working-main/
  C1_audio_processing/
  C2_ASR/
  C3_cascade/
  C4_e2e_package/
  C5_TTS/
  demo_frontend/
```

前端通过 WAV 和 JSON 文件连接模块，不会把四套模型环境合并到一起。

## 2. 创建服务器配置

在服务器项目根目录执行：

```bash
cp demo_frontend/config.example.json demo_frontend/config.json
```

编辑 `demo_frontend/config.json`，重点确认：

- `python.c1`: `/opt/conda/envs/speech_zxy/bin/python`
- `python.c2`、`python.c3`: `/opt/conda/envs/speech/bin/python`
- `python.c4`: `/opt/conda/envs/speech_tcx/bin/python`
- `python.c5`: `/opt/conda/envs/cosyvoice/bin/python`
- `models.asr_tiny`: `/root/siton-tmp/assignment_C/model/whisper-tiny`
- `models.asr_large_v3`: `/root/siton-tmp/assignment_C/model/whisper-large-v3`
- `models.*`: 其他服务器真实模型目录
- `sample_dir`: 只包含答辩样例 WAV 的服务器目录
- `jobs_dir`: 每次演示的日志与产物目录

`sample_dir` 可以是项目相对路径或绝对路径。前端递归扫描其中的 `.wav` 文件，不接受浏览器上传。

## 3. 离线检查

```bash
/opt/conda/envs/speech/bin/python demo_frontend/start_server.py \
  --config demo_frontend/config.json \
  --dry-run
```

输出中的 `ready` 为 `true` 时，表示 Python、脚本、模型和样例均存在。该命令不会导入 torch、transformers 或加载模型。

## 4. 启动服务

```bash
/opt/conda/envs/speech/bin/python demo_frontend/start_server.py \
  --config demo_frontend/config.json \
  --host 127.0.0.1 \
  --port 7860
```

保持终端运行。服务只监听服务器本机 `127.0.0.1`，不会直接暴露到校园网。

## 5. 本地电脑建立 SSH 隧道

在本地 PowerShell 运行：

```powershell
ssh -p <SSH_PORT> -N -L 127.0.0.1:18080:127.0.0.1:7860 <USER>@<SERVER_HOST>
```

浏览器打开：

```text
http://localhost:18080
```

若本地 `18080` 被占用，只需更换 `-L` 左侧端口，例如 `28080:127.0.0.1:7860`；服务器端仍保持 `7860`。

## 6. 开关规则

- VAD 开启：使用现有 Silero VAD 和动态 chunk。
- VAD 关闭：仅允许不超过约 30 秒的短音频，C2 直接生成一个固定 chunk。
- ASR 模型：只影响 C2 `--model`，可在 Whisper tiny 与 Whisper large-v3 之间切换。
- Pyannote 开启：添加 C2 `--diarize` 并使用本地模型；关闭时不加载说话人模型。
- 纠错模型开启：C2 使用 `--nbest 5 --beam_size 20`，C3 加载本地纠错模型。
- 纠错模型关闭：C2 使用 `--nbest 1 --beam_size 1`，C3 使用 ASR top-1，仍继续本地翻译和 C5 合成。
- C4 对照开启：在 C1-C2-C3-C5 完成后串行运行，避免同时占用显存。

## 7. 产物与日志

每次运行创建：

```text
demo_frontend/jobs/<job_id>/
  request.json
  status.json
  events.jsonl
  pipeline.log
  c1/
  c2/
  c3/
  c4/
  c5/
```

浏览器刷新后仍可通过 job 状态文件读取结果。停止任务会终止当前子进程并保留已完成阶段的产物。

## 8. 测试

无需模型：

```bash
python3 -m unittest discover -s demo_frontend/tests -v
```

C2/C3 兼容开关：

```bash
PYTHONPATH=C2_ASR/code python3 -m unittest C2_ASR.code.test_no_vad_short_audio C2_ASR.code.test_c2_entrypoint -v
```

```bash
cd C3_cascade/code
PYTHONPATH=. python3 -m unittest c3.test_disable_correction test_c3_cli_entrypoint -v
```
