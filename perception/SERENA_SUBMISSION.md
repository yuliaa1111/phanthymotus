# Serena 小优榜 JP6 提交候选

## 当前结论

2026-11-11：用户已要求先提交一版；三并发问题单独存档在项目根目录 `三并发.md`。
首版选用已通过音频回归的48/48分块混合GPU候选。当前推进到GitHub登录授权，
完成个人公开fork和代码发布后才能填写真实commit提交榜单。

单实例完整链路已验证；三套完整服务可以同时运行，但当前三并发首段延迟为
1.77–2.58 秒，**未达到 1 秒准入线**，此版本不是已完成验收的正式提交。
模型仅支持 JP6 / aarch64 / Python 3.10 / TensorRT 10.4.0。

## 部署

沿用官方 JP6 perception 镜像的 TensorRT、CUDA、ONNX Runtime 和 ROS 环境。
将本仓库 `perception/` 作为服务代码，设置：

```text
CONFIG_PATH=/app/perception/config.serena-jp6.yaml
NVIDIA_TF32_OVERRIDE=0
MCP_PORT=<分配的 HTTP 端口>
WS_PORT=<分配的 WebSocket 端口>
```

`/app/perception` 应替换为实际代码目录；配置默认模型目录 `/models/serena` 需要可写。
在空模型目录首次启动，插件读取 `perception/serena-release.json`，从模型 HTTP
服务器下载约 444 MB 压缩包，校验压缩包及包内 7437 个文件，再原子安装。
包内含前端依赖及英文语言模型，无需在线 pip 下载。空目录下载验收约 93 秒；
模型加载和预热在服务就绪前完成，不计作 warm 请求的首段延迟。

推荐使用 `config.serena-jp6.yaml`。`config.serena-tuning.yaml` 是失败的并发调优
实验参数，存在供音间隔问题，不用于提交。

## 可复现的验证结果

- 16 句文字前端音素/token 与训练版一致。
- 完整服务 16/16 成功，回读平均文本相似度 0.966。
- 单实例 warm MCP→ROS 首段：中文 0.75 秒、英文 0.92 秒、混合 0.68 秒。
- 42.45 秒中英混合长句完整结束；中途 interrupt 后约 0.33 秒结束，之后新请求成功。
- 三完整实例 9/9 并发样例成功结束，但首段超时；缩小分块尚未同时解决延迟和连续供音。
- 训练→ONNX 严格数值等价检查仍有已记录失败项；全 TRT vocoder 有退化，当前
  使用 TRT encoder/acoustic + 独立 ORT CUDA vocoder。

## 提交参数草稿

准备个人公开 fork 后，使用该 fork 的真实提交版本：

```yaml
env:
  - name: PHANTHYMOTUS_COMMIT_ID
    value: "<发布后生成的真实 commit id>"
  - name: PHANTHYMOTUS_REPO
    value: "https://github.com/yuliaa1111/phanthymotus.git"
  - name: TTS_PLUGIN
    value: "tts"
  - name: JP_VERSION
    value: "6"
```

该仓库地址为预期 fork 地址，核查时尚不存在；尚未创建 commit、推送或提交榜单。
发布需要用户的 GitHub 登录授权。不要把任何模型/前端语言资源大文件加入 Git。
