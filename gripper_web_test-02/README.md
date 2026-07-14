# 夹爪循环测试工具

Web 工具：通过 SSH 连机器人，循环发送夹爪张开/闭合指令，
可设停顿时间和重复次数，逐次记录成功/失败，自动落盘 CSV。

## 部署：直接把整个文件夹发给同事

同事只需一次性准备：装 Python（https://www.python.org/downloads/ ，
安装时勾选 **Add python.exe to PATH**）。

之后双击 `启动夹爪测试.bat` 即可——首次运行自动安装 flask 和 paramiko，
然后启动工具并自动打开浏览器。

其他跑法（备选）：
- **Linux 上跑**：`pip install flask paramiko && python3 gripper_test_web.py --lan`，
  同事浏览器打开打印出的地址即可（Windows 零安装，但要求这台机器开着且同网段）
- **打包 exe**：`pyinstaller --onefile --name GripperTest gripper_test_web.py`
  （PyInstaller 不能交叉编译，必须在 Windows 上打）

## 同事怎么用

1. 双击 `启动夹爪测试.bat`，浏览器自动打开 `http://127.0.0.1:8000`
2. 确认 IP / 用户名 / 密码（会记住上次填的）
3. 选左/右夹爪，填重复次数和停顿时间，点【开始测试】
4. 页面实时显示每次操作的结果；CSV 自动保存在程序所在目录的 `results/` 下（Excel 可直接打开）

一次循环 = 闭合 → 停顿 → 张开 → 停顿。闭合和张开各记一次操作。
成功判定：服务响应 `status=0`；超时（默认 15s）或其他 status 都算失败，
status 含义：0=成功 1=控制失败 2=闭合但未夹取 3=设备未连接 4=设备未使能。

## config.json

| 字段 | 说明 |
|------|------|
| host / port / user / password | 机器人 SSH 信息（界面上也能改，会写回） |
| source_cmd | SSH 登录后加载 ROS 环境的命令（非交互 SSH 不加载 .bashrc，必须显式 source）。注意必须用 `set -a` 加载 `/etc/environment.d/ros`（ROS_DOMAIN_ID=14、cyclonedds），否则服务不可见、调用一律超时 |
| call_timeout_sec | 单次服务调用超时（秒），超时记失败 |
| gripper_params | force / position / speed，随指令下发 |

密码明文存在 config.json，内网测试工具，够用。

## 本地开发/演示

```bash
python3 gripper_test_web.py --selftest    # 响应解析自检
python3 gripper_test_web.py --mock        # 不连机器人，假数据联调界面
python3 gripper_test_web.py --port 8001   # 换端口（默认 8000，可与 --lan/--mock 组合）
```
