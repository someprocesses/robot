#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""夹爪循环测试工具（Web 界面）

在 Windows 上运行（打包成 exe），通过 SSH 连接机器人，
循环发送 ros2 夹爪张开/闭合指令，逐次记录成功/失败并落盘 CSV。

用法:
    python gripper_test_web.py            # 正常运行，浏览器自动打开
    python gripper_test_web.py --mock     # 无机器人联调界面（假 SSH）
    python gripper_test_web.py --selftest # 运行自检
"""
import csv
import json
import os
import re
import sys
import threading
import time
import webbrowser
from datetime import datetime

from flask import Flask, jsonify, request, send_file

# 打包成 exe 后，配置和结果放在 exe 旁边
BASE_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) \
    else os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
RESULT_DIR = os.path.join(BASE_DIR, "results")

DEFAULT_CONFIG = {
    "host": "192.168.4.185",
    "port": 22,
    "user": "linux",
    "password": "",
    # 非交互 SSH 不会加载 .bashrc 的 ROS 部分，必须显式 source。
    # /etc/environment.d/ros 是 KEY=value 格式（ROS_DOMAIN_ID、RMW 等），
    # 需要 set -a 才能 export 给 ros2 子进程，缺了它服务会不可见、调用超时。
    "source_cmd": "source /opt/ros/humble/setup.bash && set -a && source /etc/environment.d/ros "
                  "&& set +a && source /opt/spiderrobot/setup.bash",
    "call_timeout_sec": 15,
    "gripper_params": {"force": 30.0, "position": 0.5, "speed": 0.2},
}

# cmd_code: 0=右闭合, 1=右张开, 1000=左闭合, 1001=左张开
CMD_CODE = {("right", "close"): 0, ("right", "open"): 1,
            ("left", "close"): 1000, ("left", "open"): 1001}
SERVICE = {"right": "/spr_arm_driver_r/gripper_control",
           "left": "/spr_arm_driver_l/gripper_control"}
STATUS_MEANING = {0: "成功", 1: "控制失败", 2: "闭合但未夹取",
                  3: "设备未连接", 4: "设备未使能"}
ACTION_NAME = {"open": "张开", "close": "闭合"}


def load_config():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
        merged = dict(DEFAULT_CONFIG)
        merged.update(cfg)
        return merged
    except FileNotFoundError:
        save_config(DEFAULT_CONFIG)
    except (ValueError, OSError) as e:
        # 手改出语法错误/写坏的 config 不能让 exe 起不来
        print("config.json 无法解析(%s)，已用默认配置" % e)
    return dict(DEFAULT_CONFIG)


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def parse_response(output):
    """从 ros2 service call 输出提取 (status, message)，找不到返回 (None, 原文摘要)。"""
    m = re.search(r"status=(-?\d+)", output)
    if not m:
        return None, output.strip()[-200:]
    status = int(m.group(1))
    mm = re.search(r"message='([^']*)'", output)
    return status, (mm.group(1) if mm else "")


class SSHExecutor:
    """一次测试保持一条 SSH 长连接，每条指令一个 exec channel。"""

    def __init__(self, cfg):
        import paramiko  # 延迟导入：mock 模式无需安装
        self.cfg = cfg
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(cfg["host"], port=int(cfg["port"]),
                            username=cfg["user"], password=cfg["password"],
                            timeout=8, look_for_keys=True, allow_agent=True)

    def call(self, arm, action):
        """发送一次夹爪指令，返回 (ok, status, message, duration_sec)。"""
        cfg = self.cfg
        p = cfg["gripper_params"]
        t = int(cfg["call_timeout_sec"])
        yaml = ("{cmd_code: %d, force: %s, position: %s, speed: %s}"
                % (CMD_CODE[(arm, action)], p["force"], p["position"], p["speed"]))
        cmd = ("%s && timeout %d ros2 service call %s "
               "spr_arm_interfaces/srv/GripperControl \"%s\""
               % (cfg["source_cmd"], t, SERVICE[arm], yaml))
        start = time.monotonic()
        # 连接级异常（断线/通道超时）直接抛出终止本轮测试，
        # 不能记成夹爪失败——否则断网一夜会把失败率统计整个污染掉
        _, stdout, stderr = self.client.exec_command(cmd, timeout=t + 15)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        exit_code = stdout.channel.recv_exit_status()
        dur = time.monotonic() - start
        if exit_code == 124:
            return False, None, "超时(%ds)" % t, dur
        status, message = parse_response(out)
        if status is None:
            return False, None, "无法解析响应: %s" % (message or err.strip()[-200:]), dur
        return status == 0, status, message, dur

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass


class MockExecutor:
    """--mock 模式：不连机器人，固定节奏返回结果，每第 13 次模拟失败。"""

    def __init__(self, cfg):
        self.n = 0

    def call(self, arm, action):
        self.n += 1
        time.sleep(0.3)
        if self.n % 13 == 0:
            return False, 2, "模拟失败", 0.3
        return True, 0, "ok(mock)", 0.3

    def close(self):
        pass


# ---------------- 测试状态机 ----------------

app = Flask(__name__)
MOCK = "--mock" in sys.argv
LOCK = threading.Lock()
EXECUTOR = {"cur": None}  # 当前测试的 SSH 连接，/stop 靠它打断执行中的指令
STATE = {
    "running": False, "stop": False, "error": "",
    "arm": "", "cycles_total": 0, "cycles_done": 0,
    "total_ops": 0, "fail_ops": 0, "rows": [], "csv_file": "",
}


def record(row, csv_writer, csv_fh):
    with LOCK:
        STATE["total_ops"] += 1
        if not row["ok"]:
            STATE["fail_ops"] += 1
        STATE["rows"].append(row)
        del STATE["rows"][:-200]  # 页面只保留最近 200 条，全量在 CSV
    csv_writer.writerow([row["time"], row["cycle"], row["action"],
                         "%.2f" % row["duration"], row["status"],
                         "成功" if row["ok"] else "失败", row["message"]])
    csv_fh.flush()


def sleep_interruptible(sec):
    end = time.monotonic() + sec
    while time.monotonic() < end:
        if STATE["stop"]:
            return
        time.sleep(min(0.2, end - time.monotonic()))


def run_test(cfg, arm, cycles, pause):
    executor = None
    csv_fh = None
    try:
        executor = (MockExecutor if MOCK else SSHExecutor)(cfg)
        EXECUTOR["cur"] = executor
        os.makedirs(RESULT_DIR, exist_ok=True)
        fname = "gripper_%s_%s.csv" % (arm, datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = os.path.join(RESULT_DIR, fname)
        # utf-8-sig 让 Excel 直接打开不乱码
        csv_fh = open(path, "w", newline="", encoding="utf-8-sig")
        writer = csv.writer(csv_fh)
        writer.writerow(["时间", "轮次", "动作", "耗时(s)", "status", "结果", "message"])
        with LOCK:
            STATE["csv_file"] = path

        for cycle in range(1, cycles + 1):
            for action in ("close", "open"):
                if STATE["stop"]:
                    return
                ok, status, message, dur = executor.call(arm, action)
                meaning = STATUS_MEANING.get(status, "")
                record({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "cycle": cycle,
                    "action": ACTION_NAME[action],
                    "duration": dur,
                    "status": "" if status is None else "%d(%s)" % (status, meaning),
                    "ok": ok,
                    "message": message,
                }, writer, csv_fh)
                sleep_interruptible(pause)
            with LOCK:
                STATE["cycles_done"] = cycle
    except Exception as e:
        # 用户主动停止时断开 SSH 会让执行中的指令抛异常，这属于正常停止
        if not STATE["stop"]:
            with LOCK:
                STATE["error"] = "测试中断: %s" % e
    finally:
        EXECUTOR["cur"] = None
        if executor:
            executor.close()
        if csv_fh:
            csv_fh.close()
        with LOCK:
            STATE["running"] = False


@app.route("/")
def index():
    return PAGE


@app.route("/config")
def get_config():
    cfg = load_config()
    return jsonify({"host": cfg["host"], "user": cfg["user"],
                    "password": cfg["password"], "mock": MOCK})


@app.route("/start", methods=["POST"])
def start():
    # 所有可能失败的操作都在置 running=True 之前做，
    # 否则任何异常都会把 running 永久卡死在 True
    try:
        d = request.get_json(force=True)
        arm = d["arm"]
        assert arm in ("left", "right")
        cycles = int(d["cycles"])
        pause = float(d["pause"])
        assert cycles >= 1 and pause >= 0
    except Exception:
        return jsonify({"error": "参数无效"}), 400
    cfg = load_config()
    cfg.update(host=str(d.get("host", cfg["host"])).strip(),
               user=str(d.get("user", cfg["user"])).strip(),
               password=str(d.get("password", cfg["password"])))
    try:
        save_config(cfg)  # 记住连接信息，下次预填
    except OSError:
        pass  # 目录只读也不影响本次测试
    with LOCK:
        if STATE["running"]:
            return jsonify({"error": "测试正在运行"}), 409
        STATE.update(running=True, stop=False, error="", rows=[],
                     total_ops=0, fail_ops=0, cycles_done=0, csv_file="",
                     arm=arm, cycles_total=cycles)
    threading.Thread(target=run_test, args=(cfg, arm, cycles, pause),
                     daemon=True).start()
    return jsonify({"ok": True})


@app.route("/stop", methods=["POST"])
def stop():
    STATE["stop"] = True
    ex = EXECUTOR["cur"]
    if ex:
        ex.close()  # 打断正在阻塞等待的 SSH 指令，否则要等它超时才能停
    return jsonify({"ok": True})


@app.route("/status")
def status():
    with LOCK:
        return jsonify(STATE)


@app.route("/download")
def download():
    path = STATE["csv_file"]
    if not path or not os.path.exists(path):
        return "还没有测试记录", 404
    return send_file(path, as_attachment=True,
                     download_name=os.path.basename(path))


PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>夹爪循环测试</title>
<style>
  body { font-family: "Microsoft YaHei", sans-serif; margin: 24px auto; max-width: 900px; color: #222; }
  h2 { margin-bottom: 4px; }
  fieldset { border: 1px solid #ccc; border-radius: 6px; margin-bottom: 12px; }
  label { margin-right: 14px; }
  input[type=text], input[type=password], input[type=number] { width: 130px; padding: 3px; }
  button { padding: 6px 22px; font-size: 15px; margin-right: 10px; cursor: pointer; }
  #btnStart { background: #2b7a2b; color: #fff; border: none; border-radius: 4px; }
  #btnStop  { background: #a33; color: #fff; border: none; border-radius: 4px; }
  button:disabled { background: #999; }
  .stats span { display: inline-block; margin-right: 24px; font-size: 15px; }
  .stats b { font-size: 20px; }
  #err { color: #a33; font-weight: bold; }
  table { border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 13px; }
  th, td { border: 1px solid #ddd; padding: 4px 8px; text-align: left; }
  th { background: #f3f3f3; }
  tr.fail td { background: #fde8e8; }
  #mockTag { color: #b60; }
</style>
</head>
<body>
<h2>夹爪循环测试 <small id="mockTag"></small></h2>

<fieldset>
  <legend>机器人连接</legend>
  <label>IP <input type="text" id="host"></label>
  <label>用户名 <input type="text" id="user"></label>
  <label>密码 <input type="password" id="password"></label>
</fieldset>

<fieldset>
  <legend>测试参数</legend>
  <label><input type="radio" name="arm" value="left" checked> 左夹爪</label>
  <label><input type="radio" name="arm" value="right"> 右夹爪</label>
  <label>重复次数 <input type="number" id="cycles" value="10" min="1"></label>
  <label>停顿时间(秒) <input type="number" id="pause" value="2" min="0" step="0.5"></label>
</fieldset>

<p>
  <button id="btnStart" onclick="start()">开始测试</button>
  <button id="btnStop" onclick="stopTest()" disabled>停止</button>
  <a href="/download" target="_blank">下载 CSV</a>
</p>

<div class="stats">
  <span>轮次 <b id="cyc">0/0</b></span>
  <span>总操作 <b id="tot">0</b></span>
  <span>失败 <b id="fail" style="color:#a33">0</b></span>
  <span>失败率 <b id="rate">0%</b></span>
</div>
<p id="err"></p>
<p id="csvPath" style="color:#666;font-size:13px"></p>

<table>
  <thead><tr><th>时间</th><th>轮次</th><th>动作</th><th>耗时(s)</th><th>status</th><th>结果</th><th>message</th></tr></thead>
  <tbody id="rows"></tbody>
</table>

<script>
async function loadCfg() {
  const c = await (await fetch('/config')).json();
  host.value = c.host; user.value = c.user; password.value = c.password;
  if (c.mock) mockTag.textContent = '[MOCK 模式，未连机器人]';
}
async function start() {
  const body = {
    host: host.value, user: user.value, password: password.value,
    arm: document.querySelector('input[name=arm]:checked').value,
    cycles: cycles.value, pause: pause.value
  };
  const r = await fetch('/start', {method: 'POST',
    headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const d = await r.json();
  if (d.error) { err.textContent = d.error; return; }
  err.textContent = '';
}
function stopTest() { fetch('/stop', {method: 'POST'}); }
function esc(s) { const d = document.createElement('span'); d.textContent = s; return d.innerHTML; }
async function poll() {
  try {
    const s = await (await fetch('/status')).json();
    btnStart.disabled = s.running;
    btnStop.disabled = !s.running || s.stop;
    btnStop.textContent = (s.running && s.stop) ? '停止中…' : '停止';
    cyc.textContent = s.cycles_done + '/' + s.cycles_total;
    tot.textContent = s.total_ops; fail.textContent = s.fail_ops;
    rate.textContent = s.total_ops ? (100 * s.fail_ops / s.total_ops).toFixed(1) + '%' : '0%';
    err.textContent = s.error || '';
    csvPath.textContent = s.csv_file ? '记录文件: ' + s.csv_file : '';
    rows.innerHTML = s.rows.slice().reverse().map(r =>
      `<tr class="${r.ok ? '' : 'fail'}"><td>${r.time}</td><td>${r.cycle}</td><td>${r.action}</td>` +
      `<td>${r.duration.toFixed(2)}</td><td>${esc(r.status)}</td>` +
      `<td>${r.ok ? '✓ 成功' : '✗ 失败'}</td><td>${esc(r.message)}</td></tr>`).join('');
  } catch (e) { /* 服务器重启间隙忽略 */ }
}
loadCfg();
setInterval(poll, 600);
</script>
</body>
</html>"""


def selftest():
    ok_out = ("requester: making request: ...\n\nresponse:\n"
              "spr_arm_interfaces.srv.GripperControl_Response(status=0, message='ok')\n")
    assert parse_response(ok_out) == (0, "ok")
    fail_out = "GripperControl_Response(status=2, message='闭合但未夹取')"
    assert parse_response(fail_out) == (2, "闭合但未夹取")
    assert parse_response("garbage")[0] is None
    assert CMD_CODE[("left", "open")] == 1001 and CMD_CODE[("right", "close")] == 0
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit(0)
    try:
        load_config()  # 首次运行生成 config.json
        port = 8000
        if "--port" in sys.argv:
            port = int(sys.argv[sys.argv.index("--port") + 1])
        lan = "--lan" in sys.argv  # 跑在 Linux 上、给局域网内的 Windows 浏览器访问
        if lan:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            print("夹爪测试工具已启动，浏览器访问: http://%s:%d%s"
                  % (ip, port, "  [MOCK]" if MOCK else ""))
        else:
            url = "http://127.0.0.1:%d" % port
            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
            print("夹爪测试工具已启动: %s%s" % (url, "  [MOCK]" if MOCK else ""))
        app.run(host="0.0.0.0" if lan else "127.0.0.1", port=port, threaded=True)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        # 双击运行时窗口一闪而过什么都看不到，停住让人能读到错误
        print("启动失败: %s" % e)
        input("按回车退出...")
