#!/usr/bin/env python3
"""
双机械臂联合轨迹执行器（左臂 L + 右臂 R）
支持独立或协同控制左右机械臂的五次多项式轨迹生成与执行
"""
import argparse
import math
import random
from enum import Enum
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from spr_arm_interfaces.srv import GripperControl


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class ArmType(Enum):
    LEFT = "L"
    RIGHT = "R"


class GripperMode(Enum):
    NONE = 'none'
    CLOSE = 'close'
    OPEN = 'open'
    BOTH = 'both'  # goal1前闭合, goal2前张开


class GripperCmdState(Enum):
    IDLE = 0
    WAITING = 1
    DONE = 2


class SeqState(Enum):
    IDLE = 0
    SEND_GRIPPER_1 = 1
    WAIT_GRIPPER_1 = 2
    SEND_1 = 3
    WAIT_RES_1 = 4
    DELAY_2S = 5
    SEND_GRIPPER_2 = 6
    WAIT_GRIPPER_2 = 7
    SEND_2 = 8
    WAIT_RES_2 = 9
    DONE = 10


def quintic_coeffs(p0, v0, a0, pT, vT, aT, T: float):
    """
    Solve quintic polynomial:
      p(t)=c0+c1 t+c2 t^2+c3 t^3+c4 t^4+c5 t^5
    with boundary constraints at t=0 and t=T:
      p(0)=p0, p'(0)=v0, p''(0)=a0
      p(T)=pT, p'(T)=vT, p''(T)=aT
    """
    if T <= 0:
        raise ValueError("T must be > 0")

    c0 = p0
    c1 = v0
    c2 = a0 / 2.0

    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    T5 = T4 * T

    # Solve linear system for c3,c4,c5
    # Using standard closed-form
    d0 = pT - (c0 + c1 * T + c2 * T2)
    d1 = vT - (c1 + 2.0 * c2 * T)
    d2 = aT - (2.0 * c2)

    c3 = (10.0 * d0 - 4.0 * d1 * T + 0.5 * d2 * T2) / T3
    c4 = (-15.0 * d0 + 7.0 * d1 * T - d2 * T2) / T4
    c5 = (6.0 * d0 - 3.0 * d1 * T + 0.5 * d2 * T2) / T5

    return (c0, c1, c2, c3, c4, c5)


def quintic_eval(c, t: float):
    c0, c1, c2, c3, c4, c5 = c
    p = c0 + c1*t + c2*t**2 + c3*t**3 + c4*t**4 + c5*t**5
    v = c1 + 2*c2*t + 3*c3*t**2 + 4*c4*t**3 + 5*c5*t**4
    a = 2*c2 + 6*c3*t + 12*c4*t**2 + 20*c5*t**3
    return p, v, a


class TrajectoryActionClient(Node):
    def __init__(self, freq: float, num_joints: int, single: bool,
                 min_points: int, max_points: int, dt: float, max_step: float,
                 gripper_mode: str = 'none',
                 gripper_force: float = 30.0,
                 gripper_position: float = 0.5,
                 gripper_speed: float = 0.2):
        super().__init__('dual_arm_trajectory_client')

        self.num_joints = int(num_joints)
        self.dt = float(dt)
        self.max_step = float(max_step)
        self.safety_margin = 0.95

        self.min_points = int(min_points)
        self.max_points = int(max_points)

        self.single = bool(single)
        self.freq = float(freq)

        # ========== 夹爪控制配置 ==========
        self.gripper_mode = GripperMode(gripper_mode)
        self.gripper_force = float(gripper_force)
        self.gripper_position = float(gripper_position)
        self.gripper_speed = float(gripper_speed)

        # 夹爪 Service Client（左臂/右臂各一个）
        self.left_gripper_client = self.create_client(GripperControl, '/spr_arm_driver_l/gripper_control')
        self.right_gripper_client = self.create_client(GripperControl, '/spr_arm_driver_r/gripper_control')
        while not self.left_gripper_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('等待左侧夹爪服务...')
        self.get_logger().info('左侧夹爪服务已连接')
        while not self.right_gripper_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('等待右侧夹爪服务...')
        self.get_logger().info('右侧夹爪服务已连接')

        # 夹爪状态缓存
        self.left_gripper_pending = GripperCmdState.IDLE
        self.right_gripper_pending = GripperCmdState.IDLE
        self.left_gripper_goal_code: Optional[int] = None
        self.right_gripper_goal_code: Optional[int] = None

        # joint limits base (7 joints)
        upper7 = [3.1067, 0.5, 3.1067, 0.5, 3.1067, 1.0472, math.pi / 6.0]
        lower7 = [-3.1067, -0.5, -3.1067, -0.5, -3.1067, -1.0472, -math.pi / 6.0]

        # override: joint_2 (index 1) -> ±pi/6
        upper7[1] = math.pi / 6.0
        lower7[1] = -math.pi / 6.0

        # ========== 左臂配置 ==========
        self.left_joint_names = [f'L_joint{i+1}' for i in range(self.num_joints)]
        self.left_joint_lower: List[float] = []
        self.left_joint_upper: List[float] = []
        for i in range(self.num_joints):
            if i < 7:
                lo, hi = lower7[i], upper7[i]
            else:
                lo, hi = -math.pi, math.pi

            center = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo) * self.safety_margin
            self.left_joint_lower.append(center - half)
            self.left_joint_upper.append(center + half)

        # 左关节状态缓存
        self._left_joint_pos_map: Dict[str, float] = {}
        self._have_left_joints = False
        self._printed_left_initial = False

        # 左臂订阅器
        left_state_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.create_subscription(JointState, '/spr_arm_driver_l/joint_states', self.left_joint_state_cb, left_state_qos)

        # 左臂动作客户端
        self.left_action_client = ActionClient(self, FollowJointTrajectory, '/spr_arm_driver_l/follow_joint_trajectory')
        self.get_logger().info('等待左侧动作服务器...')
        self.left_action_client.wait_for_server()
        self.get_logger().info('左侧动作服务器已连接')

        # ========== 右臂配置 ==========
        self.right_joint_names = [f'R_joint{i+1}' for i in range(self.num_joints)]
        self.right_joint_lower: List[float] = []
        self.right_joint_upper: List[float] = []
        for i in range(self.num_joints):
            if i < 7:
                lo, hi = lower7[i], upper7[i]
            else:
                lo, hi = -math.pi, math.pi

            center = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo) * self.safety_margin
            self.right_joint_lower.append(center - half)
            self.right_joint_upper.append(center + half)

        # 右关节状态缓存
        self._right_joint_pos_map: Dict[str, float] = {}
        self._have_right_joints = False
        self._printed_right_initial = False

        # 右臂订阅器
        right_state_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.create_subscription(JointState, '/spr_arm_driver_r/joint_states', self.right_joint_state_cb, right_state_qos)

        # 右臂动作客户端
        self.right_action_client = ActionClient(self, FollowJointTrajectory, '/spr_arm_driver_r/follow_joint_trajectory')
        self.get_logger().info('等待右侧动作服务器...')
        self.right_action_client.wait_for_server()
        self.get_logger().info('右侧动作服务器已连接')

        # ========== 状态机（每个臂独立）==========
        self.left_state = SeqState.IDLE
        self.right_state = SeqState.IDLE
        self.seq_id = 0
        self.result_ready_left_1 = False
        self.result_ready_right_1 = False
        self.result_ready_left_2 = False
        self.result_ready_right_2 = False
        self.last_error_code_left_1: Optional[int] = None
        self.last_error_code_right_1: Optional[int] = None
        self.last_error_code_left_2: Optional[int] = None
        self.last_error_code_right_2: Optional[int] = None
        self.delay_until_ns: Optional[int] = None
        self._next_start_ns: Optional[int] = None

        self.loop_timer = self.create_timer(0.05, self.loop)

        vmax = self.max_step / self.dt
        self.get_logger().info('=' * 60)
        self.get_logger().info('双机械臂联合轨迹执行器')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'Quintic 段拼接版：dt={self.dt}s, max_step={self.max_step}rad (vmax={vmax:.3f}rad/s)')
        self.get_logger().info(f'points=random[{self.min_points},{self.max_points}] (internal min>=20)')
        self.get_logger().info(f'joint_2 限位=±pi/6(再乘 95% 安全裕度)')
        self.get_logger().info(f'single={self.single}, seq_start_freq={self.freq}Hz')
        self.get_logger().info(f'gripper_mode={self.gripper_mode.value}, force={self.gripper_force}, '
                              f'position={self.gripper_position}, speed={self.gripper_speed}')
        self.get_logger().info('=' * 60)

    # -------- joint_states callbacks --------
    def left_joint_state_cb(self, msg: JointState):
        for n, p in zip(msg.name, msg.position):
            self._left_joint_pos_map[n] = float(p)
        self._have_left_joints = all(n in self._left_joint_pos_map for n in self.left_joint_names)

        if self._have_left_joints and not self._printed_left_initial:
            raw = [self._left_joint_pos_map[n] for n in self.left_joint_names]
            clamped, flags = self._clamp_positions(raw, self.left_joint_lower, self.left_joint_upper)
            if any(flags):
                self.get_logger().warn(
                    f'[左臂] 初始 joint_states 超限 -> clamp 继续。\n'
                    f'raw={ [round(x,6) for x in raw] }\n'
                    f'clamped={ [round(x,6) for x in clamped] }\n'
                    f'flags={flags}'
                )
            else:
                self.get_logger().info(f'[左臂] 初始 joint_states OK: { [round(x,6) for x in clamped] }')
            self._printed_left_initial = True

    def right_joint_state_cb(self, msg: JointState):
        for n, p in zip(msg.name, msg.position):
            self._right_joint_pos_map[n] = float(p)
        self._have_right_joints = all(n in self._right_joint_pos_map for n in self.right_joint_names)

        if self._have_right_joints and not self._printed_right_initial:
            raw = [self._right_joint_pos_map[n] for n in self.right_joint_names]
            clamped, flags = self._clamp_positions(raw, self.right_joint_lower, self.right_joint_upper)
            if any(flags):
                self.get_logger().warn(
                    f'[右臂] 初始 joint_states 超限 -> clamp 继续。\n'
                    f'raw={ [round(x,6) for x in raw] }\n'
                    f'clamped={ [round(x,6) for x in clamped] }\n'
                    f'flags={flags}'
                )
            else:
                self.get_logger().info(f'[右臂] 初始 joint_states OK: { [round(x,6) for x in clamped] }')
            self._printed_right_initial = True

    def _clamp_positions(self, pos: List[float], lower: List[float], upper: List[float]) -> Tuple[List[float], List[bool]]:
        out, flags = [], []
        for i in range(len(pos)):
            x = float(pos[i])
            y = clamp(x, lower[i], upper[i])
            out.append(y)
            flags.append(abs(y - x) > 1e-12)
        return out, flags

    def get_current_positions_clamped(self, joint_names: List[str], 
                                      joint_lower: List[float], 
                                      joint_upper: List[float],
                                      pos_map: Dict[str, float]) -> Optional[List[float]]:
        if not all(n in pos_map for n in joint_names):
            return None
        raw = [pos_map[n] for n in joint_names]
        clamped, _ = self._clamp_positions(raw, joint_lower, joint_upper)
        return clamped

    # -------- quintic piecewise generation --------
    def _choose_turning_values(self, start: float, j: int, 
                                joint_lower: float, joint_upper: float,
                                n_turns: int) -> List[float]:
        """
        返回转折点位置（value），n_turns=1 或 2
        目标：幅值大（靠近限位），同时保持在限位内
        """
        start = clamp(start, joint_lower, joint_upper)

        margin = 0.05 * (joint_upper - joint_lower)
        up_val = clamp(joint_upper - margin, joint_lower, joint_upper)
        dn_val = clamp(joint_lower + margin, joint_lower, joint_upper)

        direction = random.choice([-1, 1])
        if n_turns == 1:
            v1 = up_val if direction > 0 else dn_val
            return [v1]
        else:
            # 两次换向：上->下 或 下->上
            if direction > 0:
                return [up_val, dn_val]
            else:
                return [dn_val, up_val]

    def _gen_joint_quintic(self, n_points: int, start: float, j: int,
                          joint_lower: float, joint_upper: float) -> Tuple[List[float], List[float], List[float]]:
        """
        单关节 quintic 拼接，最多 3 段：
        - start -> turn1 -> turn2 -> start
        - 端点速度/加速度全为 0，保证每段拼接 C2
        """
        start = clamp(start, joint_lower, joint_upper)

        # 决定换向次数 1 或 2（最多 2）
        n_turns = 2 if n_points >= 60 else 1  # 点很多更像 sin 就用 2 次换向
        if n_points >= 120:
            n_turns = random.choice([1, 2])
        turns = self._choose_turning_values(start, j, joint_lower, joint_upper, n_turns)

        # 段数：turns=1 => 2 段；turns=2 => 3 段
        knots = [start] + turns + [start]
        n_segs = len(knots) - 1

        # 给每段分配点数（段长），至少每段>=2 点，且总和=n_points
        # 用比例分配：中间段稍长一点
        if n_segs == 2:
            seg_pts = [n_points // 2, n_points - n_points // 2]
        else:
            # 3 段：1:2:1 或 1:1:1 随机
            if random.random() < 0.7:
                a = n_points // 4
                b = n_points - 2 * a
                seg_pts = [a, b - a, a]  # 修正后仍总和=n_points（下面会再 normalize）
            else:
                seg_pts = [n_points // 3, n_points // 3, n_points - 2 * (n_points // 3)]

            # 确保每段>=2
            for k in range(3):
                seg_pts[k] = max(seg_pts[k], 2)
            # normalize total
            s = sum(seg_pts)
            seg_pts[-1] += (n_points - s)

        # 再确保每段>=2 且总和正确
        for k in range(n_segs):
            seg_pts[k] = max(seg_pts[k], 2)
        s = sum(seg_pts)
        seg_pts[-1] += (n_points - s)

        # 每段时间
        seg_T = [max((seg_pts[k] - 1) * self.dt, self.dt) for k in range(n_segs)]

        # 构造整条序列（注意拼接时避免重复点：每段除第一段外跳过第 0 个采样）
        q_all: List[float] = []
        v_all: List[float] = []
        a_all: List[float] = []

        for k in range(n_segs):
            p0 = knots[k]
            p1 = knots[k + 1]
            T = seg_T[k]

            # 端点速度/加速度=0（保证 C2 拼接）
            c = quintic_coeffs(p0, 0.0, 0.0, p1, 0.0, 0.0, T)

            n_k = seg_pts[k]
            for i in range(n_k):
                t = (i * self.dt)
                if t > T:
                    t = T
                q, v, a = quintic_eval(c, t)
                q = clamp(q, joint_lower, joint_upper)
                if k > 0 and i == 0:
                    continue  # avoid duplicate knot sample
                q_all.append(q)
                v_all.append(v)
                a_all.append(a)

        # 可能因为时间截断导致长度偏差，修正到 n_points
        if len(q_all) > n_points:
            q_all = q_all[:n_points]
            v_all = v_all[:n_points]
            a_all = a_all[:n_points]
        elif len(q_all) < n_points:
            # 末尾补齐
            while len(q_all) < n_points:
                q_all.append(q_all[-1])
                v_all.append(0.0)
                a_all.append(0.0)

        # 强制首尾= start
        q_all[0] = start
        q_all[-1] = start
        v_all[0] = 0.0
        v_all[-1] = 0.0
        a_all[0] = 0.0
        a_all[-1] = 0.0

        return q_all, v_all, a_all

    def _enforce_max_step_1d(self, q: List[float], lo: float, hi: float) -> None:
        """
        对单关节序列做硬约束：|dq|<=max_step，并限位。
        做两遍（backward+forward）保证终点也不突变。
        """
        n = len(q)
        # forward
        for i in range(1, n):
            diff = q[i] - q[i - 1]
            if abs(diff) > self.max_step:
                q[i] = q[i - 1] + math.copysign(self.max_step, diff)
            q[i] = clamp(q[i], lo, hi)

        # backward (protect end)
        for i in range(n - 2, -1, -1):
            diff = q[i + 1] - q[i]
            if abs(diff) > self.max_step:
                q[i] = q[i + 1] - math.copysign(self.max_step, diff)
            q[i] = clamp(q[i], lo, hi)

        # forward again
        for i in range(1, n):
            diff = q[i] - q[i - 1]
            if abs(diff) > self.max_step:
                q[i] = q[i - 1] + math.copysign(self.max_step, diff)
            q[i] = clamp(q[i], lo, hi)

    def generate_trajectory(self, n_points: int, start: List[float],
                           joint_lower: List[float], joint_upper: List[float]) -> \
            Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        """
        逐关节生成 quintic 序列（C2），再做 max_step 硬约束（采样级别）。
        """
        qj_all: List[List[float]] = []
        vj_all: List[List[float]] = []
        aj_all: List[List[float]] = []

        for j in range(self.num_joints):
            q, v, a = self._gen_joint_quintic(n_points, start[j], j,
                                              joint_lower[j], joint_upper[j])

            # max_step & limits hard enforcement on sampled q
            self._enforce_max_step_1d(q, joint_lower[j], joint_upper[j])

            # 由于 q 被调整过，v/a 用差分重新计算（保证与 q 一致，且更平滑）
            v_new = [0.0] * n_points
            a_new = [0.0] * n_points
            dt = self.dt
            v_max = self.max_step / self.dt

            for i in range(n_points):
                if i == 0 or i == n_points - 1:
                    v_new[i] = 0.0
                else:
                    v_new[i] = clamp((q[i + 1] - q[i - 1]) / (2.0 * dt), -v_max, v_max)

            for i in range(n_points):
                if i == 0 or i == n_points - 1:
                    a_new[i] = 0.0
                else:
                    a_new[i] = (v_new[i + 1] - v_new[i - 1]) / (2.0 * dt)

            qj_all.append(q)
            vj_all.append(v_new)
            aj_all.append(a_new)

        pos = [[qj_all[j][i] for j in range(self.num_joints)] for i in range(n_points)]
        vel = [[vj_all[j][i] for j in range(self.num_joints)] for i in range(n_points)]
        acc = [[aj_all[j][i] for j in range(self.num_joints)] for i in range(n_points)]
        return pos, vel, acc

    # -------- build goal --------
    def build_goal(self, joint_names: List[str],
                  joint_lower: List[float], joint_upper: List[float],
                  pos_map: Dict[str, float]) -> Optional[FollowJointTrajectory.Goal]:
        start = self.get_current_positions_clamped(joint_names, joint_lower, joint_upper, pos_map)
        if start is None:
            return None

        n_points = random.randint(self.min_points, self.max_points)
        if n_points < 20:
            n_points = 20

        pos_seq, vel_seq, acc_seq = self.generate_trajectory(n_points, start,
                                                            joint_lower, joint_upper)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = joint_names

        pts: List[JointTrajectoryPoint] = []
        for i in range(n_points):
            p = JointTrajectoryPoint()
            p.positions = pos_seq[i]
            p.velocities = vel_seq[i]
            p.accelerations = acc_seq[i]
            p.time_from_start = Duration(
                sec=int(i * self.dt),
                nanosec=int((i * self.dt % 1.0) * 1e9),
            )
            pts.append(p)

        goal.trajectory.points = pts

        goal.path_tolerance = []
        goal.goal_tolerance = []
        for i in range(self.num_joints):
            path_tol = JointTolerance()
            path_tol.name = joint_names[i]
            path_tol.position = 0.1
            path_tol.velocity = 0.2
            goal.path_tolerance.append(path_tol)

            goal_tol = JointTolerance()
            goal_tol.name = joint_names[i]
            goal_tol.position = 0.05
            goal_tol.velocity = 0.1
            goal.goal_tolerance.append(goal_tol)

        goal.goal_time_tolerance = Duration(sec=1, nanosec=0)
        return goal

    # -------- gripper control --------
    def _send_gripper_cmd(self, is_left: bool, cmd_code: int) -> bool:
        """
        发送夹爪控制指令（异步）
        返回 True 表示请求已发送，False 表示服务不可用
        """
        client = self.left_gripper_client if is_left else self.right_gripper_client
        if not client.service_is_ready():
            self.get_logger().warn(f'[{"左" if is_left else "右"}爪] 服务不可用')
            return False

        req = GripperControl.Request()
        req.cmd_code = cmd_code
        req.force = self.gripper_force
        req.position = self.gripper_position
        req.speed = self.gripper_speed

        arm_label = '\u5de6' if is_left else '\u53f3'
        code_desc = '\u95ed\u5408' if cmd_code in (0, 1000) else '\u5f20\u5f00'
        self.get_logger().info(f'[Seq {self.seq_id}-{arm_label}\u722a] \u53d1\u9001 {code_desc} (cmd_code={cmd_code})')

        future = client.call_async(req)
        if is_left:
            self.left_gripper_goal_code = cmd_code
            future.add_done_callback(self._gripper_done_cb_left)
        else:
            self.right_gripper_goal_code = cmd_code
            future.add_done_callback(self._gripper_done_cb_right)
        return True

    def _gripper_done_cb_left(self, future):
        try:
            resp = future.result()
            self.get_logger().info(f'[\u5de6\u722a] \u7ed3\u679c: success={resp.success}')
        except Exception as e:
            self.get_logger().error(f'[\u5de6\u722a] \u8c03\u7528\u5931\u8d25: {e}')
        self.left_gripper_pending = GripperCmdState.DONE

    def _gripper_done_cb_right(self, future):
        try:
            resp = future.result()
            self.get_logger().info(f'[\u53f3\u722a] \u7ed3\u679c: success={resp.success}')
        except Exception as e:
            self.get_logger().error(f'[\u53f3\u722a] \u8c03\u7528\u5931\u8d25: {e}')
        self.right_gripper_pending = GripperCmdState.DONE

    def _get_gripper_cmd_code(self, arm: str, for_goal1: bool):
        """\u6839\u636e\u5939\u722a\u6a21\u5f0f\u548c\u5f53\u524d\u662f goal1/goal2\uff0c\u8fd4\u56de cmd_code"""
        mode = self.gripper_mode
        if mode == GripperMode.NONE:
            return None
        is_left = (arm == 'left')
        if mode == GripperMode.CLOSE:
            return 1000 if is_left else 0
        elif mode == GripperMode.OPEN:
            return 1001 if is_left else 1
        elif mode == GripperMode.BOTH:
            if for_goal1:
                return 1000 if is_left else 0
            else:
                return 1001 if is_left else 1
        return None

    def _do_pre_gripper(self, arm: str, for_goal1: bool) -> bool:
        """
        \u5728\u53d1\u9001\u8f68\u8ff9\u524d\u6267\u884c\u5939\u722a\u52a8\u4f5c
        \u8fd4\u56de True \u8868\u793a\u5df2\u53d1\u51fa\u5939\u722a\u8bf7\u6c42
        \u8fd4\u56de False \u8868\u793a\u4e0d\u9700\u8981\u5939\u722a\u52a8\u4f5c
        """
        cmd_code = self._get_gripper_cmd_code(arm, for_goal1)
        if cmd_code is None:
            return False
        is_left = (arm == 'left')
        if is_left:
            self.left_gripper_pending = GripperCmdState.WAITING
        else:
            self.right_gripper_pending = GripperCmdState.WAITING
        self._send_gripper_cmd(is_left, cmd_code)
        return True

    def _is_gripper_done(self, arm: str) -> bool:
        """\u68c0\u67e5\u5939\u722a\u52a8\u4f5c\u662f\u5426\u5b8c\u6210"""
        if arm == 'left':
            return self.left_gripper_pending == GripperCmdState.DONE
        else:
            return self.right_gripper_pending == GripperCmdState.DONE

    # -------- action callbacks (LEFT ARM) --------
    def _goal1_response_cb_left(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error('[Seq-左] goal1 被拒绝')
            self.last_error_code_left_1 = None
            self.result_ready_left_1 = True
            return
        rf = gh.get_result_async()
        rf.add_done_callback(self._goal1_result_cb_left)

    def _goal1_result_cb_left(self, future):
        res = future.result().result
        self.last_error_code_left_1 = int(res.error_code)
        self.result_ready_left_1 = True

    def _goal2_response_cb_left(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error('[Seq-左] goal2 被拒绝')
            self.last_error_code_left_2 = None
            self.result_ready_left_2 = True
            return
        rf = gh.get_result_async()
        rf.add_done_callback(self._goal2_result_cb_left)

    def _goal2_result_cb_left(self, future):
        res = future.result().result
        self.last_error_code_left_2 = int(res.error_code)
        self.result_ready_left_2 = True

    # -------- action callbacks (RIGHT ARM) --------
    def _goal1_response_cb_right(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error('[Seq-右] goal1 被拒绝')
            self.last_error_code_right_1 = None
            self.result_ready_right_1 = True
            return
        rf = gh.get_result_async()
        rf.add_done_callback(self._goal1_result_cb_right)

    def _goal1_result_cb_right(self, future):
        res = future.result().result
        self.last_error_code_right_1 = int(res.error_code)
        self.result_ready_right_1 = True

    def _goal2_response_cb_right(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error('[Seq-右] goal2 被拒绝')
            self.last_error_code_right_2 = None
            self.result_ready_right_2 = True
            return
        rf = gh.get_result_async()
        rf.add_done_callback(self._goal2_result_cb_right)

    def _goal2_result_cb_right(self, future):
        res = future.result().result
        self.last_error_code_right_2 = int(res.error_code)
        self.result_ready_right_2 = True

    # -------- state machine --------
    def _handle_arm_state_machine(self, arm_label: str, now_ns: int):
        """处理单臂状态机（含夹爪控制）"""
        is_left = (arm_label == 'left')
        state = self.left_state if is_left else self.right_state
        have_joints = self._have_left_joints if is_left else self._have_right_joints
        joint_names = self.left_joint_names if is_left else self.right_joint_names
        joint_lower = self.left_joint_lower if is_left else self.right_joint_lower
        joint_upper = self.left_joint_upper if is_left else self.right_joint_upper
        pos_map = self._left_joint_pos_map if is_left else self._right_joint_pos_map
        action_client = self.left_action_client if is_left else self.right_action_client
        goal1_cb_response = self._goal1_response_cb_left if is_left else self._goal1_response_cb_right
        goal1_cb_result = self._goal1_result_cb_left if is_left else self._goal1_result_cb_right
        goal2_cb_response = self._goal2_response_cb_left if is_left else self._goal2_response_cb_right
        goal2_cb_result = self._goal2_result_cb_left if is_left else self._goal2_result_cb_right
        result_ready_1 = self.result_ready_left_1 if is_left else self.result_ready_right_1
        result_ready_2 = self.result_ready_left_2 if is_left else self.result_ready_right_2
        last_error_1 = self.last_error_code_left_1 if is_left else self.last_error_code_right_1
        last_error_2 = self.last_error_code_left_2 if is_left else self.last_error_code_right_2
        prefix = '\u5de6' if is_left else '\u53f3'

        if state == SeqState.IDLE:
            if not have_joints:
                pass
            elif self.single:
                self.seq_id += 1
                state = SeqState.SEND_GRIPPER_1
            else:
                if self._next_start_ns is None:
                    self._next_start_ns = now_ns
                if now_ns >= self._next_start_ns:
                    self.seq_id += 1
                    state = SeqState.SEND_GRIPPER_1
                    self._next_start_ns = now_ns + int((1.0 / max(self.freq, 1e-6)) * 1e9)

        # ====== goal1 前夹爪动作 ======
        elif state == SeqState.SEND_GRIPPER_1:
            need_wait = self._do_pre_gripper(arm_label, for_goal1=True)
            if need_wait:
                state = SeqState.WAIT_GRIPPER_1
            else:
                state = SeqState.SEND_1  # 不需要夹爪，直接发轨迹

        elif state == SeqState.WAIT_GRIPPER_1:
            if not self._is_gripper_done(arm_label):
                return
            # 夹爪完成，加一点延时再发轨迹
            self.delay_until_ns = now_ns + int(0.3 * 1e9)
            state = SeqState.SEND_1

        elif state == SeqState.SEND_1:
            goal = self.build_goal(joint_names, joint_lower, joint_upper, pos_map)
            if goal is None:
                self.get_logger().warn(f'[Seq-{prefix}\u81c2] \u7b49\u5f85 joint_states...')
                return
            if is_left:
                self.result_ready_left_1 = False
                self.last_error_code_left_1 = None
            else:
                self.result_ready_right_1 = False
                self.last_error_code_right_1 = None
            self.get_logger().info(f'[Seq {self.seq_id}-{prefix}\u81c2] \u53d1\u9001 goal1, points={len(goal.trajectory.points)}')
            f = action_client.send_goal_async(goal)
            f.add_done_callback(goal1_cb_response)
            state = SeqState.WAIT_RES_1

        elif state == SeqState.WAIT_RES_1:
            ready = self.result_ready_left_1 if is_left else self.result_ready_right_1
            err = self.last_error_code_left_1 if is_left else self.last_error_code_right_1
            if not ready:
                return
            self.get_logger().info(f'[Seq {self.seq_id}-{prefix}\u81c2] goal1 \u7ed3\u675f\uff0cerror_code={err}; \u7b49\u5f85 2 \u79d2...')
            self.delay_until_ns = now_ns + int(2.0 * 1e9)
            state = SeqState.DELAY_2S

        elif state == SeqState.DELAY_2S:
            if self.delay_until_ns is None or now_ns < self.delay_until_ns:
                return
            state = SeqState.SEND_GRIPPER_2

        # ====== goal2 前夹爪动作 ======
        elif state == SeqState.SEND_GRIPPER_2:
            need_wait = self._do_pre_gripper(arm_label, for_goal1=False)
            if need_wait:
                state = SeqState.WAIT_GRIPPER_2
            else:
                state = SeqState.SEND_2

        elif state == SeqState.WAIT_GRIPPER_2:
            if not self._is_gripper_done(arm_label):
                return
            self.delay_until_ns = now_ns + int(0.3 * 1e9)
            state = SeqState.SEND_2

        elif state == SeqState.SEND_2:
            goal = self.build_goal(joint_names, joint_lower, joint_upper, pos_map)
            if goal is None:
                self.get_logger().warn(f'[Seq-{prefix}\u81c2] \u7b49\u5f85 joint_states...')
                return
            if is_left:
                self.result_ready_left_2 = False
                self.last_error_code_left_2 = None
            else:
                self.result_ready_right_2 = False
                self.last_error_code_right_2 = None
            self.get_logger().info(f'[Seq {self.seq_id}-{prefix}\u81c2] \u53d1\u9001 goal2, points={len(goal.trajectory.points)}')
            f = action_client.send_goal_async(goal)
            f.add_done_callback(goal2_cb_response)
            state = SeqState.WAIT_RES_2

        elif state == SeqState.WAIT_RES_2:
            ready = self.result_ready_left_2 if is_left else self.result_ready_right_2
            err = self.last_error_code_left_2 if is_left else self.last_error_code_right_2
            if not ready:
                return
            self.get_logger().info(f'[Seq {self.seq_id}-{prefix}\u81c2] goal2 \u7ed3\u675f\uff0cerror_code={err}')
            state = SeqState.DONE

        elif state == SeqState.DONE:
            if self.single:
                if is_left:
                    self.left_state = SeqState.IDLE
                else:
                    self.get_logger().info('single \u6a21\u5f0f\uff1a\u5e8f\u5217\u5b8c\u6210\uff0c\u9000\u51fa')
                    rclpy.shutdown()
                    return
            else:
                state = SeqState.IDLE

        # \u4fdd\u5b58\u72b6\u6001
        if is_left:
            self.left_state = state
        else:
            self.right_state = state

    def loop(self):
        now_ns = self.get_clock().now().nanoseconds

        # ========== \u5904\u7406\u5de6\u81c2\u72b6\u6001\u673a ==========
        self._handle_arm_state_machine('left', now_ns)

        # ========== \u5904\u7406\u53f3\u81c2\u72b6\u6001\u673a ==========
        self._handle_arm_state_machine('right', now_ns)


def main():
    parser = argparse.ArgumentParser(description='双机械臂联合轨迹执行器')
    parser.add_argument('--freq', '-f', type=float, default=1.0,
                        help='非 single 模式：每秒最多启动多少个序列 (序列=2 个 action)。默认 1')
    parser.add_argument('--num-joints', '-n', type=int, default=7)
    parser.add_argument('--single', action='store_true', help='只执行一次序列 (2 个 action) 然后退出')

    parser.add_argument('--min-points', type=int, default=20, help='最小点数（建议>=20）')
    parser.add_argument('--max-points', type=int, default=1999, help='最大点数')

    parser.add_argument('--dt', type=float, default=0.1, help='点间隔时间 (s)，默认 0.1')
    parser.add_argument('--max-step', type=float, default=0.04, help='相邻点最大位移 (rad)，默认 0.04')

    parser.add_argument('--gripper', type=str, default='none',
                        choices=['none', 'close', 'open', 'both'],
                        help='夹爪模式: none=不控制, close=每次轨迹前闭合, open=每次轨迹前张开, both=goal1闭合+goal2张开')
    parser.add_argument('--gripper-force', type=float, default=30.0, help='夹爪夹持力，默认 30.0')
    parser.add_argument('--gripper-position', type=float, default=0.5, help='夹爪位置，默认 0.5')
    parser.add_argument('--gripper-speed', type=float, default=0.2, help='夹爪速度，默认 0.2')

    args = parser.parse_args()

    rclpy.init()
    node = TrajectoryActionClient(
        freq=args.freq,
        num_joints=args.num_joints,
        single=args.single,
        min_points=args.min_points,
        max_points=args.max_points,
        dt=args.dt,
        max_step=args.max_step,
        gripper_mode=args.gripper,
        gripper_force=args.gripper_force,
        gripper_position=args.gripper_position,
        gripper_speed=args.gripper_speed,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()