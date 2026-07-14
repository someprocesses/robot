#!/usr/bin/env python3
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


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class SeqState(Enum):
    IDLE = 0
    SEND_1 = 1
    WAIT_RES_1 = 2
    DELAY_2S = 3
    SEND_2 = 4
    WAIT_RES_2 = 5
    DONE = 6


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
                 min_points: int, max_points: int, dt: float, max_step: float):
        super().__init__('trajectory_action_client')

        self.num_joints = int(num_joints)
        self.joint_names = [f'R_joint{i+1}' for i in range(self.num_joints)]

        self.dt = float(dt)
        self.max_step = float(max_step)
        self.safety_margin = 0.95

        self.min_points = int(min_points)
        self.max_points = int(max_points)

        self.single = bool(single)
        self.freq = float(freq)

        # joint limits base (7 joints)
        upper7 = [3.1067, 0.5, 3.1067, 0.5, 3.1067, 1.0472, math.pi / 6.0]
        lower7 = [-3.1067, -0.5, -3.1067, -0.5, -3.1067, -1.0472, -math.pi / 6.0]

        # override: joint_2 (index 1) -> ±pi/6
        upper7[1] = math.pi / 6.0
        lower7[1] = -math.pi / 6.0

        self.joint_lower: List[float] = []
        self.joint_upper: List[float] = []
        for i in range(self.num_joints):
            if i < 7:
                lo, hi = lower7[i], upper7[i]
            else:
                lo, hi = -math.pi, math.pi

            center = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo) * self.safety_margin
            self.joint_lower.append(center - half)
            self.joint_upper.append(center + half)

        # joint_states cache
        self._joint_pos_map: Dict[str, float] = {}
        self._have_all_joints = False
        self._printed_initial = False

        joint_state_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.create_subscription(JointState, '/spr_arm_driver_r/joint_states', self.joint_state_cb, joint_state_qos)

        # action client
        self.action_client = ActionClient(self, FollowJointTrajectory, '/spr_arm_driver_r/follow_joint_trajectory')
        self.get_logger().info('等待动作服务器...')
        self.action_client.wait_for_server()
        self.get_logger().info('动作服务器已连接')

        # state machine
        self.state = SeqState.IDLE
        self.seq_id = 0
        self.result_ready_1 = False
        self.result_ready_2 = False
        self.last_error_code_1: Optional[int] = None
        self.last_error_code_2: Optional[int] = None
        self.delay_until_ns: Optional[int] = None
        self._next_start_ns: Optional[int] = None

        self.loop_timer = self.create_timer(0.05, self.loop)

        vmax = self.max_step / self.dt
        self.get_logger().info(
            f'Quintic段拼接版：dt={self.dt}s, max_step={self.max_step}rad (vmax={vmax:.3f}rad/s)\n'
            f'points=random[{self.min_points},{self.max_points}] (internal min>=20)\n'
            f'joint_2限位=±pi/6(再乘95%安全裕度)\n'
            f'single={self.single}, seq_start_freq={self.freq}Hz'
        )

    # -------- joint_states --------
    def joint_state_cb(self, msg: JointState):
        for n, p in zip(msg.name, msg.position):
            self._joint_pos_map[n] = float(p)
        self._have_all_joints = all(n in self._joint_pos_map for n in self.joint_names)

        if self._have_all_joints and not self._printed_initial:
            raw = [self._joint_pos_map[n] for n in self.joint_names]
            clamped, flags = self._clamp_positions(raw)
            if any(flags):
                self.get_logger().warn(
                    f'初始 joint_states 超限 -> clamp继续。\n'
                    f'raw={ [round(x,6) for x in raw] }\n'
                    f'clamped={ [round(x,6) for x in clamped] }\n'
                    f'flags={flags}'
                )
            else:
                self.get_logger().info(f'初始 joint_states OK: { [round(x,6) for x in clamped] }')
            self._printed_initial = True

    def _clamp_positions(self, pos: List[float]) -> Tuple[List[float], List[bool]]:
        out, flags = [], []
        for i in range(self.num_joints):
            x = float(pos[i])
            y = clamp(x, self.joint_lower[i], self.joint_upper[i])
            out.append(y)
            flags.append(abs(y - x) > 1e-12)
        return out, flags

    def get_current_positions_clamped(self) -> Optional[List[float]]:
        if not self._have_all_joints:
            return None
        raw = [self._joint_pos_map[n] for n in self.joint_names]
        clamped, _ = self._clamp_positions(raw)
        return clamped

    # -------- quintic piecewise generation (<=2 direction changes) --------
    def _choose_turning_values(self, start: float, j: int, n_turns: int) -> List[float]:
        """
        返回转折点位置（value），n_turns=1或2
        目标：幅值大（靠近限位），同时保持在限位内
        """
        lo = self.joint_lower[j]
        hi = self.joint_upper[j]
        start = clamp(start, lo, hi)

        margin = 0.05 * (hi - lo)
        up_val = clamp(hi - margin, lo, hi)
        dn_val = clamp(lo + margin, lo, hi)

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

    def _gen_joint_quintic(self, n_points: int, start: float, j: int) -> Tuple[List[float], List[float], List[float]]:
        """
        单关节 quintic 拼接，最多3段：
        - start -> turn1 -> turn2 -> start
        - 端点速度/加速度全为0，保证每段拼接 C2
        """
        lo, hi = self.joint_lower[j], self.joint_upper[j]
        start = clamp(start, lo, hi)

        # 决定换向次数 1或2（最多2）
        n_turns = 2 if n_points >= 60 else 1  # 点很多更像sin就用2次换向
        if n_points >= 120:
            n_turns = random.choice([1, 2])
        turns = self._choose_turning_values(start, j, n_turns)

        # 段数：turns=1 => 2段；turns=2 => 3段
        knots = [start] + turns + [start]
        n_segs = len(knots) - 1

        # 给每段分配点数（段长），至少每段>=2点，且总和=n_points
        # 用比例分配：中间段稍长一点
        if n_segs == 2:
            seg_pts = [n_points // 2, n_points - n_points // 2]
        else:
            # 3段：1:2:1 或 1:1:1 随机
            if random.random() < 0.7:
                a = n_points // 4
                b = n_points - 2 * a
                seg_pts = [a, b - a, a]  # 修正后仍总和=n_points（下面会再normalize）
            else:
                seg_pts = [n_points // 3, n_points // 3, n_points - 2 * (n_points // 3)]

            # 确保每段>=2
            for k in range(3):
                seg_pts[k] = max(seg_pts[k], 2)
            # normalize total
            s = sum(seg_pts)
            seg_pts[-1] += (n_points - s)

        # 再确保每段>=2且总和正确
        for k in range(n_segs):
            seg_pts[k] = max(seg_pts[k], 2)
        s = sum(seg_pts)
        seg_pts[-1] += (n_points - s)

        # 每段时间
        seg_T = [max((seg_pts[k] - 1) * self.dt, self.dt) for k in range(n_segs)]

        # 构造整条序列（注意拼接时避免重复点：每段除第一段外跳过第0个采样）
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
                q = clamp(q, lo, hi)
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

    def generate_trajectory(self, n_points: int, start: List[float]) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        """
        逐关节生成 quintic 序列（C2），再做 max_step 硬约束（采样级别）。
        """
        qj_all: List[List[float]] = []
        vj_all: List[List[float]] = []
        aj_all: List[List[float]] = []

        for j in range(self.num_joints):
            q, v, a = self._gen_joint_quintic(n_points, start[j], j)

            # max_step & limits hard enforcement on sampled q
            self._enforce_max_step_1d(q, self.joint_lower[j], self.joint_upper[j])

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
    def build_goal(self) -> Optional[FollowJointTrajectory.Goal]:
        start = self.get_current_positions_clamped()
        if start is None:
            return None

        n_points = random.randint(self.min_points, self.max_points)
        if n_points < 20:
            n_points = 20

        pos_seq, vel_seq, acc_seq = self.generate_trajectory(n_points, start)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = self.joint_names

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
            path_tol.name = self.joint_names[i]
            path_tol.position = 0.1
            path_tol.velocity = 0.2
            goal.path_tolerance.append(path_tol)

            goal_tol = JointTolerance()
            goal_tol.name = self.joint_names[i]
            goal_tol.position = 0.05
            goal_tol.velocity = 0.1
            goal.goal_tolerance.append(goal_tol)

        goal.goal_time_tolerance = Duration(sec=1, nanosec=0)
        return goal

    # -------- action callbacks --------
    def _goal1_response_cb(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error('[Seq] goal1 被拒绝')
            self.last_error_code_1 = None
            self.result_ready_1 = True
            return
        rf = gh.get_result_async()
        rf.add_done_callback(self._goal1_result_cb)

    def _goal1_result_cb(self, future):
        res = future.result().result
        self.last_error_code_1 = int(res.error_code)
        self.result_ready_1 = True

    def _goal2_response_cb(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error('[Seq] goal2 被拒绝')
            self.last_error_code_2 = None
            self.result_ready_2 = True
            return
        rf = gh.get_result_async()
        rf.add_done_callback(self._goal2_result_cb)

    def _goal2_result_cb(self, future):
        res = future.result().result
        self.last_error_code_2 = int(res.error_code)
        self.result_ready_2 = True

    # -------- state machine --------
    def loop(self):
        now_ns = self.get_clock().now().nanoseconds

        if self.state == SeqState.IDLE:
            if not self._have_all_joints:
                return

            if self.single:
                self.seq_id += 1
                self.state = SeqState.SEND_1
                return

            if self._next_start_ns is None:
                self._next_start_ns = now_ns
            if now_ns >= self._next_start_ns:
                self.seq_id += 1
                self.state = SeqState.SEND_1
                self._next_start_ns = now_ns + int((1.0 / max(self.freq, 1e-6)) * 1e9)

        elif self.state == SeqState.SEND_1:
            goal = self.build_goal()
            if goal is None:
                self.get_logger().warn('[Seq] 等待 joint_states...')
                return
            self.result_ready_1 = False
            self.last_error_code_1 = None
            self.get_logger().info(f'[Seq {self.seq_id}] 发送 goal1, points={len(goal.trajectory.points)}')
            f = self.action_client.send_goal_async(goal)
            f.add_done_callback(self._goal1_response_cb)
            self.state = SeqState.WAIT_RES_1

        elif self.state == SeqState.WAIT_RES_1:
            if not self.result_ready_1:
                return
            self.get_logger().info(f'[Seq {self.seq_id}] goal1 结束, error_code={self.last_error_code_1}; 等待2秒...')
            self.delay_until_ns = now_ns + int(2.0 * 1e9)
            self.state = SeqState.DELAY_2S

        elif self.state == SeqState.DELAY_2S:
            if self.delay_until_ns is None or now_ns < self.delay_until_ns:
                return
            self.state = SeqState.SEND_2

        elif self.state == SeqState.SEND_2:
            goal = self.build_goal()
            if goal is None:
                self.get_logger().warn('[Seq] 等待 joint_states...')
                return
            self.result_ready_2 = False
            self.last_error_code_2 = None
            self.get_logger().info(f'[Seq {self.seq_id}] 发送 goal2, points={len(goal.trajectory.points)}')
            f = self.action_client.send_goal_async(goal)
            f.add_done_callback(self._goal2_response_cb)
            self.state = SeqState.WAIT_RES_2

        elif self.state == SeqState.WAIT_RES_2:
            if not self.result_ready_2:
                return
            self.get_logger().info(f'[Seq {self.seq_id}] goal2 结束, error_code={self.last_error_code_2}')
            self.state = SeqState.DONE

        elif self.state == SeqState.DONE:
            if self.single:
                self.get_logger().info('single 模式：序列完成，退出')
                rclpy.shutdown()
                return
            self.state = SeqState.IDLE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freq', '-f', type=float, default=1.0,
                        help='非single模式：每秒最多启动多少个序列(序列=2个action)。默认1')
    parser.add_argument('--num-joints', '-n', type=int, default=7)
    parser.add_argument('--single', action='store_true', help='只执行一次序列(2个action)然后退出')

    parser.add_argument('--min-points', type=int, default=20, help='最小点数（建议>=20）')
    parser.add_argument('--max-points', type=int, default=1999, help='最大点数')

    parser.add_argument('--dt', type=float, default=0.1, help='点间隔时间(s)，默认0.1')
    parser.add_argument('--max-step', type=float, default=0.04, help='相邻点最大位移(rad)，默认0.04')

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
