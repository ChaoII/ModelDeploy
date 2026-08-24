# WorkoutMonitor — 健身动作计数

> 架构层：`modeldeploy::vision::solution`
> 源码：`csrc/vision/solutions/workout_monitor.h` / `.cpp`

## 职责

基于关键点**夹角阈值的状态机**，对健身动作（如下蹲 / 深蹲、弯举）进行**重复次数计数**。配合姿态 / 关键点检测模型使用。

## 应用场景

- 健身 / 康复动作计次（深蹲、弯举等以"关节角度先收后放"为特征的动作）
- 实时运动计数（每帧喂夹角，累积次数）
- 也暴露 C API（`MD_SOLUTION_WORKOUT`）与 Python 绑定，便于集成上层应用

## 输入输出

- 输入：当前关节角（度），由 `angle()` 用三个关键点算出或外部提供
- 输出：累计次数 `int reps()`

## 关键 API

```cpp
using modeldeploy::vision::solution::WorkoutMonitor;

WorkoutMonitor wm(float min_deg = 70.0f, float max_deg = 160.0f);  // 下蹲/伸直阈值
static float WorkoutMonitor::angle(Point3f a, Point3f b, Point3f c); // 以 b 为顶点的夹角(度)
int wm.reps() const;
void wm.update(float elbow_deg);   // 每帧喂入角度
wm.reset();
```

## 原理与算法

```cpp
// 夹角：仅用 x,y 二维投影，b 为顶点；向量退化时返回 180°(伸直)
float angle(a, b, c) {
    dot = (a-b)·(c-b);
    n1 = |a-b|, n2 = |c-b|;
    if (n1<=0 || n2<=0) return 180.0f;
    return acos(clamp(dot/(n1*n2), -1, 1)) * 180 / π;
}
// 状态机
void update(float deg) {
    if (deg < min_)      down_ = true;                             // 下蹲
    else if (deg > max_ && down_) { ++reps_; down_ = false; }      // 伸直 → 记一次
}
```

有向状态机：必须先进入 `down`（角度 < `min_deg`），再回升到 `> max_deg` 才计数一次；连续处于同一状态不重复计数。

## 典型用法

C++：

```cpp
WorkoutMonitor wm(70.0f, 160.0f);
auto& pose = pose_model.predict(img);                  // 姿态模型
float deg = WorkoutMonitor::angle(shoulder, elbow, wrist);
wm.update(deg);                                        // 每帧
int count = wm.reps();
```

Python：

```python
from modeldeploy import WorkoutMonitor
wm = WorkoutMonitor(min_deg=70.0, max_deg=160.0)
deg = WorkoutMonitor.angle(Point3f(0,0,0), Point3f(1,1,0), Point3f(2,0,0))
wm.update(deg)
n = wm.reps()
```

## 效果

仅依赖角度阈值即可稳定计次，实时性高，适合健身 / 康复场景。
