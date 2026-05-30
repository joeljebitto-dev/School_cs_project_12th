# Report: 3D Robot Arm Kinematics and PID Control

## 1. Aim

The aim of this project is to demonstrate three core robotics ideas:

1. Forward kinematics: find the end-effector position from known joint angles.
2. Inverse kinematics: find joint angles that reach a desired position.
3. PID control: move simulated robot joints smoothly toward target angles.

The project uses a 3D 3-DOF arm in MuJoCo. Tkinter is used for input and for
displaying the math results. Every slider also has a numeric input box, so the
demo supports both quick dragging and exact typed values.

## 2. Robot Model

The robot has three revolute joints:

```text
q1 = base yaw about the world z-axis
q2 = shoulder pitch
q3 = elbow pitch
```

The link lengths are:

```text
L1 = 0.45 m  upper arm
L2 = 0.35 m  forearm
L3 = 0.25 m  tool extension
```

The shoulder is raised above the floor:

```text
base_height = 0.18 m
```

Because the robot has three degrees of freedom, it controls Cartesian position:

```text
position = [x, y, z]
```

It does not control full end-effector orientation. A real robot usually needs
extra wrist joints for that.

For a clean classroom demonstration, the app uses natural motion limits:

```text
q1 yaw      = -180 deg to 180 deg
q2 shoulder =    0 deg to 110 deg
q3 elbow    = -135 deg to   0 deg
```

The moving robot points must also remain at least `0.02 m` above the floor.
This prevents the end effector or links from being driven into visually
unnatural below-floor positions.

## 3. Forward Kinematics

Forward kinematics answers this question:

```text
If q1, q2, and q3 are known, where is the end effector?
```

The shoulder and elbow determine the vertical height and horizontal reach.
The yaw joint rotates that horizontal reach around the z-axis.

First compute the horizontal reach:

```text
r = L1*cos(q2) + (L2 + L3)*cos(q2 + q3)
```

Then rotate that reach by yaw:

```text
x = r*cos(q1)
y = r*sin(q1)
```

Finally compute height:

```text
z = base_height + L1*sin(q2) + (L2 + L3)*sin(q2 + q3)
```

This is implemented in `forward_kinematics()` in `kinematics/forward.py`.

## 4. Jacobian

The Jacobian describes how small joint changes affect the end-effector
position:

```text
dposition = J(q) * dq
```

For this robot:

```text
r  = L1*cos(q2) + (L2 + L3)*cos(q2 + q3)
dr/dq2 = -L1*sin(q2) - (L2 + L3)*sin(q2 + q3)
dr/dq3 =              - (L2 + L3)*sin(q2 + q3)
```

The partial derivatives are:

```text
dx/dq1 = -r*sin(q1)
dx/dq2 = dr/dq2*cos(q1)
dx/dq3 = dr/dq3*cos(q1)

dy/dq1 =  r*cos(q1)
dy/dq2 = dr/dq2*sin(q1)
dy/dq3 = dr/dq3*sin(q1)

dz/dq1 = 0
dz/dq2 = L1*cos(q2) + (L2 + L3)*cos(q2 + q3)
dz/dq3 =              (L2 + L3)*cos(q2 + q3)
```

This is implemented in `jacobian()` in `kinematics/forward.py`.

## 5. Inverse Kinematics

Inverse kinematics answers this question:

```text
What joint angles should the robot use to reach a target [x, y, z]?
```

The project uses an iterative numerical method. First it computes:

```text
position_error = target_position - current_position
```

For reachable targets, the solver first computes the two geometric elbow
branches, rejects unsafe candidates, and chooses the branch with the higher
elbow. This avoids a confusing solution where the shoulder points downward and
the elbow folds back up even though the end-effector position is correct.

Then it uses the Jacobian to estimate a joint update:

```text
position_error ~= J(q) * dq
```

Instead of directly inverting the Jacobian, the project uses damped least
squares:

```text
dq = alpha * J.T * inv(J*J.T + lambda^2*I) * position_error
```

where:

```text
alpha   = step size
lambda  = damping value
I       = identity matrix
```

Damping is useful because robot arms can enter singular positions where a
normal inverse becomes unstable.

This is implemented in `inverse_kinematics()` in `kinematics/inverse.py`.

## 6. PID Control

PID control moves the robot joints toward target joint angles. For each joint:

```text
error = target_angle - current_angle
```

The torque command is:

```text
torque = Kp*error + Ki*integral(error) + Kd*derivative(error)
```

The terms mean:

```text
Kp: pushes harder when the error is large
Ki: removes small long-term steady errors
Kd: resists fast motion and reduces overshoot
```

In this project the target is fixed, so:

```text
derivative(error) ~= -joint_velocity
```

The controller also limits torque and clamps the integral value. This prevents
the controller from producing unrealistic forces.

This is implemented in `PIDController` in `control/pid.py`.

The UI has a `Use PID Motion` toggle:

```text
PID off: FK sliders move the robot directly in real time.
PID on:  FK Apply appears and starts smooth PID motion toward the target.
PID tab: target inputs move the robot with PID and gains update live.
IK:      target sliders/boxes move the marker; Apply is always required.
```

## 7. MuJoCo Simulation

MuJoCo simulates the robot dynamics:

- Joint 1 is a yaw hinge about `z`.
- Joints 2 and 3 are pitch hinges about local `-y`.
- The `-y` pitch axis makes positive pitch raise the arm in `+z`.
- Each joint has one motor actuator.
- Joint limits match the safe/natural demo limits.
- Gravity is disabled so the project focuses on the kinematics and controller.
- A red target marker shows the IK target position.

The model is created in `simulation/mujoco_sim.py`. The `RobotSimulation`
class in the same file owns the MuJoCo model, data, viewer, and the small
stepping loop used by Tkinter.

## 8. Testing

The project includes tests for:

- FK at zero angles.
- FK yaw rotation into the y-axis.
- FK shoulder pitch raising the arm in z.
- Jacobian correctness using finite differences.
- IK convergence for a reachable target.
- IK failure reporting for an unreachable target.
- Rejection of shoulder-down, back-folding elbow, and below-floor poses.
- IK rejection when a target is only reachable through an unsafe posture.
- PID torque limiting and integral clamping.
- MuJoCo model loading and stepping.
- MuJoCo end-effector site position matching `forward_kinematics()`.
- The `RobotSimulation` wrapper setting poses and stepping PID safely.

## 9. Screenshots

Add screenshots here after running the project:

```text
Screenshot 1: Forward Kinematics tab with joint sliders
Screenshot 2: Inverse Kinematics tab with target sliders and target marker
Screenshot 3: PID Control tab with target/gain sliders and error/torque plot
Screenshot 4: MuJoCo viewer showing the 3D arm
```

## 10. Conclusion

This project shows how the same robot can be understood in three layers:

1. Geometry gives forward kinematics.
2. The Jacobian gives inverse kinematics.
3. PID control turns target joint angles into simulated motor torques.

Keeping the math separate from the GUI makes the project easier to test,
explain, and extend.
