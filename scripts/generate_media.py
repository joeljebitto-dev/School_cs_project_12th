import sys
import os
import mujoco
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.mujoco_sim import RobotSimulation
from control.pid import PIDController
from config import DEFAULT_PID_GAINS, DEFAULT_PID_TARGET_DEGREES

def main():
    print("Initializing simulation...")
    sim = RobotSimulation()
    
    # We need a renderer to get frames
    print("Setting up renderer...")
    renderer = mujoco.Renderer(sim.model, height=480, width=640)
    
    # Generate static image
    print("Generating static image...")
    # Set to some interesting pose
    sim.set_joint_angles(np.radians([30, 45, -60]))
    mujoco.mj_forward(sim.model, sim.data)
    renderer.update_scene(sim.data)
    pixels = renderer.render()
    pixels = renderer.render()
    img = Image.fromarray(pixels)
    img.save("robot_arm.png")
    print("Saved robot_arm.png")

    # Generate GIF
    print("Generating GIF...")
    sim.set_joint_angles(np.radians([0, 0, 0]))
    
    kp = np.array([DEFAULT_PID_GAINS[f"joint{i+1}"][0] for i in range(3)])
    ki = np.array([DEFAULT_PID_GAINS[f"joint{i+1}"][1] for i in range(3)])
    kd = np.array([DEFAULT_PID_GAINS[f"joint{i+1}"][2] for i in range(3)])
    pid = PIDController(kp, ki, kd)
    
    target_angles = np.radians(DEFAULT_PID_TARGET_DEGREES)
    
    frames = []
    # simulate for 2 seconds (2.0 / 0.005 = 400 steps)
    # capture frame every 10 steps (0.05s / 20 fps)
    for step in range(400):
        # We step simulation manually instead of sim.step_pid_control to get finer control or just use it
        error = target_angles - sim.data.qpos[:3]
        torque = pid.compute(target_angles, sim.data.qpos[:3], sim.data.qvel[:3], 0.005)
        sim.data.ctrl[:3] = torque
        mujoco.mj_step(sim.model, sim.data)
        
        if step % 10 == 0:
            renderer.update_scene(sim.data)
            pixels = renderer.render()
            frames.append(Image.fromarray(pixels))
    
    frames[0].save(
        "robot_motion.gif",
        save_all=True,
        append_images=frames[1:],
        duration=50, # ms per frame
        loop=0
    )
    print("Saved robot_motion.gif")

if __name__ == "__main__":
    main()
