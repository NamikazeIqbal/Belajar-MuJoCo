# -*- coding: utf-8 -*-
"""
Created on fri Jan 13 20:16:59 2023

@author: Muhammad Iqbal Maula
"""

import mujoco as mj
import numpy as np
from mujoco.glfw import glfw

from mujoco_base import MuJoCoBase


class ControlArmLetoy(MuJoCoBase):
    def __init__(self, xml_path):
        super().__init__(xml_path)

    def reset(self):
        # Set initial angle of pendulum
        self.data.qpos[0] = np.pi/2

        # Set camera configuration
        self.cam.azimuth = 90.0
        self.cam.distance = 5.0
        self.cam.elevation = -5
        self.cam.lookat = np.array([0.012768, -0.000000, 1.254336])

        mj.set_mjcb_control(self.controller)

    def controller(self, model, data):
     
        self.data.ctrl[2] = 5 * (2 - self.data.sensordata[0]) + 2 * self.data.sensordata[1]
        self.data.ctrl[5] = 3 * (2 - self.data.sensordata[2]) + 2 * self.data.sensordata[1]
        
    def simulate(self):
        while not glfw.window_should_close(self.window):
            simstart = self.data.time

            while (self.data.time - simstart < 1.0/60.0):
                # Step simulation environment
                mj.mj_step(self.model, self.data)

            # get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(
                self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

            # Update scene and render
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)

            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()

        glfw.terminate()


def main():
    xml_path = "./model/arm_letoy.xml"
    sim = ControlArmLetoy(xml_path)
    sim.reset()
    sim.simulate()


if __name__ == "__main__":
    main()
