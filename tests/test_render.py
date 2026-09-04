"""Smoke-test SAPIEN renderer initialization."""

import warnings

import gymnasium as gym
import sapien.core as sapien
import toppra as ta


warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


class SapienRenderSmokeTest(gym.Env):

    def __init__(self):
        super().__init__()
        ta.setup_logging("CRITICAL")  # hide logging
        self.setup_scene()

    def setup_scene(self, **kwargs):
        """
        Set the scene
            - Set up the basic scene: light source, viewer.
        """
        self.engine = sapien.Engine()
        # declare sapien renderer
        from sapien.render import set_global_config

        set_global_config(max_num_materials=50000, max_num_textures=50000)
        self.renderer = sapien.SapienRenderer()
        # give renderer to sapien sim
        self.engine.set_renderer(self.renderer)

        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")

        # declare sapien scene
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)


if __name__ == "__main__":
    SapienRenderSmokeTest()
    print("\033[32mRender Well\033[0m")
