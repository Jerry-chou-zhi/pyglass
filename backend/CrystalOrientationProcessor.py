import py4DSTEM
import numpy as np
import pickle
import cv2

class CrystalOrientationProcessor:
    __slots__ = [
        "datacube",
        "_dp_mean",
        "_dp_max",
        "_bragg_peaks",
        "_bragg_peaks_fit",	
        "_orientation",
        "_orientation_map",
        "_crystal",
        "_orientation_images",
        "_images_orientation",
        "_plot_diffraction_pattern_params",
        "_plot_orientation_params",
        "_k_max",
        "_pixel_size_inv_Ang_guess",
        "_bragg_k_power",
        "_plot_result",
        "_figsize",
        "_angle_step_zone_axis",
        "_angle_step_in_plane",
        "_sigma_excitation_error",
        "_zone_axis_range",
        "_xind",
        "_yind",
        "_sigma_compare",
        "_range_plot",
        "_ind_orientation",
        "_sigma_excitation_error",
        "_bragg_peaks_compare",
        "_img",
        "_in_plane",
        "_out_plane"
    ]

    def load_data(self, file_path: str):
        """加载4D-STEM数据并初始化参数"""
        self.datacube = py4DSTEM.import_file(filepath=file_path)
        self._dp_mean = self.datacube.get_dp_mean()
        self._dp_max = self.datacube.get_dp_max()
        self._bragg_peaks = None
        self._orientation = None
        self._crystal = None
        self._orientation_map = None
        self._orientation_images = None
        self._plot_diffraction_pattern_params = {
            "scale_markers": 1000,
            "scale_markers_compare": 100,
            "plot_range_kx_ky": None,
            "min_marker_size": 50,
            "figsize": (5, 5)
        }
        self._plot_orientation_params = {
            "orientation_ind": 0,
            "corr_range": [0.9, 1.0],
            "camera_dist": 0,
            "show_axes": True,
            "corr_normalize": 1,
            "figsize": (16, 40),
            "swap_axes_xy_limits": True,
            "plot_layout": 0
        }
        self._k_max = 2
        self._pixel_size_inv_Ang_guess = 0.0169
        self._bragg_k_power = 2.0
        self._plot_result = True
        self._figsize = (8, 3)
        self._angle_step_zone_axis = 1
        self._angle_step_in_plane = 6.0
        self._sigma_excitation_error = 0.04
        self._zone_axis_range = 'auto'
        self._xind = None 
        self._yind = None
        self._sigma_compare = 0.03
        self._range_plot = None
        self._ind_orientation = 0
        self._sigma_excitation_error = 0.04
        self._bragg_peaks_fit = None
        self._bragg_peaks_compare = None
        self._images_orientation = None
        self._img = None
        self._in_plane = None
        self._out_plane = None
        
    def get_orientation_planes_plot(self, in_plane_path: str = "in_plane.png", out_plane_path: str = "out_plane.png"):
    """保存取向的平面图像
    
    Args:
        in_plane_path (str): 平面内取向图像的保存路径，默认为"in_plane.png"
        out_plane_path (str): 平面外取向图像的保存路径，默认为"out_plane.png"
    """
        if self._in_plane is None or self._out_plane is None:
            raise ValueError("平面图像数据未生成，请先调用get_in_out_plane方法")
        
        import cv2
        cv2.imwrite(in_plane_path, self._in_plane)
        cv2.imwrite(out_plane_path, self._out_plane)

    def get_in_out_plane(self):
        self._in_plane = np.floor(np.clip(np.squeeze(self._img[:,:, 2:-1:0, 0]) * 256, 0, 255)).astype(np.uint8)
        self._out_plane = np.floor(np.clip(np.squeeze(self._img[:,:, 2:-1:0, 1]) * 256, 0, 255)).astype(np.uint8)

    def get_img(self):
        self._img = np.array(self._images_orientation)
    
    def get_images_orientation(self):
        self._images_orientation = self._crystal.plot_orientation_maps(
            self._orientation_map,
            **self._plot_orientation_params
        )
        fig = self._images_orientation
        return fig[0]

    def get_orientation_map(self):
        print(self._orientation_map)

    def load_orientation_map(self, file_path: str):
        with open(file_path, 'rb') as f:
            self._orientation_map = pickle.load(f)

    def set_orientation_map(self):
        self._orientation_map = self._crystal.match_orientations(
            self._bragg_peaks,
        )

    def get_plot_diffraction_pattern_params(self):
        fig = py4DSTEM.process.diffraction.plot_diffraction_pattern(
            self._bragg_peaks_fit,
            bragg_peaks_compare=self._bragg_peaks_compare,
            **self._plot_diffraction_pattern_params
        )
        return fig[0]

    def set_bragg_peaks_compare(self):
        """设置用于对比的布拉格峰
        
        Args:
            bragg_peaks: 布拉格峰数据对象
            x_index: x方向索引
            y_index: y方向索引
        """
        self._bragg_peaks_compare = self._bragg_peaks.cal[self._xind, self._yind]

    def set_probe_position(self, xind: int, yind: int):
        """设置探针位置坐标
        
        Args:
            xind: x方向的索引位置
            yind: y方向的索引位置
        """
        self._xind = xind
        self._yind = yind

    def get_scattering_intensity(self):
        """使用初始化的参数进行校准"""
        if self._bragg_peaks is not None and self._crystal is not None:
            self._bragg_peaks.calibration.set_Q_pixel_size(self.pixel_size_inv_Ang_guess)
            self._bragg_peaks.calibration.set_Q_pixel_units('A^-1')
            self._bragg_peaks.setcal()

            # 显示叠加图
            fig = self._crystal.plot_scattering_intensity(
                bragg_peaks=self._bragg_peaks,
                bragg_k_power=self._bragg_k_power
            )
            return fig[0]
        else:
            raise ValueError("Bragg peaks or crystal data is not loaded.")


    def load_crystal(self, positions=None, intensities=None, lattice_params=None):
        """
        加载晶体文件并返回一个Crystal对象。
        
        Args:
            positions (np.ndarray): 晶体位置数组，默认为 [[0, 0, 0]].
            intensities (np.ndarray): 晶体强度数组，默认为 [46].
            lattice_params (np.ndarray): 晶体格子参数数组，默认为 [2.79806820, 2.79806820, 2.79806820, 60, 60, 60].
        
        Returns:
            Crystal: 初始化的晶体对象。
        """
        if positions is None:
            positions = np.array([[0, 0, 0]])
        if intensities is None:
            intensities = np.array([46])
        if lattice_params is None:
            lattice_params = np.array([2.79806820, 2.79806820, 2.79806820, 60, 60, 60])
        
        self._crystal = py4DSTEM.process.diffraction.Crystal(
            positions,
            intensities,
            lattice_params
        )


    def cal_structure_factors(self):
        """计算晶体的结构因子"""
        if self._crystal is not None:
            self._crystal.calculate_structure_factors(self.k_max)
        else:
            raise ValueError("Crystal object is not loaded.")

    def calibrate_pixel_size(self):
        """
        校准像素大小并可选择性地显示结果。
        
        Returns:
            bragg_peaks_cali: 校准后的 bragg peaks 对象
        """
        if self._bragg_peaks is None or self._crystal is None:
            raise ValueError("Bragg peaks 或 crystal 数据未加载。")
        
        # 校准像素大小
        bragg_peaks_cali = self._crystal.calibrate_pixel_size(
            bragg_peaks=self._bragg_peaks,
            bragg_k_power=self._bragg_k_power,
            plot_result=self._plot_result,
            figsize=self._figsize
        )
        
        # 更新校准后的 bragg peaks
        self._bragg_peaks = bragg_peaks_cali
        

    def get_initial_pixel_size(self):
        """
        获取当前像素大小和单位。
        
        Returns:
            tuple: (pixel_size, units)
        """
        if self._bragg_peaks is None:
            raise ValueError("Bragg peaks 数据未加载。")
        
        size = self._bragg_peaks.calibration.get_Q_pixel_size()
        units = self._bragg_peaks.calibration.get_Q_pixel_units()
        print(f'   Initial pixel size = {np.round(size,8)} {units}')

    def get_crystal_3d_structure(self, zone_axis=(1,0.8,0.5), figsize=(6,3), camera_dist=6):
        """
        绘制晶体的3D结构图
        
        参数:
        crystal: Crystal对象
        zone_axis: 晶带轴方向, 默认(1,0.8,0.5)
        figsize: 图形大小, 默认(6,3)
        camera_dist: 相机距离, 默认6
        
        返回:
        matplotlib figure对象
        """
        fig = self._crystal.plot_structure(
            zone_axis_lattice=zone_axis,
            figsize=figsize,
            camera_dist=camera_dist
        )
        return fig[0]

    def get_crystal_structure_factors(self, zone_axis=[1,1,1], figsize=(4,4)):
        """
        绘制晶体的结构因子图
        
        参数:
        crystal: Crystal对象
        zone_axis: 晶带轴方向, 默认[1,1,1]
        figsize: 图形大小, 默认(4,4)
        
        返回:
        matplotlib figure对象
        """
        fig = self._crystal.plot_structure_factors(
            zone_axis_lattice=zone_axis,
            figsize=figsize
        )
        return fig[0]
    
    def set_orientation_plan(self):
        self._crystal.orientation_plan(
            angle_step_zone_axis=self._angle_step_zone_axis,
            angle_step_in_plane=self._angle_step_in_plane,
            sigma_excitation_error=self._sigma_excitation_error,
            zone_axis_range=self._zone_axis_range
        )
        

    def set_orientation(self, plot_corr=False, plot_polar=False, verbose=True):
        """
        通过匹配衍射图案来设置晶体取向
        
        参数:
        plot_corr: bool, 是否绘制相关性图
        plot_polar: bool, 是否绘制极坐标图
        verbose: bool, 是否打印详细信息
        
        返回:
        orientation: 匹配得到的取向对象
        """
        if self.xind is None or self.yind is None:
            raise ValueError("请先使用set_probe_position设置探针位置")
        
        if self._crystal is None or self._bragg_peaks is None:
            raise ValueError("晶体数据或衍射峰数据未加载")
        
        self._orientation = self._crystal.match_single_pattern(
            self._bragg_peaks.cal[self._xind, self._yind],
            plot_corr=plot_corr,
            plot_polar=plot_polar,
            verbose=verbose
        )
        
        

    def set_range_plot(self, margin=0.1):
        """设置衍射图案的绘图范围
        
        Args:
            margin (float): 在k_max基础上增加的边距值，默认0.1
            
        Returns:
            np.ndarray: 包含[x_range, y_range]的数组
        """
        range_value = self._k_max + margin
        self._range_plot = np.array([range_value, range_value])

    def get_bragg_peaks_fit(self):
        """
        生成衍射图样的布拉格峰
        
        参数:
            ind_orientation: 取向索引,默认为0
            sigma_excitation_error: 激发误差,默认为0.04
            
        返回:
            bragg_peaks_fit: 拟合后的布拉格峰
        """
        if self._orientation is None:
            raise ValueError("Orientation is not set. Please set the orientation first.")
        
        self._bragg_peaks_fit = self._crystal.generate_diffraction_pattern(
            self._orientation,
            ind_orientation=self._ind_orientation, 
            sigma_excitation_error=self._sigma_excitation_error
        )
        
        print(self._bragg_peaks_fit)  # 打印拟合结果
        
