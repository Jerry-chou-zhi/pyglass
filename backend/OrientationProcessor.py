import py4DSTEM
import numpy as np


class OrientationProcessor:
    __slots__ = [
        "datacube",  
        "datacube_probe",
        "_dp_mean",
        "_probe_dp_mean",
        "_dp_max",
        "_geometry_adf",
        "_orientation_plan",  # 取向计划对象
        "_orientation_map",  # 计算生成的取向映射
        "_strain_map",  # 应变映射对象
        "_grain_boundaries",  # 晶界数据
        "_rxs", 
        "_rys", 
        "_colors", 
        "_guess_center",
        "_radial_range",
        "_orientation_params",  # 用于创建取向计划的参数
        "_strain_params",  # 用于计算应变的参数
        "_grain_boundary_params",  # 用于提取晶界的参数
        "_ellipse_fit_range",  # 用于拟合椭圆的范围
        "_bragg_peaks",  # Bragg峰数据
        "_bvm_raw",  # 原始Bragg矢量图
        "_bvm_centered",  # 对齐中心后的Bragg矢量图
        "_amorph_ROI",  # 无序环区域
        "_p_ellipse",  # 椭圆拟合结果参数
        "_p_dsg",  # 椭圆拟合图形化参数
        "_adf", 
        "_bf", 
        "_probe_ROI",
        "_probe", 
        "_probe_semiangle",
        "_probe_qx0",
        "_probe_qy0",
        "_kernel", 
        "_probe_params",  # 探针的初始化参数
        "_disk_detect_params",  
    ]
   
    def (self, mode: str = "raw", figsize: tuple = (4, 4), alpha: float = 0.3, fill: bool = True):
        bragg_vector_map = self._bragg_peaks.get_bvm(mode=mode)
    
        py4DSTEM.show(
            bragg_vector_map,
            annulus={
                'center': self._guess_center,  
                'radii': self._radial_range,  
                'alpha': alpha,
                'fill': fill,
            },
            figsize=figsize,
        )
    
    def set_guess_center(self, center: tuple):
        self._guess_center = center
        self._update_geometry_adf()
        
    def set_radial_range(self, radial: tuple):
        self._radial_range = radial
        self._update_geometry_adf()

    def _update_geometry_adf(self):
        self._geometry_adf = (self._guess_center,  self._radial_range)

         
    def save_bragg_disks(self, filepath: str): #新加的，好像是必要的
        py4DSTEM.save(
            filepath,
            self._bragg_peaks,  
            mode="o",
        )
    
    def calc_all_bragg_disks(self):
        self._bragg_peaks = self.datacube.find_Bragg_disks(
            template=self._probe.kernel,
            **self._disk_detect_params,
        )
         
    def get_origin_grid_image(self):
        fig = py4DSTEM.visualize.show_image_grid(
            get_ar=lambda i: self.datacube.data[self._rxs[i], self._rys[i], :, :],
            H=2,
            W=3,
            scaling="log",
            vmin=0.02,   #确认
            vmax=1,
            get_bordercolor=lambda i: self._colors[i],
            returnfig=True,
        )
        return fig[0]
        
    def get_selected_bragg_disks_image(self):
        disks_selected = self.datacube.find_Bragg_disks(
            data=(self._rxs, self._rys),
            template=self._probe.kernel,
            **self._disk_detect_params,
        )    
    def set_disk_detect_params(self, params: dict):
        self._disk_detect_params = params
        
    def get_kernel_image(self):
        fig = py4DSTEM.visualize.show_kernel(
            self._probe.kernel,
            R=20,
            L=20,
            W=1,
            figsize = (8,4),
            returnfig=True,
        )
        return fig[0]
        
    def set_probe(self):
        self._probe = self.datacube.get_vacuum_probe(ROI=self._probe_ROI) #没改
        self._probe_semiangle, self._probe_qx0, self._probe_qy0 = (
            py4DSTEM.process.calibration.get_probe_size(self._probe.probe)#没改
        )
        self._probe.get_kernel(
            mode="sigmoid",
            #原来的origin删了
            radii=(self._probe_semiangle * 0, self._probe_semiangle * 4), #改动
            bilinear = True, #改动
        )
  
    def get_points_image(self):
        fig = py4DSTEM.visualize.show_points(
            self._adf,
            x=self._rxs,
            y=self._rys,
            pointcolor=self._colors,
            returnfig=True,
        )
        return fig[0]
    
    def set_rxs_rys(self, rxs: list, rys: list):
        self._rxs = rxs
        self._rys = rys
         
    def get_adf_image(self):
        fig = show([self._adf], bordercolor="w", cmap="gray", returnfig=True)
        return fig[0]
    
    def set_geometry_adf(self, center: tuple, radii: tuple):
        self._geometry_adf = (center, radii)
        
    def get_position_detector_image(self):
        fig = self.datacube.position_detector(
            data=self._dp_max,
            mode="annular",
            geometry=self._geometry_adf,
            returnfig=True,
        )
        return fig[0]
        
    def calc_adf(self):
        self._adf = self.datacube.get_virtual_image(
            mode="annulus",
            geometry=self._geometry_adf,
            name="dark_field",
        )
        
    def calibrate_diffraction_space(self, pixel_size_inv_ang, units="A^-1"):
        self.datacube.calibration.set_Q_pixel_size(pixel_size_inv_ang)
        self.datacube.calibration.set_Q_pixel_units(units)
        self.datacube_probe.calibration.set_Q_pixel_size(pixel_size_inv_ang)
        self.datacube_probe.calibration.set_Q_pixel_units(units)

    def calibrate_real_space(self, pixel_size_ang, units="A"):
        self.datacube.calibration.set_R_pixel_size(pixel_size_ang)
        self.datacube.calibration.set_R_pixel_units(units)

    def get_dp_image(self, vmax=1, power=0.5, cmap="inferno"):
        fig = py4DSTEM.show(
            [
                self._dp_mean,
                self._dp_max,
                self._probe_dp_mean,
            ],
            vmax=vmax,
            power=power,
            cmap=cmap,
            returnfig=True,
        )
        return fig[0]  
        
    def load_data(self, data_file_path: str, probe_file_path: str):      
        self.datacube = py4DSTEM.import_file(filepath=data_file_path)
        self._dp_mean = self.datacube.get_dp_mean()
        self._dp_max = self.datacube.get_dp_max()
        self.datacube_probe = py4DSTEM.import_file(filepath=probe_file_path)
        self._probe_dp_mean = self.datacube_probe.get_dp_mean()
        self._geometry_adf = None
        self._bf = None
        self._adf = None
        self._rxs = None
        self._rys = None
        self.orientation_plan = None
        self.orientation_map = None
        self.strain_map = None
        self.grain_boundaries = None
        self._colors = [
            "deeppink",
            "coral",
            "gold",
            "chartreuse",
            "dodgerblue",
            "rebeccapurple",
        ]
        self._probe = None
        self._probe_ROI = None
        self._probe_semiangle = 4 #改动
        self._probe_qx0 = None
        self._probe_qy0 = None
        self._disk_detect_params = None
        self._guess_center = None
        self._radial_range = None

