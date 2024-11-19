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
        "_kernel",  # 探针核
        "_probe_params",  # 探针的初始化参数
        "_disk_detect_params",  # Bragg盘检测的参数
    ]
    def set_probe_xlims_ylims(self, xlims: tuple, ylims: tuple):
        self._probe_xlims = xlims
        self._probe_ylims = ylims
        self._probe_ROI = np.zeros(self.datacube.rshape, dtype=bool)
        self._probe_ROI[
            self._probe_xlims[0] : self._probe_xlims[1],
            self._probe_ylims[0] : self._probe_ylims[1],
        ] = True
   
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
        
    def create_orientation_plan(self, params):
        """
        Create an orientation plan for crystal alignment.

        Args:
            params (dict): Parameters for orientation plan generation.
        """
        self.orientation_plan = self.datacube.create_orientation_plan(**params)

    def execute_orientation_plan(self):
        """
        Execute the orientation plan to compute orientations.
        """
        if self.orientation_plan is None:
            raise ValueError("Orientation plan has not been created.")
        self.orientation_map = self.orientation_plan.execute()

    def visualize_orientation_map(self, orientation_ind=0, show_axes=False):
        """
        Visualize the computed orientation map.

        Args:
            orientation_ind (int): Index of the orientation to visualize.
            show_axes (bool): Whether to show axes in the visualization.
        """
        if self.orientation_map is None:
            raise ValueError("Orientation map has not been computed.")
        fig = py4DSTEM.visualize.plot_orientation_map(
            self.orientation_map, orientation_ind=orientation_ind, show_axes=show_axes, returnfig=True
        )
        return fig

    def compute_strain_map(self, strain_params):
        """
        Compute the strain map based on orientation data.

        Args:
            strain_params (dict): Parameters for strain calculation.
        """
        if self.orientation_map is None:
            raise ValueError("Orientation map has not been computed.")
        self.strain_map = py4DSTEM.StrainMap(braggvectors=self.orientation_map, **strain_params)

    def visualize_strain_map(self):
        """
        Visualize the computed strain map.
        """
        if self.strain_map is None:
            raise ValueError("Strain map has not been computed.")
        fig = py4DSTEM.visualize.plot_strain_map(self.strain_map, returnfig=True)
        return fig

    def extract_grain_boundary_data(self, params):
        """
        Extract grain boundary data based on the orientation map.

        Args:
            params (dict): Parameters for grain boundary extraction.
        """
        if self.orientation_map is None:
            raise ValueError("Orientation map has not been computed.")
        self.grain_boundaries = self.dataset.extract_grain_boundaries(self.orientation_map, **params)
        return self.grain_boundaries

    def visualize_grain_boundaries(self):
        """
        Visualize the extracted grain boundaries.
        """
        if self.grain_boundaries is None:
            raise ValueError("Grain boundaries have not been extracted.")
        fig = py4DSTEM.visualize.plot_grain_boundaries(self.grain_boundaries, returnfig=True)
        return fig

    def save_results(self, save_path):
        """
        Save the computed results to a file.

        Args:
            save_path (str): Path to save the results.
        """
        if self.orientation_map is None:
            raise ValueError("No orientation map to save.")
        py4DSTEM.save(save_path, self.orientation_map, mode="o")

    def create_virtual_image(self, mode, geometry, name):
        """
        Create a virtual image (bright or dark field).

        Args:
            mode (str): "circle" for BF, "annulus" for ADF.
            geometry (tuple): Geometry definition for virtual image.
            name (str): Name of the virtual image.
        """
        if self.datacube is None:
            raise ValueError("Datacube has not been loaded.")
        return self.datacube.get_virtual_image(mode=mode, geometry=geometry, name=name)

    def detect_bragg_disks(self, template, detect_params):
        """
        Detect Bragg disks in the diffraction data.

        Args:
            template: Template for disk detection.
            detect_params (dict): Parameters for disk detection.
        """
        if self.datacube is None:
            raise ValueError("Dataset has not been loaded.")
        self.bragg_disks = self.datacube.find_Bragg_disks(template=template, **detect_params)

    def visualize_bragg_disks(self, rxs, rys, colors, scale=300):
        """
        Visualize detected Bragg disks on selected diffraction patterns.

        Args:
            rxs (list): List of x-indices for visualization.
            rys (list): List of y-indices for visualization.
            colors (list): Colors for marking the disks.
            scale (int): Scale for visualization.
        """
        fig = py4DSTEM.visualize.show_image_grid(
            get_ar=lambda i: self.datacube.data[rxs[i], rys[i], :, :],
            H=2,
            W=3,
            get_bordercolor=lambda i: colors[i],
            get_x=lambda i: self.bragg_disks[i].data["qx"],
            get_y=lambda i: self.bragg_disks[i].data["qy"],
            get_pointcolors=lambda i: colors[i],
            open_circles=True,
            scale=scale,
            returnfig=True,
        )
        return fig

    def fit_ellipse_amorphous_ring(self, amorphous_diffraction, fit_radii):
        """
        Fit an ellipse to an amorphous diffraction ring.

        Args:
            amorphous_diffraction: Amorphous diffraction pattern.
            fit_radii (tuple): Radii range for ellipse fitting.
        """
        center = amorphous_diffraction.calibration.get_origin_mean()
        self.p_ellipse, self.p_dsg = py4DSTEM.process.calibration.fit_ellipse_amorphous_ring(
            data=amorphous_diffraction.data, center=center, fitradii=fit_radii
        )

    def visualize_ellipse_fit(self, amorphous_diffraction, fit_radii):
        """
        Visualize the ellipse fit on the amorphous diffraction ring.

        Args:
            amorphous_diffraction: Amorphous diffraction pattern.
            fit_radii (tuple): Radii range for ellipse fitting.
        """
        fig = py4DSTEM.visualize.show_amorphous_ring_fit(
            amorphous_diffraction.data,
            fitradii=fit_radii,
            p_dsg=self.p_dsg,
            returnfig=True,
        )
        return fig

